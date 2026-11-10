"""Alert triage: rank alerts so investigators see the likely-real ones first.

The rules in `monitor.py` catch every injected case and also flag legitimate
customers who look similar (home purchases, small businesses, cash businesses).
Triage scores each alert with a small per-typology logistic regression, trained on
past alert outcomes (here, the generator's ground truth stands in for investigator
dispositions), and suppresses alerts below a threshold chosen to keep recall at 1.0
on the training data.

Features use only what a bank would have when the alert fires: the account's own
history before the alert, and how many *other* customers have paid the same
counterparty. They never use counterparty names or ids directly. The synthetic
generator names its counterparties by kind (TITLE01, BRK.., CUS..), so a model
allowed to read ids would learn the generator, not the behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

from .models import FAN_IN, PASS_THROUGH, STRUCTURING, TYPOLOGIES, Alert, Transaction

LOOKBACK = timedelta(days=30)

FEATURE_NAMES = {
    STRUCTURING: ["prior_cash_deposits_30d", "deposits_in_pattern", "total_over_ctr"],
    PASS_THROUGH: ["out_legs", "dest_paid_by_other_customers", "days_to_first_out", "out_ratio"],
    FAN_IN: ["prior_distinct_senders_30d", "distinct_senders", "mean_amount_k"],
}


class History:
    """Append-only index over a transaction stream, queried as of a point in time."""

    def __init__(self, txns: list[Transaction] = ()):
        self.by_id: dict[str, Transaction] = {}
        self.by_account: dict[str, list[Transaction]] = defaultdict(list)
        self._ts: dict[str, list] = defaultdict(list)
        self._first_payment: set[tuple[str, str]] = set()
        # recipient -> (first time each customer paid them, customer), in time order
        self._payers: dict[str, list[tuple[object, str]]] = defaultdict(list)
        self._last_ts = None
        for t in txns:
            self.add(t)

    def add(self, t: Transaction) -> None:
        """Append one transaction. Transactions must arrive in time order, as they do off the stream.

        The offline evaluation builds a History from a whole dataset and queries it as of each
        alert; the API builds one transaction by transaction and queries it the moment an alert
        fires. Same class both ways, so the features can't drift apart in code.
        """
        if self._last_ts is not None and t.ts < self._last_ts:  # every lookup below assumes time order
            raise ValueError(f"out-of-order transaction {t.txn_id}: {t.ts} < {self._last_ts}")
        self._last_ts = t.ts
        self.by_id[t.txn_id] = t
        self.by_account[t.account_id].append(t)
        self._ts[t.account_id].append(t.ts)
        if t.direction == "out" and (t.counterparty_id, t.account_id) not in self._first_payment:
            self._first_payment.add((t.counterparty_id, t.account_id))
            self._payers[t.counterparty_id].append((t.ts, t.account_id))

    def account_between(self, account: str, start, end) -> list[Transaction]:
        """The account's transactions with start <= ts < end."""
        ts = self._ts.get(account, [])
        return self.by_account.get(account, [])[bisect_left(ts, start):bisect_left(ts, end)]

    def other_payers_before(self, recipient: str, account: str, when) -> int:
        """How many customers other than `account` paid `recipient` before `when`."""
        return sum(1 for ts, acct in self._payers.get(recipient, []) if ts < when and acct != account)


def features(alert: Alert, hist: History) -> list[float]:
    ev = [hist.by_id[i] for i in alert.evidence]
    first = ev[0].ts
    if alert.typology == STRUCTURING:
        prior = hist.account_between(alert.account_id, first - LOOKBACK, first)
        prior_cash = sum(1 for t in prior if t.direction == "in" and t.channel == "cash")
        total = sum(t.amount for t in ev)
        return [math.log1p(prior_cash), len(ev), total / 10_000]
    if alert.typology == PASS_THROUGH:
        anchor, outs = ev[0], [t for t in ev if t.direction == "out"]
        reach = max(hist.other_payers_before(t.counterparty_id, alert.account_id, alert.detected_at) for t in outs)
        days_to_out = (outs[0].ts - anchor.ts).total_seconds() / 86_400
        return [len(outs), math.log1p(reach), days_to_out, sum(t.amount for t in outs) / anchor.amount]
    if alert.typology == FAN_IN:
        prior = hist.account_between(alert.account_id, first - LOOKBACK, first)
        prior_senders = {t.counterparty_id for t in prior if t.direction == "in" and t.channel != "cash"}
        senders = {t.counterparty_id for t in ev}
        return [math.log1p(len(prior_senders)), len(senders), sum(t.amount for t in ev) / len(ev) / 1000]
    raise ValueError(f"unknown typology {alert.typology}")


@dataclass
class LogReg:
    """L2-regularised logistic regression on standardised features, batch gradient descent."""

    l2: float = 0.01
    lr: float = 0.5
    epochs: int = 2000
    w: list[float] = field(default_factory=list)
    b: float = 0.0
    mu: list[float] = field(default_factory=list)
    sd: list[float] = field(default_factory=list)

    def fit(self, X: list[list[float]], y: list[int], weights: list[float] | None = None) -> LogReg:
        """`weights` scale each example's pull on the fit (e.g. 1/sample-rate for sampled reviews)."""
        n, d = len(X), len(X[0])
        wts = [1.0] * n if weights is None else list(weights)
        W = sum(wts)
        self.mu = [sum(w * r[j] for r, w in zip(X, wts)) / W for j in range(d)]
        self.sd = [math.sqrt(sum(w * (r[j] - self.mu[j]) ** 2 for r, w in zip(X, wts)) / W) or 1.0 for j in range(d)]
        Z = [self._std(r) for r in X]
        pos = sum(w for w, yi in zip(wts, y) if yi)
        # classes are imbalanced; weight them equally so the boundary isn't pulled toward the majority
        cw = {1: W / (2 * pos), 0: W / (2 * (W - pos))} if 0 < pos < W else {0: 1.0, 1: 1.0}
        self.w, self.b = [0.0] * d, 0.0
        for _ in range(self.epochs):
            gw, gb = [0.0] * d, 0.0
            for z, yi, wi in zip(Z, y, wts):
                err = (_sigmoid(sum(wj * zj for wj, zj in zip(self.w, z)) + self.b) - yi) * cw[yi] * wi
                gb += err
                for j in range(d):
                    gw[j] += err * z[j]
            self.w = [wj - self.lr * (g / W + self.l2 * wj) for wj, g in zip(self.w, gw)]
            self.b -= self.lr * gb / W
        return self

    def predict(self, x: list[float]) -> float:
        return _sigmoid(sum(wj * zj for wj, zj in zip(self.w, self._std(x))) + self.b)

    def contributions(self, x: list[float]) -> list[float]:
        """Each feature's push on the log-odds, relative to an average alert (positive = toward real)."""
        return [wj * zj for wj, zj in zip(self.w, self._std(x))]

    def to_dict(self) -> dict:
        return {"w": self.w, "b": self.b, "mu": self.mu, "sd": self.sd}

    @classmethod
    def from_dict(cls, d: dict) -> LogReg:
        return cls(w=list(d["w"]), b=d["b"], mu=list(d["mu"]), sd=list(d["sd"]))

    def _std(self, x: list[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(x, self.mu, self.sd)]


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


Example = tuple[Alert, History, bool]  # (alert, history it was raised in, was it a real case)

MARGINS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


@dataclass
class Decision:
    """What triage decided about one alert, with everything needed to explain it later.

    A bank has to be able to say, months after the fact, why an alert never reached an
    investigator. So the record carries the model version and the exact inputs, and
    `Triage.replay` recomputes the score from them.
    """

    alert_id: str
    typology: str
    model_version: str
    features: dict[str, float]
    score: float
    threshold: float
    queued: bool
    contributions: dict[str, float]  # log-odds push per feature, most negative first

    @property
    def reason(self) -> str:
        names = list(self.contributions)
        if self.queued:
            why = f"; biggest push up: {names[-1]}" if names else ""
            return f"queued: score {self.score:.3f} >= threshold {self.threshold:.3f}{why}"
        why = f"; biggest push down: {names[0]}" if names else ""
        return f"suppressed: score {self.score:.3f} < threshold {self.threshold:.3f}{why}"


class Triage:
    """One model and one suppression threshold per typology."""

    def __init__(self):
        self.models: dict[str, LogReg] = {}
        self.base: dict[str, float] = {}  # lowest score among real training cases
        self.margin = 0.0  # thresholds sit this fraction below `base`

    def fit(self, examples: list[Example], weights: list[float] | None = None) -> Triage:
        wts = [1.0] * len(examples) if weights is None else weights
        for typ in TYPOLOGIES:
            rows = [(features(a, h), int(real), w) for (a, h, real), w in zip(examples, wts) if a.typology == typ]
            if not rows or not any(y for _, y, _ in rows):
                continue
            model = LogReg().fit([x for x, _, _ in rows], [y for _, y, _ in rows], [w for _, _, w in rows])
            self.models[typ] = model
            self.base[typ] = min(model.predict(x) for x, y, _ in rows if y) * 0.999
        return self

    def calibrate(self, examples: list[Example]) -> Triage:
        """Pick the smallest margin that keeps every real case in a held-out validation set.

        Keeping every real *training* case isn't enough: on new data a real case can
        score a little lower than anything the model was fitted on.
        """
        scored = [(self.score(a, h), a.typology, real) for a, h, real in examples]
        for m in MARGINS:
            self.margin = m
            if all(s >= self.threshold(typ) for s, typ, real in scored if real):
                break
        return self

    def threshold(self, typology: str) -> float:
        return self.base.get(typology, 0.0) * (1 - self.margin)

    def score(self, alert: Alert, hist: History) -> float:
        model = self.models.get(alert.typology)
        return 1.0 if model is None else model.predict(features(alert, hist))

    def keep(self, alert: Alert, hist: History) -> bool:
        """False means the alert is suppressed (still logged, just not queued for an investigator)."""
        return self.score(alert, hist) >= self.threshold(alert.typology)

    def decide(self, alert: Alert, hist: History) -> Decision:
        typ = alert.typology
        model = self.models.get(typ)
        if model is None:  # no model for this typology: everything is queued
            return Decision(alert.alert_id, typ, self.version, {}, 1.0, 0.0, True, {})
        x = features(alert, hist)
        names = FEATURE_NAMES[typ]
        score, threshold = model.predict(x), self.threshold(typ)
        pushes = sorted(zip(names, model.contributions(x)), key=lambda p: p[1])
        return Decision(
            alert.alert_id, typ, self.version, dict(zip(names, x)), score, threshold, score >= threshold,
            {n: round(c, 3) for n, c in pushes},
        )

    def replay(self, d: Decision) -> float:
        """Recompute a logged decision's score from its recorded inputs."""
        if d.model_version != self.version:
            raise ValueError(f"decision was made by model {d.model_version}, this is {self.version}")
        model = self.models.get(d.typology)
        return 1.0 if model is None else model.predict([d.features[n] for n in FEATURE_NAMES[d.typology]])

    # -- persistence -----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "margin": self.margin,
            "typologies": {
                t: {"features": FEATURE_NAMES[t], "base": self.base[t], **m.to_dict()} for t, m in self.models.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> Triage:
        t = cls()
        t.margin = d["margin"]
        for typ, m in d["typologies"].items():
            if m["features"] != FEATURE_NAMES[typ]:
                raise ValueError(f"{typ} model was trained on {m['features']}, code computes {FEATURE_NAMES[typ]}")
            t.models[typ] = LogReg.from_dict(m)
            t.base[typ] = m["base"]
        return t

    @property
    def version(self) -> str:
        """Content hash of the parameters: two models with the same version score identically."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return "triage-" + hashlib.sha256(blob).hexdigest()[:12]

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, sort_keys=True)
            f.write("\n")

    @classmethod
    def load(cls, path) -> Triage:
        with open(path) as f:
            return cls.from_dict(json.load(f))
