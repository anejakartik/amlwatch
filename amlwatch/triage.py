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
    """Read-only index over a transaction stream, queried as of a point in time."""

    def __init__(self, txns: list[Transaction]):
        self.by_id = {t.txn_id: t for t in txns}
        self.by_account: dict[str, list[Transaction]] = defaultdict(list)
        first_payment: dict[tuple[str, str], object] = {}
        for t in txns:  # txns arrive sorted by time
            self.by_account[t.account_id].append(t)
            if t.direction == "out":
                first_payment.setdefault((t.counterparty_id, t.account_id), t.ts)
        self._ts = {acct: [t.ts for t in ts] for acct, ts in self.by_account.items()}
        # recipient -> sorted (first time each customer paid them, customer)
        self._payers: dict[str, list[tuple[object, str]]] = defaultdict(list)
        for (recipient, acct), ts in first_payment.items():
            self._payers[recipient].append((ts, acct))
        for payers in self._payers.values():
            payers.sort()

    def account_between(self, account: str, start, end) -> list[Transaction]:
        """The account's transactions with start <= ts < end."""
        ts = self._ts.get(account, [])
        return self.by_account[account][bisect_left(ts, start):bisect_left(ts, end)]

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

    def fit(self, X: list[list[float]], y: list[int]) -> LogReg:
        n, d = len(X), len(X[0])
        self.mu = [sum(r[j] for r in X) / n for j in range(d)]
        self.sd = [math.sqrt(sum((r[j] - self.mu[j]) ** 2 for r in X) / n) or 1.0 for j in range(d)]
        Z = [self._std(r) for r in X]
        pos = sum(y)
        # classes are imbalanced; weight them equally so the boundary isn't pulled toward the majority
        cw = {1: n / (2 * pos), 0: n / (2 * (n - pos))} if 0 < pos < n else {0: 1.0, 1: 1.0}
        self.w, self.b = [0.0] * d, 0.0
        for _ in range(self.epochs):
            gw, gb = [0.0] * d, 0.0
            for z, yi in zip(Z, y):
                err = (_sigmoid(sum(wj * zj for wj, zj in zip(self.w, z)) + self.b) - yi) * cw[yi]
                gb += err
                for j in range(d):
                    gw[j] += err * z[j]
            self.w = [wj - self.lr * (g / n + self.l2 * wj) for wj, g in zip(self.w, gw)]
            self.b -= self.lr * gb / n
        return self

    def predict(self, x: list[float]) -> float:
        return _sigmoid(sum(wj * zj for wj, zj in zip(self.w, self._std(x))) + self.b)

    def _std(self, x: list[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(x, self.mu, self.sd)]


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


Example = tuple[Alert, History, bool]  # (alert, history it was raised in, was it a real case)

MARGINS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


class Triage:
    """One model and one suppression threshold per typology."""

    def __init__(self):
        self.models: dict[str, LogReg] = {}
        self.base: dict[str, float] = {}  # lowest score among real training cases
        self.margin = 0.0  # thresholds sit this fraction below `base`

    def fit(self, examples: list[Example]) -> Triage:
        for typ in TYPOLOGIES:
            rows = [(features(a, h), int(real)) for a, h, real in examples if a.typology == typ]
            if not rows or not any(y for _, y in rows):
                continue
            model = LogReg().fit([x for x, _ in rows], [y for _, y in rows])
            self.models[typ] = model
            self.base[typ] = min(model.predict(x) for x, y in rows if y) * 0.999
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
