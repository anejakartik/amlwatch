"""Score alerts against the generator's ground truth.

Scoring is per (account, typology): an alert counts as a true positive if that
typology was injected into that account. Alerts on look-alike accounts are
broken out separately, because they're the false positives an investigator would
actually spend time on.
"""

from __future__ import annotations

from .generator import Dataset, generate
from .models import TYPOLOGIES, Alert
from .monitor import Monitor
from .triage import Example, History, Triage, features

FIT_SEEDS = range(11, 16)
VAL_SEEDS = range(16, 21)


def score(alerts: list[Alert], data: Dataset) -> dict:
    predicted = {(a.account_id, a.typology) for a in alerts}
    actual = {(acct, typ) for acct, typs in data.truth.items() for typ in typs}

    out: dict[str, dict] = {}
    for typ in TYPOLOGIES:
        p = {x for x in predicted if x[1] == typ}
        a = {x for x in actual if x[1] == typ}
        tp = len(p & a)
        fps = p - a
        out[typ] = {
            "injected": len(a),
            "alerts": len(p),
            "true_positives": tp,
            "false_positives": len(fps),
            "false_positives_on_lookalikes": sum(1 for acct, _ in fps if acct in data.lookalikes),
            "precision": round(tp / len(p), 3) if p else None,
            "recall": round(tp / len(a), 3) if a else None,
        }
    return out


def labelled_alerts(seeds) -> list[Example]:
    """Run the monitor over each seed's dataset and label every alert with the ground truth."""
    out: list[Example] = []
    for seed in seeds:
        data = generate(seed=seed)
        hist = History(data.transactions)
        for a in Monitor().run(data.transactions):
            out.append((a, hist, a.typology in data.truth.get(a.account_id, set())))
    return out


def triage_report(fit_seeds, val_seeds, test_seeds) -> tuple[Triage, dict]:
    """Fit on one set of seeds, pick the margin on a second, report on a third.

    Scored per alert, not per account: an account that alerts four times is four
    pieces of investigator work.
    """
    triage = Triage().fit(labelled_alerts(fit_seeds)).calibrate(labelled_alerts(val_seeds))
    test = labelled_alerts(test_seeds)
    out: dict[str, dict] = {}
    for typ in TYPOLOGIES:
        rows = [(real, triage.keep(a, h)) for a, h, real in test if a.typology == typ]
        real = sum(r for r, _ in rows)
        kept = sum(k for _, k in rows)
        real_kept = sum(r and k for r, k in rows)
        out[typ] = {
            "alerts": len(rows),
            "kept": kept,
            "real": real,
            "real_kept": real_kept,
            "precision_before": round(real / len(rows), 3) if rows else None,
            "precision_after": round(real_kept / kept, 3) if kept else None,
            "recall_after": round(real_kept / real, 3) if real else None,
        }
    return triage, out


def train_triage() -> Triage:
    """The shipped model: fitted on seeds 11-15, margin picked on 16-20. Never sees the demo seeds."""
    return Triage().fit(labelled_alerts(FIT_SEEDS)).calibrate(labelled_alerts(VAL_SEEDS))


def online_parity(seeds) -> tuple[int, list[str]]:
    """Check the features computed live equal the ones the offline evaluation scored.

    Offline, each alert's features come from a History holding the whole dataset,
    future included, queried as of the alert. Live, there is only what has arrived so
    far. If any feature quietly read the future, the two would differ and the offline
    numbers would be better than anything the stream could reproduce.

    Returns (alerts checked, ids of alerts whose features differ).
    """
    checked, mismatched = 0, []
    for seed in seeds:
        data = generate(seed=seed)
        offline = History(data.transactions)
        live, monitor = History(), Monitor()
        for txn in data.transactions:
            live.add(txn)
            for a in monitor.ingest(txn):
                checked += 1
                if features(a, live) != features(a, offline):
                    mismatched.append(f"seed {seed}: {a.alert_id}")
    return checked, mismatched
