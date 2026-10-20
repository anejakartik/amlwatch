"""Score alerts against the generator's ground truth.

Scoring is per (account, typology): an alert counts as a true positive if that
typology was injected into that account. Alerts on look-alike accounts are
broken out separately, because they're the false positives an investigator would
actually spend time on.
"""

from __future__ import annotations

from .generator import Dataset
from .models import TYPOLOGIES, Alert


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
