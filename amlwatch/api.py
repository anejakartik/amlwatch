"""HTTP surface: stream transactions in, read alerts, draft validated SAR narratives."""

from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .generator import generate
from .models import TYPOLOGIES, Alert, Transaction
from .monitor import Monitor
from .narrative import ClaudeDrafter, draft_narrative, evidence_table
from .train import MODEL_PATH
from .triage import Decision, History, Triage

app = FastAPI(
    title="amlwatch",
    version=__version__,
    description="Streaming AML transaction monitoring, alert triage, and citation-checked SAR narrative drafts.",
)


class TransactionIn(BaseModel):
    txn_id: str
    ts: datetime
    account_id: str
    counterparty_id: str
    direction: str = Field(pattern="^(in|out)$")
    amount: float = Field(gt=0)
    channel: str = Field(pattern="^(cash|wire|ach|card)$")


class AlertOut(BaseModel):
    alert_id: str
    account_id: str
    typology: str
    detected_at: datetime
    evidence: list[str]
    facts: dict


class ProblemOut(BaseModel):
    sentence: str
    reason: str


class DraftOut(BaseModel):
    alert_id: str
    drafter: str
    status: str
    text: str
    attempts: int
    problems: list[ProblemOut]


class TriageOut(BaseModel):
    model_version: str
    queued: bool
    score: float
    threshold: float
    reason: str
    features: dict[str, float]
    contributions: dict[str, float]


class _State:
    def __init__(self):
        self.monitor = Monitor()
        self.history = History()
        self.triage = Triage.load(os.getenv("AMLWATCH_TRIAGE_MODEL", MODEL_PATH))
        self.decisions: dict[str, Decision] = {}

    @property
    def txns(self) -> dict[str, Transaction]:
        return self.history.by_id

    def ingest(self, txn: Transaction) -> list[Alert]:
        # history first: the alert's features must see the transaction that completed it
        self.history.add(txn)
        new = self.monitor.ingest(txn)
        for a in new:
            self.decisions[a.alert_id] = self.triage.decide(a, self.history)
        return new


STATE = _State()


def _alert_out(a: Alert) -> AlertOut:
    return AlertOut(**a.__dict__)


def _triage_out(d: Decision) -> TriageOut:
    return TriageOut(
        model_version=d.model_version, queued=d.queued, score=round(d.score, 4), threshold=round(d.threshold, 4),
        reason=d.reason, features=d.features, contributions=d.contributions,
    )


class QueuedAlertOut(AlertOut):
    triage: TriageOut


def _with_triage(a: Alert) -> QueuedAlertOut:
    return QueuedAlertOut(**a.__dict__, triage=_triage_out(STATE.decisions[a.alert_id]))


@app.on_event("startup")
def _seed_demo_data() -> None:
    """Load the synthetic dataset so a fresh deploy has alerts to look at."""
    if os.getenv("AMLWATCH_SEED_DEMO", "1") == "1" and not STATE.txns:
        for txn in generate(seed=int(os.getenv("AMLWATCH_SEED", "7"))).transactions:
            STATE.ingest(txn)


@app.get("/health")
def health():
    return {
        "status": "ok", "transactions": len(STATE.txns), "alerts": len(STATE.monitor.alerts),
        "triage_model": STATE.triage.version,
    }


@app.post("/transactions", response_model=list[AlertOut])
def ingest(txns: list[TransactionIn]):
    """Ingest transactions in timestamp order; returns any alerts they complete."""
    new: list[Alert] = []
    for t in txns:
        if t.txn_id in STATE.txns:
            raise HTTPException(409, f"duplicate txn_id {t.txn_id}")
        try:
            new.extend(STATE.ingest(Transaction(**t.model_dump())))
        except ValueError as err:  # out of order
            raise HTTPException(422, str(err)) from err
    return [_alert_out(a) for a in new]


@app.get("/alerts", response_model=list[QueuedAlertOut])
def alerts(typology: Optional[str] = None, status: Optional[str] = None, limit: int = 100):  # noqa: UP045 - FastAPI evaluates this on 3.9
    """Every alert, queued or suppressed. `status=queued|suppressed` filters on the triage decision."""
    if typology is not None and typology not in TYPOLOGIES:
        raise HTTPException(400, f"unknown typology; expected one of {', '.join(TYPOLOGIES)}")
    if status not in (None, "queued", "suppressed"):
        raise HTTPException(400, "status must be 'queued' or 'suppressed'")
    rows = [
        a for a in STATE.monitor.alerts
        if typology in (None, a.typology)
        and status in (None, "queued" if STATE.decisions[a.alert_id].queued else "suppressed")
    ]
    return [_with_triage(a) for a in rows[:limit]]


@app.get("/queue", response_model=list[QueuedAlertOut])
def queue(limit: int = 50):
    """The investigator's work queue: alerts triage kept, highest score first."""
    kept = [a for a in STATE.monitor.alerts if STATE.decisions[a.alert_id].queued]
    kept.sort(key=lambda a: -STATE.decisions[a.alert_id].score)
    return [_with_triage(a) for a in kept[:limit]]


@app.get("/suppressed/sample", response_model=list[QueuedAlertOut])
def suppressed_sample(n: int = 10, seed: int = 0):
    """A random sample of suppressed alerts for below-the-line review.

    Suppressed alerts are never deleted. Reviewing a random sample of them is the only
    way to learn what triage is hiding once there's no ground truth to score against.
    `seed` makes the sample reproducible, so a reviewer's sample can be re-drawn for an audit.
    """
    hidden = [a for a in STATE.monitor.alerts if not STATE.decisions[a.alert_id].queued]
    return [_with_triage(a) for a in random.Random(seed).sample(hidden, min(n, len(hidden)))]


def _find(alert_id: str) -> Alert:
    for a in STATE.monitor.alerts:
        if a.alert_id == alert_id:
            return a
    raise HTTPException(404, f"no alert {alert_id}")


@app.get("/alerts/{alert_id}")
def alert_detail(alert_id: str):
    a = _find(alert_id)
    return {**_with_triage(a).model_dump(), "evidence_table": evidence_table(a, STATE.txns)}


@app.post("/alerts/{alert_id}/narrative", response_model=DraftOut)
def narrative(alert_id: str, drafter: str = "template"):
    """Draft a SAR narrative. `drafter=claude` needs ANTHROPIC_API_KEY on the server."""
    a = _find(alert_id)
    if drafter == "template":
        d = draft_narrative(a, STATE.txns)
    elif drafter == "claude":
        try:
            d = draft_narrative(a, STATE.txns, drafter=ClaudeDrafter())
        except Exception as err:  # noqa: BLE001 - surface any SDK/config failure as 502
            raise HTTPException(502, f"claude drafter failed: {err}") from err
    else:
        raise HTTPException(400, "drafter must be 'template' or 'claude'")
    return DraftOut(
        alert_id=d.alert_id, drafter=d.drafter, status=d.status, text=d.text, attempts=d.attempts,
        problems=[ProblemOut(sentence=p.sentence, reason=p.reason) for p in d.problems],
    )


@app.get("/metrics")
def metrics():
    by_typology = {t: {"alerts": 0, "queued": 0} for t in TYPOLOGIES}
    for a in STATE.monitor.alerts:
        by_typology[a.typology]["alerts"] += 1
        by_typology[a.typology]["queued"] += STATE.decisions[a.alert_id].queued
    queued = sum(t["queued"] for t in by_typology.values())
    return {
        "transactions": len(STATE.txns), "alerts": len(STATE.monitor.alerts), "queued": queued,
        "suppressed": len(STATE.monitor.alerts) - queued, "triage_model": STATE.triage.version,
        "alerts_by_typology": by_typology,
    }
