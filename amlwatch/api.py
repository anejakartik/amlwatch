"""HTTP surface: stream transactions in, read alerts, draft validated SAR narratives."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .generator import generate
from .models import TYPOLOGIES, Alert, Transaction
from .monitor import Monitor
from .narrative import ClaudeDrafter, draft_narrative, evidence_table

app = FastAPI(
    title="amlwatch",
    version=__version__,
    description="Streaming AML transaction monitoring with citation-checked SAR narrative drafts.",
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


class _State:
    def __init__(self):
        self.monitor = Monitor()
        self.txns: dict[str, Transaction] = {}

    def ingest(self, txn: Transaction) -> list[Alert]:
        self.txns[txn.txn_id] = txn
        return self.monitor.ingest(txn)


STATE = _State()


def _alert_out(a: Alert) -> AlertOut:
    return AlertOut(**a.__dict__)


@app.on_event("startup")
def _seed_demo_data() -> None:
    """Load the synthetic dataset so a fresh deploy has alerts to look at."""
    if os.getenv("AMLWATCH_SEED_DEMO", "1") == "1" and not STATE.txns:
        for txn in generate(seed=int(os.getenv("AMLWATCH_SEED", "7"))).transactions:
            STATE.ingest(txn)


@app.get("/health")
def health():
    return {"status": "ok", "transactions": len(STATE.txns), "alerts": len(STATE.monitor.alerts)}


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


@app.get("/alerts", response_model=list[AlertOut])
def alerts(typology: Optional[str] = None, limit: int = 100):  # noqa: UP045 - FastAPI evaluates this on 3.9
    if typology is not None and typology not in TYPOLOGIES:
        raise HTTPException(400, f"unknown typology; expected one of {', '.join(TYPOLOGIES)}")
    rows = [a for a in STATE.monitor.alerts if typology in (None, a.typology)]
    return [_alert_out(a) for a in rows[:limit]]


def _find(alert_id: str) -> Alert:
    for a in STATE.monitor.alerts:
        if a.alert_id == alert_id:
            return a
    raise HTTPException(404, f"no alert {alert_id}")


@app.get("/alerts/{alert_id}")
def alert_detail(alert_id: str):
    a = _find(alert_id)
    return {**_alert_out(a).model_dump(), "evidence_table": evidence_table(a, STATE.txns)}


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
    by_typology = {t: 0 for t in TYPOLOGIES}
    for a in STATE.monitor.alerts:
        by_typology[a.typology] += 1
    return {"transactions": len(STATE.txns), "alerts": len(STATE.monitor.alerts), "alerts_by_typology": by_typology}
