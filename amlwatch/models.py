"""Core records: a bank transaction and an alert raised against an account."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# Typologies the monitor detects. Names follow common AML usage.
STRUCTURING = "structuring"  # cash deposits kept just under the $10k CTR threshold
PASS_THROUGH = "pass_through"  # funds arrive and leave again within hours
FAN_IN = "fan_in"  # many new senders converge on one account in a short window

TYPOLOGIES = (STRUCTURING, PASS_THROUGH, FAN_IN)


@dataclass(frozen=True)
class Transaction:
    txn_id: str
    ts: datetime
    account_id: str  # the monitored (customer) account
    counterparty_id: str  # the other side: a sender for "in", a recipient for "out"
    direction: str  # "in" | "out"
    amount: float  # USD, always positive
    channel: str  # "cash" | "wire" | "ach" | "card"


@dataclass
class Alert:
    alert_id: str
    account_id: str
    typology: str
    detected_at: datetime
    evidence: list[str]  # txn_ids that triggered the rule, in time order
    # Computed facts about the evidence (totals, counts, window). The narrative
    # validator checks drafted claims against the transactions, not these.
    facts: dict = field(default_factory=dict)
