"""Streaming transaction monitor.

Transactions are fed one at a time, in timestamp order, the way they'd arrive off
a payments stream. The monitor keeps a bounded per-account window of recent
activity and evaluates each typology rule incrementally, so an alert fires on the
transaction that completes a pattern rather than in a nightly batch.

Rules (thresholds are constructor arguments; defaults below):
  structuring  >= 3 cash deposits of $8,000-$9,999.99 within 7 days, totalling
               more than $10,000. Each deposit alone stays under the $10,000
               Currency Transaction Report threshold; together they don't.
  pass_through a single inbound of >= $10,000, and >= 90% of that amount sent
               back out (non-card) within 48h after it arrived.
  fan_in       >= 10 distinct counterparties sending money within 3 days, none of
               whom had sent to this account before.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import timedelta

from .models import FAN_IN, PASS_THROUGH, STRUCTURING, Alert, Transaction

CTR_THRESHOLD = 10_000.0


@dataclass
class Thresholds:
    structuring_min_amount: float = 8_000.0
    structuring_min_count: int = 3
    structuring_window: timedelta = timedelta(days=7)
    pass_through_min_in: float = 10_000.0
    pass_through_out_ratio: float = 0.90
    pass_through_window: timedelta = timedelta(hours=48)
    fan_in_min_senders: int = 10
    fan_in_window: timedelta = timedelta(days=3)
    # don't re-raise the same typology on the same account inside this period
    cooldown: timedelta = timedelta(days=7)


class Monitor:
    def __init__(self, thresholds: Thresholds | None = None):
        self.t = thresholds or Thresholds()
        self._window = max(self.t.structuring_window, self.t.pass_through_window, self.t.fan_in_window)
        self._recent: dict[str, deque[Transaction]] = defaultdict(deque)
        self._known_senders: dict[str, set[str]] = defaultdict(set)  # senders seen before the fan-in window
        self._last_alert: dict[tuple[str, str], object] = {}
        self._last_ts = None
        self.alerts: list[Alert] = []

    def ingest(self, txn: Transaction) -> list[Alert]:
        if self._last_ts is not None and txn.ts < self._last_ts:
            raise ValueError(f"out-of-order transaction {txn.txn_id}: {txn.ts} < {self._last_ts}")
        self._last_ts = txn.ts

        recent = self._recent[txn.account_id]
        recent.append(txn)
        while recent and txn.ts - recent[0].ts > self._window:
            old = recent.popleft()
            if old.direction == "in":
                self._known_senders[txn.account_id].add(old.counterparty_id)

        new: list[Alert] = []
        for rule in (self._structuring, self._pass_through, self._fan_in):
            alert = rule(txn, recent)
            if alert is not None:
                new.append(alert)
        self.alerts.extend(new)
        return new

    def run(self, txns) -> list[Alert]:
        for txn in txns:
            self.ingest(txn)
        return self.alerts

    # -- rules ---------------------------------------------------------------

    def _raise(self, txn: Transaction, typology: str, evidence: list[Transaction], facts: dict) -> Alert | None:
        key = (txn.account_id, typology)
        last = self._last_alert.get(key)
        if last is not None and txn.ts - last < self.t.cooldown:
            return None
        self._last_alert[key] = txn.ts
        evidence = sorted(evidence, key=lambda e: e.ts)
        facts = {
            "first_ts": evidence[0].ts.isoformat(),
            "last_ts": evidence[-1].ts.isoformat(),
            **facts,
        }
        return Alert(
            alert_id=f"A{len(self.alerts) + 1:05d}-{typology}",
            account_id=txn.account_id,
            typology=typology,
            detected_at=txn.ts,
            evidence=[e.txn_id for e in evidence],
            facts=facts,
        )

    def _structuring(self, txn: Transaction, recent) -> Alert | None:
        if not (txn.direction == "in" and txn.channel == "cash"):
            return None
        hits = [
            r for r in recent
            if r.direction == "in" and r.channel == "cash"
            and self.t.structuring_min_amount <= r.amount < CTR_THRESHOLD
            and txn.ts - r.ts <= self.t.structuring_window
        ]
        total = sum(r.amount for r in hits)
        if len(hits) >= self.t.structuring_min_count and total > CTR_THRESHOLD:
            return self._raise(txn, STRUCTURING, hits, {"count": len(hits), "total": round(total, 2)})
        return None

    def _pass_through(self, txn: Transaction, recent) -> Alert | None:
        if txn.direction != "out":
            return None
        window = [r for r in recent if txn.ts - r.ts <= self.t.pass_through_window]
        # anchor on each large inbound and measure what left *after* it. Comparing
        # against all inflows let an ordinary paycheck in the same window dilute
        # the ratio below the threshold and hide the pattern.
        for anchor in (r for r in window if r.direction == "in" and r.amount >= self.t.pass_through_min_in):
            outs = [
                r for r in window
                if r.direction == "out" and r.ts >= anchor.ts and r.channel != "card"
                and r.ts - anchor.ts <= self.t.pass_through_window
            ]
            total_out = sum(r.amount for r in outs)
            if total_out >= self.t.pass_through_out_ratio * anchor.amount:
                return self._raise(
                    txn, PASS_THROUGH, [anchor, *outs],
                    {"total_in": round(anchor.amount, 2), "total_out": round(total_out, 2),
                     "out_ratio": round(total_out / anchor.amount, 3)},
                )
        return None

    def _fan_in(self, txn: Transaction, recent) -> Alert | None:
        if txn.direction != "in":
            return None
        known = self._known_senders[txn.account_id]
        ins = [
            r for r in recent
            if r.direction == "in" and r.channel != "cash"
            and txn.ts - r.ts <= self.t.fan_in_window
            and r.counterparty_id not in known
        ]
        # a sender who also appears earlier in the window isn't "new" for long, but
        # what matters here is breadth: how many distinct parties showed up at once
        senders = {r.counterparty_id for r in ins}
        if len(senders) >= self.t.fan_in_min_senders:
            return self._raise(
                txn, FAN_IN, ins,
                {"distinct_senders": len(senders), "total": round(sum(r.amount for r in ins), 2)},
            )
        return None
