"""Synthetic bank transactions with known, injected laundering patterns.

No real customer data exists anywhere in this repo. The generator builds a
population of ordinary retail accounts (payroll in, card spend out, the odd cash
deposit or wire), then injects three typologies into a known subset of accounts.
Because the ground truth is known, detection can be scored with precision and
recall instead of eyeballed.

It also injects *legitimate look-alikes* on purpose, because a monitor that only
sees clean negatives looks far better than it is:
  - home purchases: a large transfer in, then a large wire out to a title company
    within two days (looks like pass-through)
  - small businesses: steady customer payments from many different senders
    (can look like fan-in on a busy day)
  - occasional genuine cash deposits in the $8k-$10k range (look like structuring)
  - cash-heavy businesses (restaurants, laundromats) depositing takings every
    couple of days, sometimes several in the $8k-$10k range in one week
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from .models import FAN_IN, PASS_THROUGH, STRUCTURING, Transaction

START = datetime(2026, 1, 5)


@dataclass
class Dataset:
    transactions: list[Transaction]  # sorted by timestamp
    truth: dict[str, set[str]]  # account_id -> typologies injected into it
    lookalikes: dict[str, str]  # account_id -> kind of legitimate look-alike


class _Builder:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.txns: list[Transaction] = []

    def add(self, ts, account, counterparty, direction, amount, channel):
        self.txns.append(
            Transaction(
                txn_id="",  # assigned after sorting so ids follow time order
                ts=ts,
                account_id=account,
                counterparty_id=counterparty,
                direction=direction,
                amount=round(amount, 2),
                channel=channel,
            )
        )

    def at(self, day: float) -> datetime:
        return START + timedelta(days=day)


def _background(b: _Builder, account: str, days: int) -> None:
    """Ordinary retail activity: payroll, card spend, bills, rare cash/wires."""
    rng = b.rng
    employer = f"EMP{rng.randint(1, 40):03d}"
    salary = rng.uniform(1800, 4500)
    for d in range(0, days, 14):  # biweekly payroll
        b.add(b.at(d + rng.uniform(0.3, 0.4)), account, employer, "in", salary, "ach")
    for _ in range(int(days * rng.uniform(0.6, 1.2))):  # card spend
        b.add(b.at(rng.uniform(0, days)), account, f"MER{rng.randint(1, 500):04d}", "out", rng.uniform(5, 180), "card")
    for d in range(0, days, 30):  # rent / bills
        biller = f"BILL{rng.randint(1, 60):03d}"
        b.add(b.at(d + rng.uniform(1, 3)), account, biller, "out", rng.uniform(600, 2200), "ach")
    if rng.random() < 0.3:  # small cash deposits
        for _ in range(rng.randint(1, 3)):
            b.add(b.at(rng.uniform(0, days)), account, "CASH", "in", rng.uniform(100, 2500), "cash")
    if rng.random() < 0.04:  # a single genuine large cash deposit (sold a car, etc.)
        b.add(b.at(rng.uniform(0, days)), account, "CASH", "in", rng.uniform(8000, 9999), "cash")


def _structuring(b: _Builder, account: str, days: int) -> None:
    rng = b.rng
    start = rng.uniform(5, days - 8)
    for _ in range(rng.randint(3, 6)):
        b.add(b.at(start + rng.uniform(0, 5)), account, "CASH", "in", rng.uniform(8000, 9900), "cash")


def _pass_through(b: _Builder, account: str, days: int) -> None:
    rng = b.rng
    start = rng.uniform(5, days - 4)
    amount = rng.uniform(20000, 80000)
    b.add(b.at(start), account, f"SRC{rng.randint(1, 999):03d}", "in", amount, "wire")
    legs = rng.randint(1, 3)
    out_total = amount * rng.uniform(0.92, 0.99)
    for _ in range(legs):
        delay_h = rng.uniform(1, 36)
        b.add(b.at(start + delay_h / 24), account, f"DST{rng.randint(1, 999):03d}", "out", out_total / legs, "wire")


def _fan_in(b: _Builder, account: str, days: int) -> None:
    rng = b.rng
    start = rng.uniform(5, days - 5)
    total = 0.0
    for _ in range(rng.randint(10, 20)):
        amt = rng.uniform(200, 900)
        total += amt
        b.add(b.at(start + rng.uniform(0, 3)), account, f"P2P{rng.randint(1, 99999):05d}", "in", amt, "ach")
    b.add(b.at(start + 3.2), account, f"DST{rng.randint(1, 999):03d}", "out", total * 0.95, "wire")


def _home_purchase(b: _Builder, account: str, days: int) -> None:
    rng = b.rng
    start = rng.uniform(5, days - 4)
    amount = rng.uniform(30000, 90000)
    b.add(b.at(start), account, f"BRK{rng.randint(1, 99):02d}", "in", amount, "wire")  # from own brokerage
    b.add(b.at(start + rng.uniform(0.5, 1.8)), account, "TITLE01", "out", amount * rng.uniform(0.9, 0.99), "wire")


def _small_business(b: _Builder, account: str, days: int) -> None:
    rng = b.rng
    for _ in range(int(days * rng.uniform(1.5, 3.0))):  # ~2 customer payments a day
        b.add(b.at(rng.uniform(0, days)), account, f"CUS{rng.randint(1, 99999):05d}", "in", rng.uniform(20, 400), "ach")


def _cash_business(b: _Builder, account: str, days: int) -> None:
    rng = b.rng
    day = rng.uniform(0, 2)
    while day < days:
        b.add(b.at(day + rng.uniform(0.6, 0.8)), account, "CASH", "in", rng.uniform(2500, 9800), "cash")
        day += rng.uniform(1.5, 3.0)


def generate(seed: int = 7, n_accounts: int = 400, days: int = 60, bad_rate: float = 0.06) -> Dataset:
    rng = random.Random(seed)
    b = _Builder(rng)
    truth: dict[str, set[str]] = {}
    lookalikes: dict[str, str] = {}
    injectors = {STRUCTURING: _structuring, PASS_THROUGH: _pass_through, FAN_IN: _fan_in}
    benign = {"home_purchase": _home_purchase, "small_business": _small_business, "cash_business": _cash_business}

    for i in range(n_accounts):
        account = f"ACC{i:05d}"
        _background(b, account, days)
        roll = rng.random()
        if roll < bad_rate:
            typology = rng.choice(list(injectors))
            injectors[typology](b, account, days)
            truth[account] = {typology}
        elif roll < bad_rate + 0.06:
            kind = rng.choice(list(benign))
            benign[kind](b, account, days)
            lookalikes[account] = kind

    b.txns.sort(key=lambda t: (t.ts, t.account_id))
    txns = [
        Transaction(f"T{n:07d}", t.ts, t.account_id, t.counterparty_id, t.direction, t.amount, t.channel)
        for n, t in enumerate(b.txns, 1)
    ]
    return Dataset(transactions=txns, truth=truth, lookalikes=lookalikes)
