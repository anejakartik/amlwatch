from datetime import datetime, timedelta

import pytest

from amlwatch.models import Transaction

T0 = datetime(2026, 3, 2, 9, 0)


@pytest.fixture
def mk():
    """Build transactions with sequential ids: mk(hours, direction, amount, channel, counterparty)."""
    counter = {"n": 0}

    def _mk(hours, direction, amount, channel="wire", counterparty="CP1", account="ACC1"):
        counter["n"] += 1
        return Transaction(
            f"T{counter['n']:07d}", T0 + timedelta(hours=hours), account, counterparty, direction, amount, channel
        )

    return _mk
