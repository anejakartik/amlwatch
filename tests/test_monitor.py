import pytest

from amlwatch.evaluate import score
from amlwatch.generator import generate
from amlwatch.models import FAN_IN, PASS_THROUGH, STRUCTURING
from amlwatch.monitor import Monitor


def test_structuring_fires_on_the_deposit_that_crosses_the_line(mk):
    m = Monitor()
    assert m.ingest(mk(0, "in", 9400, "cash", "CASH")) == []
    assert m.ingest(mk(20, "in", 8800, "cash", "CASH")) == []
    alerts = m.ingest(mk(50, "in", 9100, "cash", "CASH"))
    assert [a.typology for a in alerts] == [STRUCTURING]
    assert alerts[0].facts["count"] == 3 and alerts[0].facts["total"] == 27300


def test_structuring_ignores_deposits_outside_the_band_or_window(mk):
    m = Monitor()
    m.ingest(mk(0, "in", 9400, "cash", "CASH"))
    m.ingest(mk(1, "in", 4000, "cash", "CASH"))  # below the band
    m.ingest(mk(2, "in", 12000, "cash", "CASH"))  # above the threshold: a CTR gets filed anyway
    assert m.ingest(mk(24 * 8, "in", 9000, "cash", "CASH")) == []  # first deposit is 8 days old


def test_pass_through_needs_most_of_a_large_inbound_to_leave(mk):
    m = Monitor()
    m.ingest(mk(0, "in", 50000))
    assert m.ingest(mk(5, "out", 20000, counterparty="D1")) == []
    alerts = m.ingest(mk(10, "out", 26000, counterparty="D2"))  # 92% out
    assert [a.typology for a in alerts] == [PASS_THROUGH]
    assert alerts[0].facts["out_ratio"] == 0.92


def test_pass_through_is_not_diluted_by_a_paycheck_in_the_window(mk):
    """Regression: the ratio used to be out / all inflows, so payroll hid the pattern."""
    m = Monitor()
    m.ingest(mk(0, "in", 30000))
    m.ingest(mk(1, "in", 4000, "ach", "EMP001"))
    alerts = m.ingest(mk(6, "out", 27600, counterparty="D1"))  # 92% of the wire, 81% of all inflows
    assert [a.typology for a in alerts] == [PASS_THROUGH]


def test_card_spend_does_not_count_as_funds_passing_through(mk):
    m = Monitor()
    m.ingest(mk(0, "in", 12000))
    for h in range(1, 12):
        assert m.ingest(mk(h, "out", 1000, "card", f"MER{h}")) == []


def test_fan_in_needs_many_new_senders(mk):
    m = Monitor()
    for i in range(9):
        assert m.ingest(mk(i, "in", 300, "ach", f"P{i}")) == []
    alerts = m.ingest(mk(10, "in", 300, "ach", "P9"))
    assert [a.typology for a in alerts] == [FAN_IN]
    assert alerts[0].facts["distinct_senders"] == 10


def test_cooldown_suppresses_duplicate_alerts(mk):
    m = Monitor()
    for i in range(12):
        m.ingest(mk(i, "in", 300, "ach", f"P{i}"))
    assert sum(a.typology == FAN_IN for a in m.alerts) == 1


def test_out_of_order_transactions_are_rejected(mk):
    m = Monitor()
    late = mk(5, "in", 100)
    early = mk(1, "in", 100)
    m.ingest(late)
    with pytest.raises(ValueError):
        m.ingest(early)


@pytest.mark.parametrize("seed", [7, 1, 2])
def test_every_injected_case_is_caught(seed):
    data = generate(seed=seed)
    result = score(Monitor().run(data.transactions), data)
    assert all(r["recall"] == 1.0 for r in result.values() if r["injected"])


def test_generator_is_deterministic():
    a, b = generate(seed=3), generate(seed=3)
    assert a.transactions == b.transactions and a.truth == b.truth
