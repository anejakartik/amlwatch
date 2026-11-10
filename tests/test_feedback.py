"""The retraining loop: what a model learns when its only labels come through its own filter."""

import pytest

from amlwatch.evaluate import labelled_alerts
from amlwatch.feedback import simulate
from amlwatch.generator import generate
from amlwatch.triage import Triage


def test_disguise_off_changes_nothing():
    a, b = generate(seed=3), generate(seed=3, disguised=0.0)
    assert a.transactions == b.transactions and a.truth == b.truth


def test_disguised_cases_look_like_home_purchases():
    data = generate(seed=3, disguised=1.0)
    for acct, typs in data.truth.items():
        if "pass_through" in typs:
            wires_out = [
                t.counterparty_id for t in data.transactions
                if t.account_id == acct and t.direction == "out" and t.channel == "wire"
            ]
            assert wires_out == ["TITLE01"]


@pytest.fixture(scope="module")
def small_world():
    """A cut-down version of the README run: smaller start model, fewer months, fewer seeds."""
    start_rows = labelled_alerts(range(11, 13))
    start = Triage().fit(start_rows)
    start.margin = 0.15
    kw = dict(months=2, disguised=1.0, seeds_per_month=2, test_seeds=(7, 1), start=start, initial=start_rows)
    return {p: simulate(p, **kw) for p in ("queue_only", "queue_and_sample")}


def test_queue_only_never_reviews_what_it_suppresses(small_world):
    assert all(m.sampled == 0 and m.estimated_suppressed_real is None for m in small_world["queue_only"])


def test_sampling_finds_hidden_cases_and_retraining_on_them_catches_more(small_world):
    queue, sampled = small_world["queue_only"], small_world["queue_and_sample"]
    assert any(m.sampled_real for m in sampled)  # the sample surfaced real cases triage had hidden
    assert sampled[-1].test_pass_through_kept > queue[-1].test_pass_through_kept


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        simulate("retrain_on_vibes")
