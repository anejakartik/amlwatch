import types

import pytest

from amlwatch.generator import generate
from amlwatch.models import PASS_THROUGH, STRUCTURING, Alert
from amlwatch.monitor import Monitor
from amlwatch.narrative import ClaudeDrafter, draft_narrative, validate


@pytest.fixture
def case(mk):
    txns = [mk(0, "in", 9400, "cash", "CASH"), mk(20, "in", 8800, "cash", "CASH"), mk(50, "in", 9100, "cash", "CASH")]
    m = Monitor()
    for t in txns:
        m.ingest(t)
    alert = m.alerts[0]
    other = mk(60, "in", 5000)  # a real transaction that is not evidence for this alert
    return alert, {t.txn_id: t for t in [*txns, other]}, txns, other


def test_correct_sentence_passes(case):
    alert, txns, ev, _ = case
    cites = "".join(f"[{t.txn_id}]" for t in ev)
    text = f"The account received $27,300.00 in cash across three deposits {cites}."
    assert validate(text, alert, txns).ok


@pytest.mark.parametrize(
    "make_text, reason",
    [
        (lambda ev, o: "The account structured cash deposits to avoid reporting.", "no transaction cited"),
        (lambda ev, o: f"A deposit of $9,400.00 was made [{o.txn_id}].", "outside this alert"),
        (lambda ev, o: f"A deposit of $9,450.00 was made [{ev[0].txn_id}].", "doesn't match"),
        (lambda ev, o: f"A deposit was made on 2026-03-09 [{ev[0].txn_id}].", "isn't the date"),
        # right number, wrong transaction: $8,800 belongs to the second deposit
        (lambda ev, o: f"On 2026-03-02 a deposit of $8,800.00 was made [{ev[0].txn_id}].", "doesn't match"),
    ],
)
def test_unsupported_claims_are_rejected(case, make_text, reason):
    alert, txns, ev, other = case
    result = validate(make_text(ev, other), alert, txns)
    assert not result.ok
    assert reason in result.problems[0].reason


def test_ctr_threshold_is_an_allowed_reference_amount(case):
    alert, txns, ev, _ = case
    assert validate(f"Each deposit stayed below the $10,000 threshold [{ev[0].txn_id}].", alert, txns).ok


def test_per_direction_totals_are_allowed(mk):
    ev = [mk(0, "in", 40000), mk(3, "out", 18000, counterparty="D1"), mk(6, "out", 19000, counterparty="D2")]
    alert = Alert("A1", "ACC1", PASS_THROUGH, ev[-1].ts, [t.txn_id for t in ev])
    txns = {t.txn_id: t for t in ev}
    text = f"It received $40,000.00 [{ev[0].txn_id}] and sent $37,000.00 [{ev[1].txn_id}][{ev[2].txn_id}]."
    assert validate(text, alert, txns).ok


@pytest.mark.parametrize("seed", [7, 4])
def test_template_drafts_always_validate(seed):
    data = generate(seed=seed)
    txns = {t.txn_id: t for t in data.transactions}
    drafts = [draft_narrative(a, txns) for a in Monitor().run(data.transactions)]
    assert drafts and all(d.status == "ready" for d in drafts)


class _FakeClient:
    """Stands in for anthropic.Anthropic(); replays canned drafts and records calls."""

    def __init__(self, replies, stop_reason="end_turn"):
        self.replies = list(replies)
        self.calls = []
        self.stop_reason = stop_reason
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.replies.pop(0)
        return types.SimpleNamespace(
            stop_reason=self.stop_reason, content=[types.SimpleNamespace(type="text", text=text)]
        )


def test_claude_draft_is_retried_with_feedback_then_accepted(case):
    alert, txns, ev, _ = case
    bad = f"A cash deposit of $9,450.00 was made [{ev[0].txn_id}]."
    good = f"A cash deposit of $9,400.00 was made [{ev[0].txn_id}]."
    client = _FakeClient([bad, good])
    d = draft_narrative(alert, txns, drafter=ClaudeDrafter(client=client, model="claude-opus-5-5"))
    assert d.status == "ready" and d.attempts == 2 and d.text == good
    assert "failed these checks" in client.calls[1]["messages"][0]["content"]
    assert client.calls[0]["model"] == "claude-opus-5-5"


def test_claude_draft_that_never_validates_goes_to_review(case):
    alert, txns, ev, _ = case
    bad = "The customer is clearly laundering money."
    d = draft_narrative(alert, txns, drafter=ClaudeDrafter(client=_FakeClient([bad, bad])))
    assert d.status == "needs_review" and d.problems


def test_claude_refusal_is_raised(case):
    alert, txns, _, _ = case
    drafter = ClaudeDrafter(client=_FakeClient(["x"], stop_reason="refusal"))
    with pytest.raises(RuntimeError):
        drafter.draft(alert, txns)


def test_structuring_alerts_exist_in_demo_data():
    data = generate(seed=7)
    assert any(a.typology == STRUCTURING for a in Monitor().run(data.transactions))
