"""SAR narrative drafting, with every claim checked against the transactions.

A Suspicious Activity Report narrative is the part an investigator writes by
hand: who did what, when, how much, and why it looks suspicious. Drafting it is
a good fit for an LLM and a dangerous one, because a fluent paragraph with one
wrong amount or date is worse than no draft at all.

So a draft is only "ready" if it passes `validate()`:
  1. every sentence cites at least one transaction, as [T0001234]
  2. every cited transaction belongs to the alert's evidence
  3. every dollar amount in a sentence equals one cited transaction's amount,
     the sum of the cited transactions, the sum of the cited inflows or outflows,
     or a named regulatory reference amount (the $10,000 CTR threshold)
  4. every date (YYYY-MM-DD) in a sentence is the date of a cited transaction

A draft that fails is returned as `needs_review` with the failing sentences,
never as ready. Two drafters share this contract: a deterministic template
(no key needed, used in tests and the demo) and Claude.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .models import FAN_IN, PASS_THROUGH, STRUCTURING, Alert, Transaction

CITE_RE = re.compile(r"\[(T\d{7})\]")
MONEY_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")

# Amounts a narrative may state without a transaction behind them. Kept tiny and
# explicit: anything added here is a number the validator stops checking.
REFERENCE_AMOUNTS = {10_000.00}  # Currency Transaction Report threshold


@dataclass
class Problem:
    sentence: str
    reason: str


@dataclass
class ValidationResult:
    ok: bool
    problems: list[Problem] = field(default_factory=list)


@dataclass
class Draft:
    alert_id: str
    drafter: str
    text: str
    status: str  # "ready" | "needs_review"
    problems: list[Problem] = field(default_factory=list)
    attempts: int = 1


def money(x: float) -> str:
    return f"${x:,.2f}"


def _sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        # section headings ("Activity:") carry no claims
        if not line or line.endswith(":") or line.startswith("#"):
            continue
        out.extend(s.strip() for s in SENTENCE_RE.split(line) if s.strip())
    return out


def validate(text: str, alert: Alert, txns: dict[str, Transaction]) -> ValidationResult:
    evidence = set(alert.evidence)
    problems: list[Problem] = []
    for sentence in _sentences(text):
        cited = CITE_RE.findall(sentence)
        if not cited:
            problems.append(Problem(sentence, "no transaction cited"))
            continue
        foreign = [c for c in cited if c not in evidence]
        if foreign:
            problems.append(Problem(sentence, f"cites transactions outside this alert: {', '.join(foreign)}"))
            continue
        cited_txns = [txns[c] for c in dict.fromkeys(cited)]
        allowed = {round(t.amount, 2) for t in cited_txns} | REFERENCE_AMOUNTS
        allowed.add(round(sum(t.amount for t in cited_txns), 2))
        for direction in ("in", "out"):
            leg = [t.amount for t in cited_txns if t.direction == direction]
            if leg:
                allowed.add(round(sum(leg), 2))
        for raw in MONEY_RE.findall(sentence):
            value = round(float(raw.replace(",", "")), 2)
            if not any(abs(value - a) < 0.01 for a in allowed):
                problems.append(Problem(sentence, f"amount ${raw} doesn't match the cited transactions"))
        dates = {t.ts.date().isoformat() for t in cited_txns}
        for d in DATE_RE.findall(sentence):
            if d not in dates:
                problems.append(Problem(sentence, f"date {d} isn't the date of a cited transaction"))
    return ValidationResult(ok=not problems, problems=problems)


def evidence_table(alert: Alert, txns: dict[str, Transaction]) -> str:
    rows = ["txn_id | date | time | direction | channel | counterparty | amount"]
    for tid in alert.evidence:
        t = txns[tid]
        rows.append(
            f"{t.txn_id} | {t.ts.date().isoformat()} | {t.ts.strftime('%H:%M')} | {t.direction} | "
            f"{t.channel} | {t.counterparty_id} | {money(t.amount)}"
        )
    return "\n".join(rows)


# -- template drafter ----------------------------------------------------------


def _cite(ts: list[Transaction]) -> str:
    return "".join(f"[{t.txn_id}]" for t in ts)


def _span(first: Transaction, last: Transaction) -> str:
    a, b = first.ts.date(), last.ts.date()
    return f"on {a}" if a == b else f"between {a} and {b}"


def template_draft(alert: Alert, txns: dict[str, Transaction]) -> str:
    ev = [txns[t] for t in alert.evidence]
    first, last = ev[0], ev[-1]
    acct = alert.account_id
    lines = ["Summary:"]
    if alert.typology == STRUCTURING:
        lines.append(
            f"Account {acct} received {len(ev)} cash deposits totalling {money(sum(t.amount for t in ev))} "
            f"{_span(first, last)} {_cite(ev)}."
        )
        lines.append("Activity:")
        for t in ev:
            lines.append(f"On {t.ts.date()} the account received a cash deposit of {money(t.amount)} [{t.txn_id}].")
        lines.append("Reason for suspicion:")
        lines.append(
            f"Each deposit is below the $10,000 currency transaction reporting threshold while the combined "
            f"total of {money(sum(t.amount for t in ev))} is not {_cite(ev)}."
        )
    elif alert.typology == PASS_THROUGH:
        ins = [t for t in ev if t.direction == "in"]
        outs = [t for t in ev if t.direction == "out"]
        lines.append(
            f"Account {acct} received {money(sum(t.amount for t in ins))} {_cite(ins)} and sent "
            f"{money(sum(t.amount for t in outs))} back out {_cite(outs)}."
        )
        lines.append("Activity:")
        for t in ev:
            verb = "received" if t.direction == "in" else "sent"
            prep = "from" if t.direction == "in" else "to"
            lines.append(
                f"On {t.ts.date()} the account {verb} a {t.channel} of {money(t.amount)} {prep} "
                f"{t.counterparty_id} [{t.txn_id}]."
            )
        lines.append("Reason for suspicion:")
        lines.append(
            f"At least 90% of the incoming funds left the account within 48 hours of arriving, consistent "
            f"with the account being used to move money rather than hold it {_cite(ev)}."
        )
    elif alert.typology == FAN_IN:
        senders = sorted({t.counterparty_id for t in ev})
        lines.append(
            f"Account {acct} received {len(ev)} payments totalling {money(sum(t.amount for t in ev))} from "
            f"{len(senders)} counterparties with no prior history with the account {_cite(ev)}."
        )
        lines.append("Reason for suspicion:")
        lines.append(
            f"Many unrelated senders converging on one account {_span(first, last)} is consistent "
            f"with a collection or mule account [{first.txn_id}][{last.txn_id}]."
        )
    else:  # pragma: no cover - every typology above
        raise ValueError(alert.typology)
    return "\n".join(lines)


# -- Claude drafter ------------------------------------------------------------

SYSTEM_PROMPT = """You draft the narrative section of a Suspicious Activity Report for a bank's \
financial-crimes investigator, who will review and edit it before anything is filed.

Use only the transactions in the evidence table. Every sentence must cite the transactions it \
relies on by id in square brackets, e.g. [T0001234]. Every dollar amount you write must be either \
one cited transaction's amount or the exact sum of the transactions cited in that sentence, \
formatted like $9,450.00. Write dates as YYYY-MM-DD, and only dates of transactions cited in the \
same sentence. Do not speculate about identities, motives or facts outside the table; describe the \
pattern and why it matches the typology.

Use three sections with these headings on their own lines: "Summary:", "Activity:", \
"Reason for suspicion:". Plain text, no markdown."""


class ClaudeDrafter:
    """Drafts with Claude via the official SDK. Needs ANTHROPIC_API_KEY (or an `ant` login)."""

    name = "claude"

    def __init__(self, client=None, model: str | None = None):
        if client is None:
            import anthropic  # imported lazily so the template path needs no SDK

            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.getenv("AMLWATCH_MODEL", "claude-opus-5-5")

    def draft(self, alert: Alert, txns: dict[str, Transaction], feedback: list[Problem] | None = None) -> str:
        prompt = (
            f"Typology flagged by the monitor: {alert.typology}\n"
            f"Account: {alert.account_id}\n\nEvidence table:\n{evidence_table(alert, txns)}"
        )
        if feedback:
            issues = "\n".join(f"- {p.reason}: {p.sentence}" for p in feedback)
            prompt += f"\n\nYour previous draft failed these checks. Fix every one:\n{issues}"
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            # if a safety classifier declines, retry on a fallback model server-side
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined to draft this narrative")
        return "".join(b.text for b in response.content if b.type == "text").strip()


def draft_narrative(alert: Alert, txns: dict[str, Transaction], drafter=None, max_attempts: int = 2) -> Draft:
    """Draft, validate, and (for an LLM) retry once with the failures as feedback."""
    if drafter is None:
        text = template_draft(alert, txns)
        result = validate(text, alert, txns)
        return Draft(alert.alert_id, "template", text, "ready" if result.ok else "needs_review", result.problems)

    problems: list[Problem] | None = None
    text = ""
    for attempt in range(1, max_attempts + 1):
        text = drafter.draft(alert, txns, feedback=problems)
        result = validate(text, alert, txns)
        if result.ok:
            return Draft(alert.alert_id, drafter.name, text, "ready", [], attempt)
        problems = result.problems
    return Draft(alert.alert_id, drafter.name, text, "needs_review", problems or [], max_attempts)
