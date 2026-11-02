# amlwatch

**Streaming anti-money-laundering (AML) transaction monitoring, with SAR narrative drafts that can't
state a number the transactions don't support.**

Banks monitor every transaction for laundering patterns, and when an alert holds up, an
investigator writes a Suspicious Activity Report (SAR). The narrative section (who did what, when, how
much, and why it looks suspicious) is written by hand, and it's an obvious job for an LLM. It's
also a risky one: a fluent paragraph with one wrong amount or date is worse than no draft.

amlwatch does both halves:

1. **Monitor:** transactions stream in one at a time and three typology rules run incrementally
   per account, so an alert fires on the transaction that completes a pattern, not in a nightly batch.
2. **Draft:** a SAR narrative is drafted for the alert (deterministic template, or Claude), and is only
   marked `ready` if every sentence cites the transactions it relies on and every amount and date in
   it matches those transactions. Anything else goes back as `needs_review`, with the failing sentences.

All data is synthetic. No real customer data exists anywhere in this repo.

## Quickstart (no API key)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python demo.py
```

```
Detection over 6 synthetic datasets (155,605 transactions, scored per account)

typology       injected  alerts  recall  precision  FPs on look-alikes
structuring          46      70    1.00       0.66          24 of 24
pass_through         44      88    1.00       0.50          44 of 44
fan_in               41      86    1.00       0.48          45 of 45

Triage on the same datasets, per alert (fitted on seeds 11-15, margin 0.15 from 16-20)

typology        alerts  kept  real cases kept       precision
structuring         74    53         46 of 46     0.62 -> 0.87
pass_through        88    47         43 of 44     0.50 -> 0.92
fan_in             245    57         41 of 41     0.17 -> 0.72
all                407   157        130 of 131    61% fewer alerts
```

…followed by a SAR draft for a structuring alert, and the same draft with one deposit amount nudged by
$50, which the validator rejects.

## What it detects

| Typology | Rule (defaults, all configurable) | Why it matters |
| --- | --- | --- |
| `structuring` | ≥ 3 cash deposits of $8,000–$9,999.99 within 7 days, totalling > $10,000 | Each deposit stays under the $10,000 Currency Transaction Report threshold; together they don't |
| `pass_through` | One inbound ≥ $10,000, with ≥ 90% of it sent back out (non-card) within 48h | The account moves money instead of holding it |
| `fan_in` | ≥ 10 distinct, previously unseen senders within 3 days | Many unrelated payers converging on one account: a collection or mule pattern |

### How to read those numbers

Every injected case is caught, and **roughly half the pass-through and fan-in alerts are
legitimate customers.** That's on purpose. The generator plants legitimate look-alikes next to the
laundering patterns:

- **home purchases**, where a transfer arrives from a brokerage and a wire leaves for a title company
  the next day (looks like pass-through)
- **small businesses** getting steady payments from many customers (looks like fan-in)
- **cash-heavy businesses** depositing takings every couple of days (looks like structuring)

A monitor tested only against clean negatives looks far better than it is. High recall with a heavy
false-positive load is what rules-based transaction monitoring looks like in practice, and it's the
problem triage goes after (below).

The background population here is cleaner than real retail data, so treat the precision numbers as
an upper bound for these rules, not a benchmark.

## Triage (v0.2)

The rules stay as they are, so they keep catching everything. Triage scores each alert and
suppresses the ones that look like known-legitimate behavior. A suppressed alert is still logged; it
just isn't put in front of an investigator.

**Model.** One small logistic regression per typology (`amlwatch/triage.py`, pure Python), trained on
alert outcomes. Here the generator's ground truth stands in for investigator dispositions.

**Features: only what the bank has when the alert fires, and never a counterparty's name.**

| Typology | Features |
| --- | --- |
| `structuring` | cash deposits in the 30 days before the pattern, deposits in the pattern, total vs. the CTR threshold |
| `pass_through` | number of outbound legs, how many *other* customers had already paid the destination, days to the first outbound, out/in ratio |
| `fan_in` | distinct senders in the 30 days before the burst, distinct senders in the burst, mean amount |

The synthetic counterparties are named by kind (`TITLE01`, `BRK..`, `CUS..`), so a model allowed to
read ids would learn the generator, not the behavior. A test renames every counterparty and checks
the features don't change.

**Threshold: recall first.** Each typology's threshold starts just under the lowest-scoring real case
in training. That alone isn't enough: on new data a real case can score a little lower than anything
the model was fitted on. So the margin below it is picked on a separate validation set (the smallest
that keeps every real validation case), and the numbers above come from a third set of seeds the
model never saw. Fit on 11-15, margin from 16-20, report on the demo's seeds.

**What it gets right, and what it can't do.**

- Precision is scored **per alert**, because that's the investigator's workload. Per alert, fan-in
  precision before triage was 0.17, not the 0.48 per-account figure above, because a small
  business re-alerts every week once the cooldown runs out.
- **Every false positive it keeps on structuring and fan-in is a customer the bank had known for
  less than a week.** The features ask "is this normal for this customer?", and for a brand-new
  customer there's no answer. Every account in the synthetic data opens on day 0, so this
  cold-start effect is exaggerated here, but it's real: new accounts are exactly where a bank knows
  the least.
- **It missed one real case out of 131.** The case was a pass-through with one outbound wire, sent 1.5 days
  later to a destination no other customer had paid. With these features that's the same as a
  home purchase through a title company nobody at the bank has used before. On these seeds, lowering the
  threshold far enough to catch it would cost one extra false alert, but that's hindsight: a threshold
  moved after looking at the test set makes the test result meaningless. The real fix is better
  features, such as whether the destination is a registered business, which this synthetic data
  doesn't have.
- Of the 27 false positives it keeps, 23 are on customers known for under a week; the other 4 are home
  purchases.

## The narrative contract

A draft is `ready` only if `validate()` passes every sentence:

1. it cites at least one transaction, as `[T0001234]`
2. every cited transaction belongs to the alert's evidence
3. every dollar amount equals a cited transaction's amount, the sum of the cited transactions, the sum
   of the cited inflows or outflows, or a named reference amount (only the $10,000 CTR threshold)
4. every `YYYY-MM-DD` date is the date of a cited transaction

The Claude drafter gets the evidence table and these rules. If its draft fails, it gets one retry with
the failing sentences as feedback. If the retry also fails, it's returned as `needs_review`, never
as `ready`.

The validator checks the evidence. It doesn't judge whether the activity is suspicious. A legitimate
home purchase that trips the pass-through rule still gets an accurate, fully cited draft. Deciding to
file is the investigator's job, and the draft exists to save them the typing, not the judgment.

## API

```bash
uvicorn amlwatch.api:app --port 8000   # seeds the synthetic dataset on startup
```

| Endpoint | |
| --- | --- |
| `POST /transactions` | Ingest a batch in timestamp order; returns any alerts it completes. Out-of-order → 422, duplicate id → 409 |
| `GET /alerts?typology=` | List alerts |
| `GET /alerts/{id}` | Alert with its evidence table |
| `POST /alerts/{id}/narrative?drafter=template\|claude` | Draft and validate a SAR narrative |
| `GET /metrics` | Transaction and alert counts |

`drafter=claude` uses the official `anthropic` SDK with `ANTHROPIC_API_KEY`. The model defaults to
`claude-opus-5-5` (override with `AMLWATCH_MODEL`), and server-side refusal fallback is enabled.
Set `AMLWATCH_SEED_DEMO=0` to start with an empty monitor.

## Tests

```bash
ruff check . && pytest
```

Rule edge cases (band and window boundaries, card spend, cooldown, out-of-order input), a regression
test for paycheck dilution of the pass-through ratio, adversarial drafts (no citation, foreign
citation, wrong amount, wrong date, right amount but wrong transaction), the Claude retry/review path
against a fake client, the API, and triage (features never read counterparty names, reach only counts
other customers and only the past, the threshold keeps every real training case, and a held-out run).

## Roadmap

- Triage in the API: score alerts on ingest and expose a ranked queue.
- Counterparty features beyond reach: registered-business flag, account age, linked accounts.
- Kafka consumer in front of `Monitor.ingest` and durable alert storage (Postgres).
- More typologies: round-dollar wires, dormant-account reactivation, rapid cross-border movement.
- Case view UI: alert, evidence table and draft side by side, with validator flags inline.

## License

MIT
