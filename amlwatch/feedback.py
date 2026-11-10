"""Retraining triage on investigator dispositions, and what it can't see.

In production the only labels are what investigators decide. They only look at alerts
triage queued, plus whatever random sample of suppressed alerts gets a below-the-line
review. A model retrained on that is learning from its own filter: if a laundering
pattern starts scoring under the threshold, nobody reviews it, so it never becomes a
training example, and the next model hides it too.

`simulate` replays that loop month by month on synthetic data where some of the
pass-through cases are disguised as home purchases (see `generator._disguised_pass_through`),
and reports what each retraining policy catches on a fixed test set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .evaluate import FIT_SEEDS, labelled_alerts, train_triage
from .generator import generate
from .models import PASS_THROUGH
from .monitor import Monitor
from .triage import Example, History, Triage

POLICIES = ("no_retrain", "queue_only", "queue_and_sample", "queue_and_sample_weighted")


def labelled(seeds, disguised: float) -> list[Example]:
    out: list[Example] = []
    for seed in seeds:
        data = generate(seed=seed, disguised=disguised)
        hist = History(data.transactions)
        for a in Monitor().run(data.transactions):
            out.append((a, hist, a.typology in data.truth.get(a.account_id, set())))
    return out


@dataclass
class Month:
    month: int
    suppressed: int
    suppressed_real: int  # ground truth; unknowable in production
    sampled: int
    sampled_real: int  # what the below-the-line review found
    test_real_kept: int  # on the fixed test set, after retraining on this month
    test_real: int
    test_pass_through_kept: int
    test_pass_through: int
    test_queued: int
    test_alerts: int

    @property
    def estimated_suppressed_real(self) -> float | None:
        """What the bank can actually know: scale the sample's hit rate up to everything suppressed."""
        return self.sampled_real / self.sampled * self.suppressed if self.sampled else None


def _score(model: Triage, test: list[Example]) -> tuple[int, int, int, int, int]:
    real_kept = real = pt_kept = pt = queued = 0
    for a, h, y in test:
        keep = model.keep(a, h)
        queued += keep
        real += y
        real_kept += y and keep
        if y and a.typology == PASS_THROUGH:
            pt += 1
            pt_kept += keep
    return real_kept, real, pt_kept, pt, queued


def simulate(
    policy: str,
    months: int = 6,
    disguised: float = 0.5,
    sample_rate: float = 0.1,
    seeds_per_month: int = 5,
    test_seeds=(7, 1, 2, 3, 4, 5),
    sample_seed: int = 0,
    start: Triage | None = None,
    initial: list[Example] | None = None,
) -> list[Month]:
    """Run `months` rounds of: triage decides, investigators label, retrain.

    Each month's labels are the queued alerts (investigator dispositions) and, unless the
    policy is `queue_only`, a `sample_rate` random sample of suppressed alerts. The weighted
    policy counts each sampled alert 1/sample_rate times, the textbook correction for
    sampling. The threshold margin stays at the shipped model's value throughout.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    model = start or train_triage()
    margin = model.margin
    pool = list(initial) if initial is not None else labelled_alerts(FIT_SEEDS)  # what the shipped model learned from
    weights = [1.0] * len(pool)
    test = labelled(test_seeds, disguised)
    rng = random.Random(sample_seed)
    out: list[Month] = []
    for m in range(months):
        month = labelled(range(100 + seeds_per_month * m, 100 + seeds_per_month * (m + 1)), disguised)
        suppressed = suppressed_real = sampled = sampled_real = 0
        for a, h, real in month:
            if model.keep(a, h):
                pool.append((a, h, real))
                weights.append(1.0)
                continue
            suppressed += 1
            suppressed_real += real
            if policy in ("queue_and_sample", "queue_and_sample_weighted") and rng.random() < sample_rate:
                sampled += 1
                sampled_real += real
                pool.append((a, h, real))
                weights.append(1 / sample_rate if policy == "queue_and_sample_weighted" else 1.0)
        if policy != "no_retrain":
            model = Triage().fit(pool, weights)
            model.margin = margin
        out.append(Month(m + 1, suppressed, suppressed_real, sampled, sampled_real, *_score(model, test), len(test)))
    return out


def main() -> None:
    """Print the month-by-month comparison in the README (takes a few minutes)."""
    print("Half of pass-through cases disguised as home purchases (one wire out to a title company)")
    print("Each month: triage decides, investigators review the queue and a 10% sample of suppressed alerts,")
    print("the model is retrained on what they labelled. Scored on the demo seeds after each month.\n")
    for policy in ("queue_only", "queue_and_sample", "queue_and_sample_weighted"):
        print(policy)
        print("  month  suppressed  real among them  sample found  estimate  test: pass-through caught  queue")
        for r in simulate(policy):
            est = "-" if r.estimated_suppressed_real is None else f"{r.estimated_suppressed_real:.0f}"
            found = f"{r.sampled_real} of {r.sampled}" if r.sampled else "-"
            print(
                f"  {r.month:5}  {r.suppressed:10}  {r.suppressed_real:15}  {found:>12}  {est:>8}  "
                f"{r.test_pass_through_kept:>17} of {r.test_pass_through}  {r.test_queued:5}"
            )
        print()


if __name__ == "__main__":
    main()
