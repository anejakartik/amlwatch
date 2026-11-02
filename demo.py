"""One-command demo, no API key: generate data, stream it through the monitor,
score detection against ground truth, then draft and validate a SAR narrative.

    python demo.py [--seeds 7,1,2,3,4,5]
"""

from __future__ import annotations

import argparse

from amlwatch.evaluate import score, triage_report
from amlwatch.generator import generate
from amlwatch.models import STRUCTURING, TYPOLOGIES
from amlwatch.monitor import Monitor
from amlwatch.narrative import draft_narrative, validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,1,2,3,4,5")
    seeds = [int(s) for s in ap.parse_args().seeds.split(",")]

    totals = {t: {"injected": 0, "alerts": 0, "tp": 0, "lookalike_fp": 0} for t in TYPOLOGIES}
    n_txns = 0
    for seed in seeds:
        data = generate(seed=seed)
        n_txns += len(data.transactions)
        for typ, s in score(Monitor().run(data.transactions), data).items():
            totals[typ]["injected"] += s["injected"]
            totals[typ]["alerts"] += s["alerts"]
            totals[typ]["tp"] += s["true_positives"]
            totals[typ]["lookalike_fp"] += s["false_positives_on_lookalikes"]

    print(f"Detection over {len(seeds)} synthetic datasets ({n_txns:,} transactions, scored per account)\n")
    print(f"{'typology':<14}{'injected':>9}{'alerts':>8}{'recall':>8}{'precision':>11}{'FPs on look-alikes':>20}")
    for typ, t in totals.items():
        fp = t["alerts"] - t["tp"]
        print(
            f"{typ:<14}{t['injected']:>9}{t['alerts']:>8}{t['tp'] / t['injected']:>8.2f}"
            f"{t['tp'] / t['alerts']:>11.2f}{t['lookalike_fp']:>12} of {fp}"
        )

    # triage never sees the demo seeds: fitted on 11-15, margin picked on 16-20
    triage, report = triage_report(range(11, 16), range(16, 21), seeds)
    print(f"\nTriage on the same datasets, per alert (fitted on seeds 11-15, margin {triage.margin:.2f} from 16-20)\n")
    print(f"{'typology':<14}{'alerts':>8}{'kept':>6}{'real cases kept':>17}{'precision':>16}")
    tot = {"alerts": 0, "kept": 0, "real": 0, "real_kept": 0}
    for typ, r in report.items():
        for k in tot:
            tot[k] += r[k]
        precision = f"{r['precision_before']:.2f} -> {r['precision_after']:.2f}"
        print(f"{typ:<14}{r['alerts']:>8}{r['kept']:>6}{r['real_kept']:>11} of {r['real']:<3}{precision:>16}")
    print(
        f"{'all':<14}{tot['alerts']:>8}{tot['kept']:>6}{tot['real_kept']:>11} of {tot['real']:<3}"
        f"{'':>4}{1 - tot['kept'] / tot['alerts']:.0%} fewer alerts"
    )

    data = generate(seed=seeds[0])
    txns = {t.txn_id: t for t in data.transactions}
    alert = next(a for a in Monitor().run(data.transactions) if a.typology == STRUCTURING)
    draft = draft_narrative(alert, txns)
    print(f"\nSAR narrative draft for {alert.alert_id} ({draft.status}):\n")
    print(draft.text)

    # the same draft with one deposit's amount nudged, the way an LLM might misremember it
    first = txns[alert.evidence[0]]
    wrong = f"${first.amount + 50:,.2f}"
    tampered = draft.text.replace(f"${first.amount:,.2f}", wrong, 1)
    result = validate(tampered, alert, txns)
    print(f"\nSame draft with one amount changed to {wrong}: {'ready' if result.ok else 'REJECTED'}")
    for p in result.problems:
        print(f"  - {p.reason}")


if __name__ == "__main__":
    main()
