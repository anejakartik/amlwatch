from dataclasses import replace
from datetime import timedelta

import pytest

from amlwatch import evaluate as evaluate_mod
from amlwatch import triage as triage_mod
from amlwatch.evaluate import labelled_alerts, online_parity, train_triage, triage_report
from amlwatch.models import PASS_THROUGH, STRUCTURING, Alert
from amlwatch.train import MODEL_PATH
from amlwatch.triage import History, LogReg, Triage, features


def _alert(typology, evidence, account="ACC1"):
    return Alert("A1", account, typology, evidence[-1].ts, [t.txn_id for t in evidence])


def _home_purchase_scene(mk):
    """Two earlier customers paid the title company; then ACC1 buys a house."""
    earlier = [
        mk(0, "out", 40000, counterparty="TITLE", account="ACC2"),
        mk(5, "out", 52000, counterparty="TITLE", account="ACC3"),
    ]
    wire_in = mk(100, "in", 60000, counterparty="BRK")
    wire_out = mk(120, "out", 57000, counterparty="TITLE")
    return [*earlier, wire_in, wire_out], [wire_in, wire_out]


def test_features_ignore_counterparty_names(mk):
    """The generator names counterparties by kind; a model reading names would learn the generator."""
    txns, evidence = _home_purchase_scene(mk)
    renamed = {"TITLE": "X9", "BRK": "Q1"}
    txns2 = [replace(t, counterparty_id=renamed.get(t.counterparty_id, t.counterparty_id)) for t in txns]
    a = _alert(PASS_THROUGH, evidence)
    assert features(a, History(txns)) == features(a, History(txns2))


def test_destination_reach_counts_other_customers_before_the_alert(mk):
    txns, evidence = _home_purchase_scene(mk)
    later = mk(500, "out", 45000, counterparty="TITLE", account="ACC4")  # after the alert: not known yet
    hist = History([*txns, later])
    assert hist.other_payers_before("TITLE", "ACC1", evidence[-1].ts) == 2  # ACC2, ACC3; not ACC1 or ACC4


def test_structuring_history_only_looks_back_30_days(mk):
    old = mk(0, "in", 3000, "cash", "CASH")  # 40 days before the pattern
    recent = mk(24 * 35, "in", 4000, "cash", "CASH")
    pattern = [mk(24 * 40 + h, "in", 9000, "cash", "CASH") for h in (0, 5, 10)]
    a = _alert(STRUCTURING, pattern)
    prior_cash = features(a, History([old, recent, *pattern]))[0]
    assert prior_cash == features(a, History([recent, *pattern]))[0]  # the 40-day-old deposit doesn't count


def test_logreg_separates_separable_data():
    X = [[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]]
    m = LogReg().fit(X, [0, 0, 0, 1, 1, 1])
    assert all(m.predict(x) < 0.5 for x in X[:3]) and all(m.predict(x) > 0.5 for x in X[3:])


def test_threshold_keeps_every_real_training_case(mk):
    txns, evidence = _home_purchase_scene(mk)
    hist = History(txns)
    alerts = [(_alert(PASS_THROUGH, evidence), hist, False)]
    # one real case per destination reach, so the model has something to separate
    for i, legs in enumerate((2, 3)):
        a_in = mk(200 + i * 50, "in", 50000, counterparty=f"S{i}", account=f"BAD{i}")
        outs = [
            mk(201 + i * 50 + k, "out", 47000 / legs, counterparty=f"D{i}{k}", account=f"BAD{i}") for k in range(legs)
        ]
        txns += [a_in, *outs]
        alerts.append((_alert(PASS_THROUGH, [a_in, *outs], account=f"BAD{i}"), None, True))
    hist = History(txns)
    examples = [(a, hist, real) for a, _, real in alerts]
    t = Triage().fit(examples)
    assert all(t.keep(a, h) for a, h, real in examples if real)


def test_triage_cuts_alerts_on_unseen_seeds():
    triage, report = triage_report([11, 12], [13], [7])
    alerts = sum(r["alerts"] for r in report.values())
    kept = sum(r["kept"] for r in report.values())
    real = sum(r["real"] for r in report.values())
    real_kept = sum(r["real_kept"] for r in report.values())
    assert kept < alerts * 0.6
    assert real_kept >= real - 1


def test_live_features_match_the_offline_evaluation():
    """Features computed as transactions stream in equal the ones the offline numbers came from."""
    checked, mismatched = online_parity([7])
    assert checked > 50 and mismatched == []


def test_parity_check_catches_a_feature_that_reads_the_future(monkeypatch):
    """The check has teeth: a feature that peeks a day past the alert fails it."""
    real = triage_mod.features

    def leaky(alert, hist):
        x = real(alert, hist)
        later = hist.account_between(alert.account_id, alert.detected_at, alert.detected_at + timedelta(days=1))
        return [*x, len(later)]

    monkeypatch.setattr(evaluate_mod, "features", leaky)
    _, mismatched = online_parity([7])
    assert mismatched


def test_saved_model_scores_identically_and_replays_decisions(tmp_path):
    triage, _ = triage_report([11, 12], [13], [7])
    triage.save(tmp_path / "m.json")
    loaded = Triage.load(tmp_path / "m.json")
    assert loaded.version == triage.version

    for a, h, _ in labelled_alerts([7])[:40]:
        d = triage.decide(a, h)
        assert loaded.decide(a, h) == d
        assert loaded.replay(d) == d.score  # the logged inputs alone reproduce the score

    other = Triage.from_dict({**triage.to_dict(), "margin": triage.margin + 0.05})
    with pytest.raises(ValueError, match="decision was made by model"):
        other.replay(d)


def test_shipped_model_matches_its_training_recipe():
    """amlwatch/triage_model.json must be what `python -m amlwatch.train` produces, not a stale copy."""
    shipped, fresh = Triage.load(MODEL_PATH), train_triage()
    assert shipped.margin == fresh.margin
    for typ, m in fresh.models.items():
        assert shipped.models[typ].w == pytest.approx(m.w, rel=1e-9)
        assert shipped.base[typ] == pytest.approx(fresh.base[typ], rel=1e-9)


def test_model_trained_on_other_features_is_refused():
    d = Triage.load(MODEL_PATH).to_dict()
    d["typologies"][PASS_THROUGH]["features"] = ["out_legs"]
    with pytest.raises(ValueError, match="was trained on"):
        Triage.from_dict(d)
