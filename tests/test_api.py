from fastapi.testclient import TestClient

from amlwatch import api


def client():
    api.STATE = api._State()
    return TestClient(api.app)


def _txn(i, hours, amount, channel="cash", direction="in"):
    return {
        "txn_id": f"T{i:07d}", "ts": f"2026-03-02T{hours:02d}:00:00", "account_id": "ACC9",
        "counterparty_id": "CASH", "direction": direction, "amount": amount, "channel": channel,
    }


def test_streaming_ingest_returns_alert_and_narrative(monkeypatch):
    monkeypatch.setenv("AMLWATCH_SEED_DEMO", "0")
    with client() as c:
        assert c.post("/transactions", json=[_txn(1, 9, 9400), _txn(2, 11, 8800)]).json() == []
        alerts = c.post("/transactions", json=[_txn(3, 13, 9100)]).json()
        assert [a["typology"] for a in alerts] == ["structuring"]
        alert_id = alerts[0]["alert_id"]
        assert "T0000001" in c.get(f"/alerts/{alert_id}").json()["evidence_table"]
        d = c.post(f"/alerts/{alert_id}/narrative").json()
        assert d["status"] == "ready" and "[T0000003]" in d["text"]


def test_rejects_duplicates_out_of_order_and_bad_input(monkeypatch):
    monkeypatch.setenv("AMLWATCH_SEED_DEMO", "0")
    with client() as c:
        c.post("/transactions", json=[_txn(1, 12, 100)])
        assert c.post("/transactions", json=[_txn(1, 13, 100)]).status_code == 409
        assert c.post("/transactions", json=[_txn(2, 8, 100)]).status_code == 422
        assert c.post("/transactions", json=[_txn(3, 14, -5)]).status_code == 422
        assert c.get("/alerts?typology=nope").status_code == 400
        assert c.get("/alerts/A99999-x").status_code == 404


def test_demo_data_is_seeded_on_startup(monkeypatch):
    monkeypatch.setenv("AMLWATCH_SEED_DEMO", "1")
    with client() as c:
        m = c.get("/metrics").json()
        assert m["transactions"] > 10000 and m["alerts"] > 0


def test_every_alert_gets_a_logged_triage_decision(monkeypatch):
    monkeypatch.setenv("AMLWATCH_SEED_DEMO", "1")
    with client() as c:
        m = c.get("/metrics").json()
        assert m["queued"] + m["suppressed"] == m["alerts"] and 0 < m["queued"] < m["alerts"]
        assert m["triage_model"] == api.STATE.triage.version

        queue = c.get("/queue?limit=500").json()
        assert len(queue) == m["queued"]
        scores = [a["triage"]["score"] for a in queue]
        assert scores == sorted(scores, reverse=True)

        hidden = c.get("/alerts?status=suppressed&limit=1000").json()
        assert len(hidden) == m["suppressed"]
        d = hidden[0]["triage"]
        assert not d["queued"] and d["score"] < d["threshold"] and d["reason"].startswith("suppressed")
        # contributions are listed most-negative first, and the reason names the first
        assert list(d["contributions"].values()) == sorted(d["contributions"].values())
        assert next(iter(d["contributions"])) in d["reason"]


def test_below_the_line_sample_is_suppressed_and_reproducible(monkeypatch):
    monkeypatch.setenv("AMLWATCH_SEED_DEMO", "1")
    with client() as c:
        a = c.get("/suppressed/sample?n=5&seed=3").json()
        assert len(a) == 5 and not any(x["triage"]["queued"] for x in a)
        assert [x["alert_id"] for x in a] == [x["alert_id"] for x in c.get("/suppressed/sample?n=5&seed=3").json()]


def test_out_of_order_transaction_leaves_no_trace(monkeypatch):
    monkeypatch.setenv("AMLWATCH_SEED_DEMO", "0")
    with client() as c:
        c.post("/transactions", json=[_txn(1, 12, 100)])
        assert c.post("/transactions", json=[_txn(2, 8, 100)]).status_code == 422
        assert "T0000002" not in api.STATE.txns
        assert c.post("/transactions", json=[_txn(2, 13, 100)]).status_code == 200


def test_dispositions_record_their_source_and_feed_the_missed_estimate(monkeypatch):
    monkeypatch.setenv("AMLWATCH_SEED_DEMO", "1")
    with client() as c:
        assert c.get("/metrics").json()["below_the_line"] == {"reviewed": 0, "escalated": 0, "estimated_missed": None}
        queued = c.get("/queue?limit=1").json()[0]["alert_id"]
        sample = [a["alert_id"] for a in c.get("/suppressed/sample?n=4&seed=1").json()]

        d = c.post(f"/alerts/{queued}/disposition", json={"outcome": "escalated", "reviewer": "inv1"}).json()
        assert d["source"] == "queue" and d["model_version"] == api.STATE.triage.version
        for i, alert_id in enumerate(sample):  # one of four sampled alerts turns out to be real
            outcome = "escalated" if i == 0 else "closed"
            r = c.post(f"/alerts/{alert_id}/disposition", json={"outcome": outcome, "reviewer": "inv2"})
            assert r.json()["source"] == "below_the_line"

        m = c.get("/metrics").json()
        assert m["below_the_line"] == {"reviewed": 4, "escalated": 1, "estimated_missed": round(m["suppressed"] / 4, 1)}
        assert len(c.get("/dispositions?source=below_the_line").json()) == 4
        assert len(c.get("/dispositions?source=queue").json()) == 1


def test_disposition_rejects_bad_input_and_double_labels(monkeypatch):
    monkeypatch.setenv("AMLWATCH_SEED_DEMO", "1")
    with client() as c:
        alert_id = c.get("/queue?limit=1").json()[0]["alert_id"]
        assert c.post(f"/alerts/{alert_id}/disposition", json={"outcome": "maybe", "reviewer": "x"}).status_code == 422
        assert c.post(f"/alerts/{alert_id}/disposition", json={"outcome": "closed", "reviewer": ""}).status_code == 422
        assert c.post("/alerts/A99999-x/disposition", json={"outcome": "closed", "reviewer": "x"}).status_code == 404
        assert c.post(f"/alerts/{alert_id}/disposition", json={"outcome": "closed", "reviewer": "x"}).status_code == 200
        assert c.post(f"/alerts/{alert_id}/disposition", json={"outcome": "closed", "reviewer": "x"}).status_code == 409
        assert c.get("/dispositions?source=nope").status_code == 400
