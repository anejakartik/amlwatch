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
