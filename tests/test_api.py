import time

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_bootstrap_and_generation_use_bundled_assets():
    response = client.get("/api/bootstrap")
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"]["train_count"] == 5000
    assert body["dataset"]["validation_count"] == 500
    assert body["dataset"]["bundled"] is True
    assert body["model"]["parameter_count"] == 927232
    assert body["model"]["checkpoint_ready"] is True

    generated = client.post(
        "/api/generate",
        json={"prompt": "Once upon a time", "max_new_tokens": 3, "seed": 42},
    )
    assert generated.status_code == 200
    assert len(generated.json()["steps"]) == 3
    assert generated.json()["trace"]["attention"]["attention"]


def test_single_step_run_reaches_terminal_state_and_emits_metrics():
    created = client.post(
        "/api/runs",
        json={
            "name": "api-test",
            "preset": "inspect",
            "steps": 1,
            "batch_size": 2,
            "learning_rate": 0.0003,
            "model": {
                "vocab_size": 2048,
                "context_length": 8,
                "d_model": 32,
                "n_layers": 1,
                "n_heads": 4,
                "d_ff": 64,
                "dropout": 0,
            },
        },
    )
    assert created.status_code == 201
    run_id = created.json()["id"]
    for _ in range(100):
        summary = client.get(f"/api/runs/{run_id}").json()
        if summary["status"] in {"completed", "failed"}:
            break
        time.sleep(0.03)
    assert summary["status"] == "completed"
    assert summary["step"] == 1
    assert summary["last_metrics"]["loss"] > 0


def test_websocket_replays_run_events():
    created = client.post(
        "/api/runs",
        json={
            "name": "websocket-test",
            "preset": "inspect",
            "steps": 1,
            "batch_size": 1,
            "model": {
                "vocab_size": 2048,
                "context_length": 8,
                "d_model": 32,
                "n_layers": 1,
                "n_heads": 4,
                "d_ff": 64,
                "dropout": 0,
            },
        },
    )
    run_id = created.json()["id"]
    with client.websocket_connect(f"/ws/runs/{run_id}") as websocket:
        event = websocket.receive_json()
    assert event["type"] == "status"
    assert event["run_id"] == run_id
    for _ in range(100):
        summary = client.get(f"/api/runs/{run_id}").json()
        if summary["status"] in {"completed", "failed", "stopped"}:
            break
        time.sleep(0.03)


def test_websocket_heartbeat_and_activity_recovery():
    created = client.post(
        "/api/runs",
        json={
            "name": "heartbeat-test",
            "preset": "inspect",
            "steps": 100,
            "batch_size": 1,
            "model": {
                "vocab_size": 2048,
                "context_length": 8,
                "d_model": 32,
                "n_layers": 1,
                "n_heads": 4,
                "d_ff": 64,
                "dropout": 0,
            },
        },
    )
    assert created.status_code == 201
    run_id = created.json()["id"]
    client.post(f"/api/runs/{run_id}/control", json={"action": "pause"})

    active = client.get("/api/activity").json()
    assert active["microscope"]["id"] == run_id
    with client.websocket_connect(f"/ws/runs/{run_id}?after=9999") as websocket:
        heartbeat = websocket.receive_json()
    assert heartbeat["type"] == "heartbeat"
    assert heartbeat["summary"]["id"] == run_id
    assert heartbeat["telemetry"]["device"] == "cpu"

    client.post(f"/api/runs/{run_id}/control", json={"action": "stop"})


def test_websocket_marks_missing_runs_without_reconnect_ambiguity():
    with client.websocket_connect("/ws/runs/missing-run?after=12") as websocket:
        event = websocket.receive_json()
    assert event["type"] == "error"
    assert event["reason"] == "not_found"


def test_built_frontend_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Transformer Learning Studio" in response.text


def test_inspect_returns_selected_token_trace():
    response = client.post(
        "/api/inspect",
        json={"token_ids": [2, 10, 11, 12], "layer": 1, "head": 2, "token_index": 1},
    )
    assert response.status_code == 200
    trace = response.json()
    assert trace["attention"]["layer"] == 1
    assert trace["attention"]["head"] == 2
    assert trace["attention"]["token_index"] == 1
    assert len(trace["top_logits"]) == 5
