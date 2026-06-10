from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_system_and_factory_bootstrap_contracts():
    system = client.get("/api/system")
    assert system.status_code == 200
    assert "production_ready" in system.json()
    bootstrap = client.get("/api/bootstrap").json()
    assert bootstrap["factory"]["model"]["parameter_count"] > 15_000_000
    assert bootstrap["factory"]["model"]["n_kv_heads"] == 2
    assert bootstrap["factory"]["data"]["sft_count"] == 1200


def test_scale_estimator_and_cuda_guard():
    estimate = client.post(
        "/api/scale/estimate",
        json={
            "parameters_billions": 7,
            "tokens_billions": 140,
            "context_length": 4096,
            "precision_bytes": 2,
            "gpu_memory_gb": 80,
            "gpu_tflops": 312,
            "utilization": 0.4,
        },
    )
    assert estimate.status_code == 200
    assert estimate.json()["minimum_training_gpus"] >= 1
    if not client.get("/api/system").json()["production_ready"]:
        blocked = client.post("/api/factory/runs", json={})
        assert blocked.status_code == 409
        assert "CUDA" in blocked.json()["detail"]

