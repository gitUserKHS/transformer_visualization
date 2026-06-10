from backend.app import factory_engine
from backend.app.factory_engine import FactoryManager


def test_factory_artifacts_keep_only_history_and_active_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(factory_engine, "FACTORY_DIR", tmp_path)
    for run_id in ("old", "one", "two", "three", "active"):
        (tmp_path / run_id).mkdir()

    manager = FactoryManager.__new__(FactoryManager)
    manager.history = [{"id": "one"}, {"id": "two"}, {"id": "three"}]
    manager.runs = {"active": type("Run", (), {"status": "running"})()}
    manager._prune_artifacts()

    assert not (tmp_path / "old").exists()
    assert all((tmp_path / run_id).exists() for run_id in ("one", "two", "three", "active"))
