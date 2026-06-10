import threading
import time

import pytest
import torch

from backend.app import factory_engine
from backend.app.engine import TrainingManager
from backend.app.factory_engine import FactoryManager, FactoryRunState
from backend.app.schemas import FactoryRunConfig, TrainingConfig


def test_microscope_rejects_a_second_active_run(monkeypatch):
    manager = TrainingManager()
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)

    first = manager.create_run(TrainingConfig(steps=2))

    with pytest.raises(RuntimeError, match="이미 학습이 진행 중"):
        manager.create_run(TrainingConfig(steps=2))

    assert manager.active_run().id == first["id"]


def test_paused_microscope_elapsed_time_does_not_advance(monkeypatch):
    manager = TrainingManager()
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    created = manager.create_run(TrainingConfig(steps=2))
    state = manager.get_run(created["id"])
    state.step = 1

    manager.control(state.id, "pause")
    before = state.summary()
    time.sleep(0.03)
    after = state.summary()

    assert after["elapsed_seconds"] == pytest.approx(
        before["elapsed_seconds"], abs=0.005
    )
    assert after["eta_seconds"] == pytest.approx(before["eta_seconds"], abs=0.005)


def test_factory_summary_has_stage_states_progress_and_metric_series(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(factory_engine, "FACTORY_DIR", tmp_path)
    monkeypatch.setattr(factory_engine, "FACTORY_HISTORY", tmp_path / "history.json")
    manager = FactoryManager()
    config = FactoryRunConfig(stages=["pretrain"], pretrain_steps=10)
    state = FactoryRunState(
        id="active",
        config=config,
        stage="pretrain",
        stage_step=4,
        stage_total=10,
        stage_totals=manager._planned_stage_totals(config),
    )
    manager._record_metric(state, "pretrain", {"step": 4, "loss": 2.5})

    summary = state.summary()

    assert summary["status"] == "running"
    assert summary["stage_states"]["pretrain"] == "running"
    assert summary["stage_states"]["sft"] == "skipped"
    assert summary["stage_progress"] == 0.4
    assert summary["overall_progress"] == 0.4
    assert summary["metric_series"][0]["loss"] == 2.5


def test_factory_rejects_a_second_active_run(tmp_path, monkeypatch):
    monkeypatch.setattr(factory_engine, "FACTORY_DIR", tmp_path)
    monkeypatch.setattr(factory_engine, "FACTORY_HISTORY", tmp_path / "history.json")
    monkeypatch.setattr(
        factory_engine,
        "system_diagnostics",
        lambda: {"production_ready": True, "setup_command": ""},
    )
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    manager = FactoryManager()

    first = manager.create_run(FactoryRunConfig(stages=["data"]))

    with pytest.raises(RuntimeError, match="이미 학습이 진행 중"):
        manager.create_run(FactoryRunConfig(stages=["data"]))

    assert manager.active_run().id == first["id"]


def test_factory_stage_event_uses_stage_status(tmp_path, monkeypatch):
    monkeypatch.setattr(factory_engine, "FACTORY_DIR", tmp_path)
    monkeypatch.setattr(factory_engine, "FACTORY_HISTORY", tmp_path / "history.json")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    manager = FactoryManager()
    config = FactoryRunConfig(stages=["data"])
    state = FactoryRunState(
        id="stage-event",
        config=config,
        stage_totals=manager._planned_stage_totals(config),
    )
    monkeypatch.setattr(manager, "_stage_data", lambda current: None)

    manager._worker(state)

    stage_events = [event for event in state.events if event["type"] == "stage"]
    assert stage_events[0]["stage_status"] == "running"
    assert "status" not in stage_events[0]
    assert stage_events[-1]["stage_status"] == "completed"
