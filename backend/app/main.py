import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .dataset import ROOT
from .engine import TrainingManager
from .factory_engine import FactoryManager
from .schemas import (
    FactoryRunConfig,
    GenerateRequest,
    InspectRequest,
    KVBenchmarkRequest,
    PreferenceVote,
    RunControl,
    ScaleEstimateRequest,
    TrainingConfig,
)
from .system_info import gpu_telemetry, system_diagnostics


app = FastAPI(
    title="Transformer Learning Studio",
    description="실제 GPT형 모델의 학습과 추론을 시각화하는 교육용 API",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
manager = TrainingManager()
factory = FactoryManager()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/bootstrap")
def bootstrap() -> dict:
    return {
        **manager.bootstrap(),
        "system": system_diagnostics(),
        "factory": factory.bootstrap(),
    }


@app.get("/api/system")
def system() -> dict:
    return system_diagnostics()


@app.get("/api/activity")
def activity() -> dict:
    microscope = manager.active_run()
    factory_run = factory.active_run()
    return {
        "server_time": time.time(),
        "microscope": microscope.summary() if microscope else None,
        "factory": factory_run.summary() if factory_run else None,
    }


@app.post("/api/runs", status_code=201)
def create_run(config: TrainingConfig) -> dict:
    try:
        return manager.create_run(config)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    try:
        return manager.get_run(run_id).summary()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/control")
def control_run(run_id: str, control: RunControl) -> dict:
    try:
        return manager.control(run_id, control.action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/generate")
def generate(request: GenerateRequest) -> dict:
    return manager.generate(request)


@app.post("/api/inspect")
def inspect(request: InspectRequest) -> dict:
    try:
        return manager.inspect(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/factory/runs", status_code=201)
def create_factory_run(config: FactoryRunConfig) -> dict:
    try:
        return factory.create_run(config)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/factory/runs/{run_id}")
def get_factory_run(run_id: str) -> dict:
    try:
        return factory.get_run(run_id).summary()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="모델 공장 실행을 찾을 수 없습니다.") from exc


@app.post("/api/factory/runs/{run_id}/control")
def control_factory_run(run_id: str, control: RunControl) -> dict:
    try:
        return factory.control(run_id, control.action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="모델 공장 실행을 찾을 수 없습니다.") from exc


@app.get("/api/preferences/next")
def next_preference() -> dict:
    return factory.data.next_preference()


@app.post("/api/preferences")
def submit_preference(vote: PreferenceVote) -> dict:
    try:
        record = factory.data.append_vote(vote.pair_id, vote.choice)
        return {
            "saved": True,
            "record": record,
            "user_vote_count": len(factory.data.user_votes()),
        }
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="응답 쌍을 찾을 수 없습니다.") from exc


@app.post("/api/inference/kv-benchmark")
def kv_benchmark(request: KVBenchmarkRequest) -> dict:
    try:
        return factory.kv_benchmark(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/scale/estimate")
def estimate_scale(request: ScaleEstimateRequest) -> dict:
    return factory.estimate_scale(request)


@app.websocket("/ws/runs/{run_id}")
async def run_events(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    try:
        last_index = int(websocket.query_params.get("after", "-1"))
    except ValueError:
        last_index = -1
    last_heartbeat = 0.0
    try:
        while True:
            try:
                events = manager.events_after(run_id, last_index)
            except KeyError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "reason": "not_found",
                        "message": "실행을 찾을 수 없습니다.",
                    }
                )
                await websocket.close(code=4404)
                return
            for event in events:
                await websocket.send_json(event)
                last_index = event["index"]
            now = time.time()
            if now - last_heartbeat >= 1.0:
                state = manager.get_run(run_id)
                await websocket.send_json(
                    {
                        "type": "heartbeat",
                        "run_id": run_id,
                        "timestamp": now,
                        "summary": state.summary(),
                        "telemetry": {"device": "cpu"},
                    }
                )
                last_heartbeat = now
            state = manager.get_run(run_id)
            if state.status in {"completed", "stopped", "failed"} and not events:
                await asyncio.sleep(0.2)
            else:
                await asyncio.sleep(0.08)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/factory/{run_id}")
async def factory_events(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    try:
        last_index = int(websocket.query_params.get("after", "-1"))
    except ValueError:
        last_index = -1
    last_heartbeat = 0.0
    try:
        while True:
            try:
                events = factory.events_after(run_id, last_index)
            except KeyError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "reason": "not_found",
                        "message": "모델 공장 실행을 찾을 수 없습니다.",
                    }
                )
                await websocket.close(code=4404)
                return
            for event in events:
                await websocket.send_json(event)
                last_index = event["index"]
            now = time.time()
            if now - last_heartbeat >= 1.0:
                state = factory.get_run(run_id)
                await websocket.send_json(
                    {
                        "type": "heartbeat",
                        "run_id": run_id,
                        "timestamp": now,
                        "summary": state.summary(),
                        "telemetry": gpu_telemetry(),
                    }
                )
                last_heartbeat = now
            await asyncio.sleep(0.08)
    except WebSocketDisconnect:
        return


DIST_DIR = ROOT / "frontend" / "dist"
if DIST_DIR.exists():
    assets = DIST_DIR / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        requested = DIST_DIR / path
        if path and requested.is_file():
            return FileResponse(requested)
        return FileResponse(DIST_DIR / "index.html")
