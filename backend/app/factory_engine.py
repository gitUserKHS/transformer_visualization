import copy
import json
import math
import random
import shutil
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from .alignment import (
    RewardModel,
    clone_model,
    dpo_loss,
    grpo_advantages,
    masked_response_labels,
    ppo_clipped_loss,
    reward_pair_loss,
    sequence_log_probs,
)
from .dataset import ROOT
from .factory_data import FactoryData, PreferenceExample, SFTExample
from .inference_tools import (
    cache_comparison,
    continuous_batch_schedule,
    greedy_generate,
    quantization_report,
    scale_estimate,
)
from .production_model import ProductionMiniLM, ProductionModelConfig, apply_lora
from .schemas import FactoryRunConfig, KVBenchmarkRequest, ScaleEstimateRequest
from .system_info import system_diagnostics


FACTORY_DIR = ROOT / "backend" / "artifacts" / "factory"
FACTORY_HISTORY = FACTORY_DIR / "history.json"
FACTORY_STAGES = [
    "data",
    "pretrain",
    "sft",
    "reward",
    "dpo",
    "ppo",
    "grpo",
    "evaluate",
    "quantize",
]


@dataclass
class FactoryRunState:
    id: str
    config: FactoryRunConfig
    status: str = "running"
    stage: str = "queued"
    stage_step: int = 0
    stage_total: int = 0
    events: list[dict] = field(default_factory=list)
    event_counter: int = 0
    metrics: dict = field(default_factory=dict)
    checkpoints: dict = field(default_factory=dict)
    comparisons: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    peak_vram_bytes: int = 0
    updated_at: float = field(default_factory=time.time)
    paused_at: float | None = None
    paused_seconds: float = 0.0
    stage_totals: dict[str, int] = field(default_factory=dict)
    completed_stages: set[str] = field(default_factory=set)
    metric_series: list[dict] = field(default_factory=list)
    error_message: str | None = None
    paused: bool = False
    stop_requested: bool = False
    one_step: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    worker: threading.Thread | None = None

    def elapsed_seconds(self) -> float:
        end = (
            self.updated_at
            if self.status in {"completed", "failed", "stopped"}
            else self.paused_at
            if self.paused_at is not None
            else time.time()
        )
        return max(0.0, end - self.created_at - self.paused_seconds)

    def summary(self) -> dict:
        selected = set(self.config.stages)
        stage_states = {
            stage: (
                "skipped"
                if stage not in selected
                else "completed"
                if stage in self.completed_stages
                else "running"
                if stage == self.stage
                and self.status in {"running", "paused", "stopping"}
                else "pending"
            )
            for stage in FACTORY_STAGES
        }
        selected_total = sum(self.stage_totals.get(stage, 1) for stage in selected)
        completed_steps = sum(
            self.stage_totals.get(stage, 1) for stage in self.completed_stages
        )
        if self.stage in selected and self.stage not in self.completed_stages:
            completed_steps += min(
                self.stage_step, self.stage_totals.get(self.stage, self.stage_total or 1)
            )
        overall_progress = completed_steps / max(1, selected_total)
        elapsed = self.elapsed_seconds()
        speed = completed_steps / elapsed if elapsed > 0 and completed_steps else 0.0
        eta = (selected_total - completed_steps) / speed if speed > 0 else None
        return {
            "id": self.id,
            "name": self.config.name,
            "status": self.status,
            "stage": self.stage,
            "stage_step": self.stage_step,
            "stage_total": self.stage_total,
            "config": self.config.model_dump(),
            "metrics": self.metrics,
            "checkpoints": self.checkpoints,
            "comparisons": self.comparisons,
            "peak_vram_bytes": self.peak_vram_bytes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stage_states": stage_states,
            "stage_progress": self.stage_step / max(1, self.stage_total),
            "overall_progress": min(1.0, overall_progress),
            "elapsed_seconds": elapsed,
            "eta_seconds": max(0.0, eta) if eta is not None else None,
            "steps_per_second": speed,
            "metric_series": self.metric_series[-500:],
            "error_message": self.error_message,
        }


class FactoryManager:
    def __init__(self) -> None:
        FACTORY_DIR.mkdir(parents=True, exist_ok=True)
        self.data = FactoryData()
        self.runs: dict[str, FactoryRunState] = {}
        self.history = self._load_history()
        self.models: dict[str, ProductionMiniLM] = {}
        self.reward_model: RewardModel | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _load_history() -> list[dict]:
        if not FACTORY_HISTORY.exists():
            return []
        try:
            return json.loads(FACTORY_HISTORY.read_text(encoding="utf-8"))[:3]
        except (OSError, json.JSONDecodeError):
            return []

    def _save_history(self) -> None:
        FACTORY_HISTORY.write_text(
            json.dumps(self.history[:3], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _prune_artifacts(self) -> None:
        root = FACTORY_DIR.resolve()
        keep = {item["id"] for item in self.history[:3]}
        keep.update(
            run_id
            for run_id, run in self.runs.items()
            if run.status not in {"completed", "failed", "stopped"}
        )
        for child in FACTORY_DIR.iterdir():
            resolved = child.resolve()
            if (
                child.is_dir()
                and child.name not in keep
                and resolved.parent == root
            ):
                try:
                    shutil.rmtree(resolved)
                except OSError:
                    continue

    def _event(self, state: FactoryRunState, event_type: str, payload: dict) -> dict:
        event = {
            "index": state.event_counter,
            "type": event_type,
            "run_id": state.id,
            "timestamp": time.time(),
            **payload,
        }
        state.event_counter += 1
        return event

    def _emit(self, state: FactoryRunState, event_type: str, payload: dict) -> None:
        state.updated_at = time.time()
        with state.lock:
            state.events.append(self._event(state, event_type, payload))
            state.events = state.events[-2000:]

    def create_run(self, config: FactoryRunConfig) -> dict:
        diagnostics = system_diagnostics()
        if not diagnostics["production_ready"]:
            raise RuntimeError(
                "상용 모델 공장은 CUDA가 필요합니다. "
                f"{diagnostics['setup_command']}"
            )
        with self._lock:
            if self.active_run() is not None:
                raise RuntimeError("모델 공장에서 이미 학습이 진행 중입니다.")
        run_id = uuid.uuid4().hex[:10]
        state = FactoryRunState(
            id=run_id,
            config=config,
            stage_totals=self._planned_stage_totals(config),
        )
        state.events.append(self._event(state, "status", {"status": "running"}))
        state.worker = threading.Thread(
            target=self._worker, args=(state,), daemon=True, name=f"factory-{run_id}"
        )
        with self._lock:
            if self.active_run() is not None:
                raise RuntimeError("모델 공장에서 이미 학습이 진행 중입니다.")
            self.runs[run_id] = state
        state.worker.start()
        return state.summary()

    def get_run(self, run_id: str) -> FactoryRunState:
        if run_id not in self.runs:
            raise KeyError(run_id)
        return self.runs[run_id]

    def events_after(self, run_id: str, index: int) -> list[dict]:
        state = self.get_run(run_id)
        with state.lock:
            return [event for event in state.events if event["index"] > index]

    def active_run(self) -> FactoryRunState | None:
        active = [
            run
            for run in self.runs.values()
            if run.status not in {"completed", "failed", "stopped"}
        ]
        return max(active, key=lambda run: run.created_at) if active else None

    def _planned_stage_totals(self, config: FactoryRunConfig) -> dict[str, int]:
        return {
            "data": len(self.data.cleaning_trace),
            "pretrain": config.pretrain_steps,
            "sft": config.sft_steps,
            "reward": config.reward_steps,
            "dpo": config.dpo_steps,
            "ppo": config.ppo_steps,
            "grpo": config.grpo_steps,
            "evaluate": 4,
            "quantize": 3,
        }

    @staticmethod
    def _record_metric(
        state: FactoryRunState, stage: str, metric: dict
    ) -> None:
        state.metrics[stage] = metric
        state.updated_at = time.time()
        loss = metric.get("loss")
        if isinstance(loss, (int, float)) and math.isfinite(loss):
            state.metric_series.append(
                {
                    "stage": stage,
                    "step": int(metric.get("step", state.stage_step)),
                    "loss": float(loss),
                    "timestamp": state.updated_at,
                }
            )
            state.metric_series = state.metric_series[-500:]

    def control(self, run_id: str, action: str) -> dict:
        state = self.get_run(run_id)
        now = time.time()
        if action == "pause" and state.status == "running":
            state.paused = True
            state.status = "paused"
            if state.paused_at is None:
                state.paused_at = now
        elif action == "resume" and state.status == "paused":
            if state.paused_at is not None:
                state.paused_seconds += now - state.paused_at
            state.paused_at = None
            state.paused = False
            state.status = "running"
        elif action == "step" and state.status in {"paused", "running"}:
            if state.paused_at is not None:
                state.paused_seconds += now - state.paused_at
            state.paused_at = None
            state.one_step = True
            state.paused = False
            state.status = "running"
        elif action == "stop" and state.status in {"running", "paused"}:
            state.stop_requested = True
            state.status = "stopping"
        self._emit(state, "status", {"status": state.status})
        return state.summary()

    def _gate(self, state: FactoryRunState) -> None:
        while state.paused and not state.stop_requested:
            time.sleep(0.05)
        if state.stop_requested:
            raise InterruptedError("사용자가 모델 공장을 정지했습니다.")

    def _step_done(self, state: FactoryRunState) -> None:
        if torch.cuda.is_available():
            state.peak_vram_bytes = max(
                state.peak_vram_bytes, torch.cuda.max_memory_allocated()
            )
            if torch.cuda.memory_allocated() > 7.2 * 1024**3:
                raise torch.cuda.OutOfMemoryError("VRAM 안전 한도 7.2GB를 넘었습니다.")
        if state.one_step:
            state.one_step = False
            state.paused = True
            state.status = "paused"
            state.paused_at = time.time()
            state.updated_at = state.paused_at

    @staticmethod
    def _autocast():
        if not torch.cuda.is_available():
            return nullcontext()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)

    def _worker(self, state: FactoryRunState) -> None:
        torch.manual_seed(state.config.seed)
        random.seed(state.config.seed)
        torch.cuda.reset_peak_memory_stats()
        try:
            for stage in state.config.stages:
                self._gate(state)
                state.stage = stage
                state.stage_step = 0
                state.stage_total = state.stage_totals.get(stage, 1)
                self._emit(
                    state,
                    "stage",
                    {"stage": stage, "stage_status": "running"},
                )
                method = getattr(self, f"_stage_{stage}")
                attempted = False
                while True:
                    try:
                        method(state)
                        break
                    except torch.cuda.OutOfMemoryError:
                        if attempted or state.config.micro_batch_size == 1:
                            raise
                        attempted = True
                        previous_micro_batch = state.config.micro_batch_size
                        state.config.micro_batch_size = max(
                            1, state.config.micro_batch_size // 2
                        )
                        if state.config.micro_batch_size < previous_micro_batch:
                            state.config.gradient_accumulation = min(
                                32, state.config.gradient_accumulation * 2
                            )
                        torch.cuda.empty_cache()
                        self._emit(
                            state,
                            "gpu",
                            {
                                "message": "OOM 감지: microbatch를 절반으로 줄여 한 번 재시도합니다.",
                                "micro_batch_size": state.config.micro_batch_size,
                                "gradient_accumulation": state.config.gradient_accumulation,
                            },
                        )
                state.completed_stages.add(stage)
                self._emit(
                    state,
                    "stage",
                    {"stage": stage, "stage_status": "completed"},
                )
            state.status = "completed"
        except InterruptedError:
            state.status = "stopped"
        except Exception as exc:
            state.status = "failed"
            state.error_message = str(exc)
            self._emit(state, "error", {"message": str(exc)})
        finally:
            if state.paused_at is not None:
                state.paused_seconds += time.time() - state.paused_at
                state.paused_at = None
            state.updated_at = time.time()
            summary = state.summary()
            self.history = [summary] + [
                item for item in self.history if item.get("id") != state.id
            ]
            self.history = self.history[:3]
            self._save_history()
            self._prune_artifacts()
            self._emit(state, "status", {"status": state.status, "summary": summary})

    def _new_model(self) -> ProductionMiniLM:
        config = ProductionModelConfig(vocab_size=self.data.tokenizer.vocab_size)
        model = ProductionMiniLM(config).to("cuda")
        self._cuda_audit(model)
        return model

    @staticmethod
    def _cuda_audit(*modules: nn.Module) -> dict:
        parameters = [
            parameter
            for module in modules
            for parameter in module.parameters()
        ]
        cuda_parameters = sum(parameter.is_cuda for parameter in parameters)
        if cuda_parameters != len(parameters):
            raise RuntimeError(
                f"상용 트랙 파라미터 CUDA 감사 실패: {cuda_parameters}/{len(parameters)}"
            )
        return {
            "parameter_tensors": len(parameters),
            "cuda_parameter_tensors": cuda_parameters,
            "all_cuda": True,
        }

    def _record_cuda_audit(
        self, state: FactoryRunState, stage: str, *modules: nn.Module
    ) -> None:
        audit = {"stage": stage, **self._cuda_audit(*modules)}
        state.metrics["gpu"] = audit
        self._emit(state, "gpu", audit)

    def _latest_checkpoint(self, name: str) -> tuple[Path, dict] | None:
        root = FACTORY_DIR.resolve()
        for run in self.history:
            checkpoint = run.get("checkpoints", {}).get(name)
            if not checkpoint:
                continue
            path = (ROOT / checkpoint["path"]).resolve()
            if path.is_file() and path.is_relative_to(root):
                return path, run
        return None

    def _load_latest_model(self, name: str, lora: bool) -> ProductionMiniLM | None:
        found = self._latest_checkpoint(name)
        if found is None:
            return None
        path, run = found
        model = self._new_model()
        if lora:
            rank = int(run.get("config", {}).get("lora_rank", 8))
            apply_lora(model, rank)
        model.load_state_dict(load_file(str(path), device="cuda"))
        self._cuda_audit(model)
        return model

    @staticmethod
    def _optimizer(model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)

    @staticmethod
    def _parameter_group(name: str) -> str:
        if name.startswith(("embedding", "lm_head")):
            return "Embedding / logits"
        if name.startswith("blocks."):
            index = int(name.split(".", 2)[1]) + 1
            return f"Transformer block {index}"
        if name.startswith("norm"):
            return "Final RMSNorm"
        return "Other trainable parameters"

    def _gradient_flow(self, model: nn.Module) -> list[dict]:
        groups: dict[str, dict] = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            group = groups.setdefault(
                self._parameter_group(name),
                {
                    "sum_sq": 0.0,
                    "sum_abs": 0.0,
                    "max_abs": 0.0,
                    "elements": 0,
                    "tensors": 0,
                },
            )
            group["sum_sq"] += float(gradient.square().sum())
            group["sum_abs"] += float(gradient.abs().sum())
            group["max_abs"] = max(group["max_abs"], float(gradient.abs().max()))
            group["elements"] += gradient.numel()
            group["tensors"] += 1
        return [
            {
                "name": name,
                "gradient_norm": math.sqrt(values["sum_sq"]),
                "mean_abs_gradient": values["sum_abs"] / max(1, values["elements"]),
                "max_abs_gradient": values["max_abs"],
                "parameter_tensors": values["tensors"],
                "elements": values["elements"],
            }
            for name, values in groups.items()
        ]

    def _capture_parameter_samples(self, model: nn.Module) -> dict[str, list[tuple]]:
        samples: dict[str, list[tuple]] = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            flattened = parameter.detach().flatten()
            stride = max(1, flattened.numel() // 256)
            before = flattened[::stride][:256].clone()
            samples.setdefault(self._parameter_group(name), []).append(
                (parameter, stride, before)
            )
        return samples

    @staticmethod
    def _parameter_updates(samples: dict[str, list[tuple]]) -> dict[str, dict]:
        updates = {}
        for name, rows in samples.items():
            deltas = []
            for parameter, stride, before in rows:
                after = parameter.detach().flatten()[::stride][: before.numel()]
                deltas.append((after - before).float())
            combined = torch.cat(deltas) if deltas else torch.zeros(1)
            updates[name] = {
                "sampled_parameters": int(combined.numel()),
                "update_rms": float(combined.square().mean().sqrt()),
                "update_max_abs": float(combined.abs().max()),
            }
        return updates

    @staticmethod
    def _merge_backward_flow(
        gradients: list[dict], updates: dict[str, dict]
    ) -> list[dict]:
        return [{**row, **updates.get(row["name"], {})} for row in gradients]

    def _loss_surface(
        self,
        model: ProductionMiniLM,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        seed: int,
    ) -> dict:
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            directions = []
            for _ in range(2):
                direction = []
                for parameter in parameters:
                    random_tensor = torch.randn_like(parameter)
                    target_norm = parameter.detach().float().norm().clamp_min(1e-4)
                    random_norm = random_tensor.float().norm().clamp_min(1e-8)
                    direction.append(random_tensor * (target_norm / random_norm))
                directions.append(direction)

        axis = [-1.0, -0.5, 0.0, 0.5, 1.0]
        scale = 0.015
        losses = []
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for vertical in axis:
                    row = []
                    for horizontal in axis:
                        for parameter, first, second in zip(
                            parameters, directions[0], directions[1], strict=True
                        ):
                            parameter.add_(first, alpha=horizontal * scale)
                            parameter.add_(second, alpha=vertical * scale)
                        try:
                            with self._autocast():
                                loss = model(input_ids, labels=labels)["loss"]
                                assert loss is not None
                            row.append(float(loss))
                        finally:
                            for parameter, first, second in zip(
                                parameters, directions[0], directions[1], strict=True
                            ):
                                parameter.add_(first, alpha=-horizontal * scale)
                                parameter.add_(second, alpha=-vertical * scale)
                    losses.append(row)
        finally:
            model.train(was_training)
        flat = [
            (loss, row_index, column_index)
            for row_index, row in enumerate(losses)
            for column_index, loss in enumerate(row)
        ]
        minimum, row_index, column_index = min(flat)
        return {
            "axis": axis,
            "losses": losses,
            "center_loss": losses[len(axis) // 2][len(axis) // 2],
            "minimum_loss": minimum,
            "minimum_at": [axis[column_index], axis[row_index]],
            "direction_scale": scale,
            "method": "현재 trainable weight 주변의 filter-normalized 2D slice",
        }

    def _record_neural_trace(
        self,
        state: FactoryRunState,
        *,
        stage: str,
        step: int,
        loss: float,
        input_ids: torch.Tensor,
        forward_flow: list[dict],
        attention_trace: dict | None,
        backward_flow: list[dict],
        gradient_norm: float,
        learning_rate: float,
        surface: dict | None,
    ) -> None:
        token_ids = input_ids[0, :16].detach().cpu().tolist()
        previous_trace = state.metrics.get("neural_trace", {})
        if surface is None and isinstance(previous_trace, dict):
            surface = previous_trace.get("surface")
        attention = None
        if attention_trace is not None:
            matrix = attention_trace.get("attention", [])
            attention = {
                "q_shape": attention_trace.get("q_shape"),
                "kv_shape": attention_trace.get("kv_shape"),
                "preview": [row[:12] for row in matrix[:12]],
            }
        trace = {
            "stage": stage,
            "step": step,
            "loss": loss,
            "batch_shape": list(input_ids.shape),
            "tokens": self.data.tokenizer.token_strings(token_ids),
            "forward": forward_flow,
            "attention": attention,
            "backward": backward_flow,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": learning_rate,
                "pre_clip_gradient_norm": gradient_norm,
                "clip_limit": 1.0,
                "clip_scale": min(1.0, 1.0 / max(gradient_norm, 1e-12)),
            },
            "surface": surface,
        }
        state.metrics["neural_trace"] = trace
        self._emit(state, "neural_trace", trace)

    def _save_model(
        self, state: FactoryRunState, name: str, model: nn.Module
    ) -> None:
        run_dir = FACTORY_DIR / state.id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{name}.safetensors"
        tensors = {
            key: value.detach().to(torch.bfloat16).cpu().contiguous().clone()
            for key, value in model.state_dict().items()
        }
        save_file(tensors, str(path))
        state.checkpoints[name] = {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
        }

    def _stage_data(self, state: FactoryRunState) -> None:
        state.stage_total = len(self.data.cleaning_trace)
        for index, item in enumerate(self.data.cleaning_trace, start=1):
            self._gate(state)
            state.stage_step = index
            self._emit(
                state,
                "data",
                {
                    **item,
                    "retention": item["count"]
                    / self.data.cleaning_trace[0]["count"],
                },
            )
            self._step_done(state)
        state.metrics["data"] = self.data.bootstrap()

    def _stage_pretrain(self, state: FactoryRunState) -> None:
        model = self._new_model()
        self._record_cuda_audit(state, "pretrain", model)
        optimizer = self._optimizer(model, state.config.learning_rate)
        packs = self.data.packed_tokens(model.config.context_length)
        rng = random.Random(state.config.seed)
        state.stage_total = state.config.pretrain_steps
        for step in range(1, state.stage_total + 1):
            self._gate(state)
            should_trace = (
                step == 1 or step == state.stage_total or step % 100 == 0
            )
            forward_flow = []
            attention_trace = None
            trace_batch = None
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for accumulation_index in range(state.config.gradient_accumulation):
                samples = rng.sample(packs, state.config.micro_batch_size)
                batch = torch.tensor(samples, device="cuda")
                x = batch[:, :-1]
                capture = should_trace and accumulation_index == 0
                with self._autocast():
                    output = model(
                        x,
                        labels=x,
                        capture_layer=0 if capture else None,
                        capture_flow=capture,
                    )
                    loss = output["loss"]
                    assert loss is not None
                    scaled_loss = loss / state.config.gradient_accumulation
                if capture:
                    forward_flow = output["flow"]
                    attention_trace = output["trace"]
                    trace_batch = x
                scaled_loss.backward()
                accumulated += float(loss.detach())
            gradients = self._gradient_flow(model) if should_trace else []
            parameter_samples = (
                self._capture_parameter_samples(model) if should_trace else {}
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            backward_flow = (
                self._merge_backward_flow(
                    gradients, self._parameter_updates(parameter_samples)
                )
                if should_trace
                else []
            )
            state.stage_step = step
            mean_loss = accumulated / state.config.gradient_accumulation
            metric = {
                "step": step,
                "loss": mean_loss,
                "perplexity": math.exp(min(mean_loss, 20)),
                "gradient_norm": float(grad_norm),
            }
            self._record_metric(state, "pretrain", metric)
            if should_trace and trace_batch is not None:
                surface = (
                    self._loss_surface(
                        model,
                        trace_batch,
                        trace_batch,
                        state.config.seed + step,
                    )
                    if step == 1
                    else None
                )
                self._record_neural_trace(
                    state,
                    stage="pretrain",
                    step=step,
                    loss=mean_loss,
                    input_ids=trace_batch,
                    forward_flow=forward_flow,
                    attention_trace=attention_trace,
                    backward_flow=backward_flow,
                    gradient_norm=float(grad_norm),
                    learning_rate=state.config.learning_rate,
                    surface=surface,
                )
            if step == 1 or step % 10 == 0:
                self._emit(state, "pretrain", metric)
            self._step_done(state)
        self.models["base"] = model
        self._save_model(state, "base", model)

    def _ensure_base(self) -> ProductionMiniLM:
        if "base" not in self.models:
            self.models["base"] = self._load_latest_model("base", lora=False) or self._new_model()
        return self.models["base"]

    def _encode_sft(self, example: SFTExample, max_length: int) -> tuple[list[int], list[int]]:
        prompt = (
            [self.data.tokenizer.user_id]
            + self.data.tokenizer.encode(example.prompt, add_eos=False)
            + [self.data.tokenizer.assistant_id]
        )
        response = self.data.tokenizer.encode(
            example.response, add_bos=False, add_eos=True
        )
        return masked_response_labels(prompt, response, max_length)

    def _pad(
        self, rows: list[tuple[list[int], list[int]]], max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids, labels = [], []
        for row_ids, row_labels in rows:
            padding = max_length - len(row_ids)
            ids.append(row_ids + [self.data.tokenizer.pad_id] * padding)
            labels.append(row_labels + [-100] * padding)
        return torch.tensor(ids, device="cuda"), torch.tensor(labels, device="cuda")

    def _stage_sft(self, state: FactoryRunState) -> None:
        model = clone_model(self._ensure_base()).to("cuda")
        lora = apply_lora(model, state.config.lora_rank)
        self._record_cuda_audit(state, "sft", model)
        optimizer = self._optimizer(model, state.config.learning_rate)
        rng = random.Random(state.config.seed + 1)
        state.stage_total = state.config.sft_steps
        for step in range(1, state.stage_total + 1):
            self._gate(state)
            should_trace = (
                step == 1 or step == state.stage_total or step % 50 == 0
            )
            examples = rng.sample(self.data.sft_examples, state.config.micro_batch_size)
            rows = [self._encode_sft(item, model.config.context_length) for item in examples]
            x, labels = self._pad(rows, model.config.context_length)
            optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                output = model(
                    x,
                    labels=labels,
                    capture_layer=0 if should_trace else None,
                    capture_flow=should_trace,
                )
                loss = output["loss"]
                assert loss is not None
            loss.backward()
            gradients = self._gradient_flow(model) if should_trace else []
            parameter_samples = (
                self._capture_parameter_samples(model) if should_trace else {}
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            backward_flow = (
                self._merge_backward_flow(
                    gradients, self._parameter_updates(parameter_samples)
                )
                if should_trace
                else []
            )
            state.stage_step = step
            metric = {
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": float(grad_norm),
                "masked_prompt_tokens": int(labels.eq(-100).sum()),
                "response_tokens": int(labels.ne(-100).sum()),
                "lora": lora,
            }
            self._record_metric(state, "sft", metric)
            if should_trace:
                surface = (
                    self._loss_surface(
                        model,
                        x,
                        labels,
                        state.config.seed + 10_000 + step,
                    )
                    if step == 1
                    else None
                )
                self._record_neural_trace(
                    state,
                    stage="sft",
                    step=step,
                    loss=float(loss.detach()),
                    input_ids=x,
                    forward_flow=output["flow"],
                    attention_trace=output["trace"],
                    backward_flow=backward_flow,
                    gradient_norm=float(grad_norm),
                    learning_rate=state.config.learning_rate,
                    surface=surface,
                )
            if step == 1 or step % 5 == 0:
                self._emit(state, "sft", metric)
            self._step_done(state)
        self.models["sft"] = model
        self._save_model(state, "sft", model)

    def _ensure_sft(self) -> ProductionMiniLM:
        if "sft" not in self.models:
            model = self._load_latest_model("sft", lora=True)
            if model is None:
                model = clone_model(self._ensure_base()).to("cuda")
                apply_lora(model, 8)
            self.models["sft"] = model
        return self.models["sft"]

    def _encode_preference(
        self, item: PreferenceExample, max_length: int
    ) -> tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]:
        chosen = self._encode_sft(SFTExample(item.prompt, item.chosen), max_length)
        rejected = self._encode_sft(SFTExample(item.prompt, item.rejected), max_length)
        return chosen, rejected

    def _stage_reward(self, state: FactoryRunState) -> None:
        reward_model = RewardModel(clone_model(self._ensure_sft())).to("cuda")
        self._record_cuda_audit(state, "reward", reward_model)
        optimizer = self._optimizer(reward_model, state.config.learning_rate)
        data = self.data.preference_training_data()
        rng = random.Random(state.config.seed + 2)
        state.stage_total = state.config.reward_steps
        for step in range(1, state.stage_total + 1):
            self._gate(state)
            batch = rng.sample(data, state.config.micro_batch_size)
            chosen_rows, rejected_rows = zip(
                *(self._encode_preference(item, reward_model.backbone.config.context_length) for item in batch),
                strict=True,
            )
            chosen_ids, chosen_labels = self._pad(
                list(chosen_rows), reward_model.backbone.config.context_length
            )
            rejected_ids, rejected_labels = self._pad(
                list(rejected_rows), reward_model.backbone.config.context_length
            )
            chosen_mask = chosen_labels.ne(-100) | chosen_ids.ne(self.data.tokenizer.pad_id)
            rejected_mask = rejected_labels.ne(-100) | rejected_ids.ne(self.data.tokenizer.pad_id)
            optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                chosen_reward = reward_model(chosen_ids, chosen_mask.long())
                rejected_reward = reward_model(rejected_ids, rejected_mask.long())
                loss = reward_pair_loss(chosen_reward, rejected_reward)
            loss.backward()
            optimizer.step()
            accuracy = (chosen_reward > rejected_reward).float().mean()
            metric = {
                "step": step,
                "loss": float(loss.detach()),
                "chosen_reward": float(chosen_reward.detach().mean()),
                "rejected_reward": float(rejected_reward.detach().mean()),
                "reward_gap": float((chosen_reward - rejected_reward).detach().mean()),
                "accuracy": float(accuracy.detach()),
                "user_fraction": sum(item.source == "user" for item in batch) / len(batch),
            }
            self._record_metric(state, "reward", metric)
            state.stage_step = step
            if step == 1 or step % 5 == 0:
                self._emit(state, "reward", metric)
            self._step_done(state)
        self.reward_model = reward_model
        self._save_model(state, "reward_model", reward_model)

    def _ensure_reward(self) -> RewardModel:
        if self.reward_model is None:
            reward_model = RewardModel(clone_model(self._ensure_sft())).to("cuda")
            found = self._latest_checkpoint("reward_model")
            if found is not None:
                reward_model.load_state_dict(load_file(str(found[0]), device="cuda"))
            self._cuda_audit(reward_model)
            self.reward_model = reward_model
        return self.reward_model

    def _stage_dpo(self, state: FactoryRunState) -> None:
        policy = clone_model(self._ensure_sft()).to("cuda")
        reference = clone_model(self._ensure_sft()).to("cuda").eval()
        self._record_cuda_audit(state, "dpo", policy, reference)
        for parameter in reference.parameters():
            parameter.requires_grad = False
        optimizer = self._optimizer(policy, state.config.learning_rate)
        data = self.data.preference_training_data()
        rng = random.Random(state.config.seed + 3)
        state.stage_total = state.config.dpo_steps
        for step in range(1, state.stage_total + 1):
            self._gate(state)
            item = rng.choice(data)
            chosen_row, rejected_row = self._encode_preference(
                item, policy.config.context_length
            )
            chosen_ids, chosen_labels = self._pad(
                [chosen_row], policy.config.context_length
            )
            rejected_ids, rejected_labels = self._pad(
                [rejected_row], policy.config.context_length
            )
            optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                pc = sequence_log_probs(policy, chosen_ids, chosen_labels)
                pr = sequence_log_probs(policy, rejected_ids, rejected_labels)
                with torch.no_grad():
                    rc = sequence_log_probs(reference, chosen_ids, chosen_labels)
                    rr = sequence_log_probs(reference, rejected_ids, rejected_labels)
                loss, detail = dpo_loss(pc, pr, rc, rr)
            loss.backward()
            optimizer.step()
            metric = {"step": step, "loss": float(loss.detach()), **detail}
            self._record_metric(state, "dpo", metric)
            state.stage_step = step
            if step == 1 or step % 5 == 0:
                self._emit(state, "dpo", metric)
            self._step_done(state)
        self.models["dpo"] = policy
        self._save_model(state, "dpo_adapter", policy)

    def _sample_response(
        self,
        model: ProductionMiniLM,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.9,
    ) -> str:
        device = next(model.parameters()).device
        ids = self.data.tokenizer.encode(prompt, add_eos=False)[-64:]
        generated = list(ids)
        cache = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                current = (
                    torch.tensor([[generated[-1]]], device=device)
                    if cache is not None
                    else torch.tensor([generated], device=device)
                )
                with self._autocast():
                    output = model(current, past_key_values=cache, use_cache=True)
                cache = output["past_key_values"]
                probabilities = F.softmax(output["logits"][0, -1].float() / temperature, -1)
                next_id = int(torch.multinomial(probabilities, 1))
                generated.append(next_id)
                if next_id == self.data.tokenizer.eos_id:
                    break
        return self.data.tokenizer.decode(generated[len(ids) :])

    def _rollout_row(
        self, prompt: str, response: str, max_length: int
    ) -> tuple[list[int], list[int]]:
        return self._encode_sft(SFTExample(prompt, response), max_length)

    def _reward_response(self, prompt: str, response: str) -> torch.Tensor:
        reward_model = self._ensure_reward()
        row = self._rollout_row(prompt, response, reward_model.backbone.config.context_length)
        ids, labels = self._pad([row], reward_model.backbone.config.context_length)
        mask = ids.ne(self.data.tokenizer.pad_id).long()
        with torch.no_grad(), self._autocast():
            return reward_model(ids, mask)

    def _stage_ppo(self, state: FactoryRunState) -> None:
        policy = clone_model(self._ensure_sft()).to("cuda")
        reference = clone_model(self._ensure_sft()).to("cuda").eval()
        for parameter in reference.parameters():
            parameter.requires_grad = False
        value_head = nn.Linear(policy.config.d_model, 1).to("cuda")
        self._record_cuda_audit(state, "ppo", policy, reference, value_head)
        optimizer = self._optimizer(
            nn.ModuleList([policy, value_head]), state.config.learning_rate
        )
        rng = random.Random(state.config.seed + 4)
        state.stage_total = state.config.ppo_steps
        for step in range(1, state.stage_total + 1):
            self._gate(state)
            example = rng.choice(self.data.sft_examples)
            response = self._sample_response(
                policy, example.prompt, state.config.response_tokens
            )
            row = self._rollout_row(example.prompt, response, policy.config.context_length)
            ids, labels = self._pad([row], policy.config.context_length)
            with torch.no_grad(), self._autocast():
                old_log_prob = sequence_log_probs(policy, ids, labels)
                reference_log_prob = sequence_log_probs(reference, ids, labels)
                reward = self._reward_response(example.prompt, response)
            optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                output = policy(ids)
                new_log_prob = sequence_log_probs(policy, ids, labels)
                value = value_head(output["hidden_states"][:, -1]).squeeze(-1)
                kl = old_log_prob - reference_log_prob
                adjusted_reward = reward - 0.05 * kl
                advantage = (adjusted_reward - value.detach()).clamp(-5, 5)
                policy_loss, detail = ppo_clipped_loss(
                    new_log_prob, old_log_prob, advantage
                )
                value_loss = F.mse_loss(value, adjusted_reward)
                entropy = -new_log_prob.mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            loss.backward()
            optimizer.step()
            metric = {
                "step": step,
                "loss": float(loss.detach()),
                "reward": float(reward),
                "adjusted_reward": float(adjusted_reward),
                "value": float(value.detach()),
                "advantage": float(advantage.detach()),
                "kl": float(kl.detach()),
                "entropy": float(entropy.detach()),
                "response": response,
                **detail,
            }
            self._record_metric(state, "ppo", metric)
            state.stage_step = step
            self._emit(state, "ppo", metric)
            self._step_done(state)
        self.models["ppo"] = policy
        self._save_model(state, "ppo_adapter", policy)

    def _stage_grpo(self, state: FactoryRunState) -> None:
        policy = clone_model(self._ensure_sft()).to("cuda")
        reference = clone_model(self._ensure_sft()).to("cuda").eval()
        self._record_cuda_audit(state, "grpo", policy, reference)
        for parameter in reference.parameters():
            parameter.requires_grad = False
        optimizer = self._optimizer(policy, state.config.learning_rate)
        rng = random.Random(state.config.seed + 5)
        state.stage_total = state.config.grpo_steps
        for step in range(1, state.stage_total + 1):
            self._gate(state)
            example = rng.choice(self.data.sft_examples)
            responses = [
                self._sample_response(
                    policy,
                    example.prompt,
                    state.config.response_tokens,
                    temperature=0.7 + index * 0.15,
                )
                for index in range(state.config.group_size)
            ]
            rows = [
                self._rollout_row(example.prompt, response, policy.config.context_length)
                for response in responses
            ]
            ids, labels = self._pad(rows, policy.config.context_length)
            rewards = torch.cat(
                [self._reward_response(example.prompt, response) for response in responses]
            )
            advantages = grpo_advantages(rewards.unsqueeze(0)).squeeze(0)
            with torch.no_grad(), self._autocast():
                old_log_prob = sequence_log_probs(policy, ids, labels)
                reference_log_prob = sequence_log_probs(reference, ids, labels)
            optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                new_log_prob = sequence_log_probs(policy, ids, labels)
                policy_loss, detail = ppo_clipped_loss(
                    new_log_prob, old_log_prob, advantages
                )
                kl = (new_log_prob - reference_log_prob).mean()
                loss = policy_loss + 0.05 * kl
            loss.backward()
            optimizer.step()
            ranking = sorted(
                [
                    {
                        "response": response,
                        "reward": float(reward),
                        "advantage": float(advantage),
                    }
                    for response, reward, advantage in zip(
                        responses, rewards, advantages, strict=True
                    )
                ],
                key=lambda item: item["reward"],
                reverse=True,
            )
            metric = {
                "step": step,
                "loss": float(loss.detach()),
                "group_mean": float(rewards.mean()),
                "group_std": float(rewards.std(unbiased=False)),
                "advantage_mean": float(advantages.mean()),
                "kl": float(kl.detach()),
                "ranking": ranking,
                **detail,
            }
            self._record_metric(state, "grpo", metric)
            state.stage_step = step
            self._emit(state, "grpo", metric)
            self._step_done(state)
        self.models["grpo"] = policy
        self._save_model(state, "grpo_adapter", policy)

    @staticmethod
    def _repetition_score(text: str) -> float:
        words = text.lower().split()
        if not words:
            return 1.0
        return 1 - len(set(words)) / len(words)

    def _stage_evaluate(self, state: FactoryRunState) -> None:
        prompt = "Continue this children's story:\nOnce upon a time, a child found a map."
        results = {}
        candidates = [
            name for name in ("sft", "dpo", "ppo", "grpo") if name in self.models
        ]
        state.stage_total = len(candidates)
        for index, name in enumerate(candidates, start=1):
            self._gate(state)
            response = self._sample_response(self.models[name], prompt, 32, 0.7)
            reward = float(self._reward_response(prompt, response))
            results[name] = {
                "response": response,
                "reward": reward,
                "repetition": self._repetition_score(response),
                "safety": 1.0
                if not any(word in response.lower() for word in ("kill", "hate", "hurt"))
                else 0.0,
            }
            state.stage_step = index
            self._emit(state, "evaluation", {"model": name, **results[name]})
            self._step_done(state)
        hacking = {
            "response": "reward reward reward helpful helpful helpful",
            "proxy_reward": 0.95,
            "quality_score": 0.12,
            "warning": "단순 proxy를 과최적화하면 reward는 오르지만 품질은 내려갈 수 있습니다.",
        }
        state.comparisons = {"models": results, "reward_hacking": hacking}
        state.metrics["evaluate"] = state.comparisons

    def _stage_quantize(self, state: FactoryRunState) -> None:
        model = self.models.get("sft") or self._ensure_sft()
        state.stage_total = 3
        report = quantization_report(model)
        for index, (mode, detail) in enumerate(report.items(), start=1):
            self._gate(state)
            state.stage_step = index
            self._emit(state, "quantization", {"mode": mode, **detail})
            self._step_done(state)
        state.metrics["quantize"] = report

    def kv_benchmark(self, request: KVBenchmarkRequest) -> dict:
        if not torch.cuda.is_available():
            raise RuntimeError("KV cache 실측은 CUDA 환경에서 실행해야 합니다.")
        model = self.models.get("sft")
        if model is None:
            model = self._new_model()
            self.models["inference"] = model
        cached = greedy_generate(
            model, self.data.tokenizer, request.prompt, request.max_new_tokens, True
        )
        uncached = greedy_generate(
            model, self.data.tokenizer, request.prompt, request.max_new_tokens, False
        )
        return {
            "cached": cached,
            "uncached": uncached,
            "tokens_match": cached["new_token_ids"] == uncached["new_token_ids"],
            "speedup": uncached["elapsed_ms"] / max(cached["elapsed_ms"], 1e-9),
            "cache_modes": cache_comparison(
                model, len(cached["token_ids"]), bytes_per_value=2
            ),
            "continuous_batching": continuous_batch_schedule([12, 28, 7, 40]),
            "note": (
                "동적 KV cache 실측입니다. 작은 eager 모델과 짧은 문맥에서는 "
                "cache 연결 비용 때문에 속도 향상이 나타나지 않을 수 있습니다."
            ),
        }

    @staticmethod
    def estimate_scale(request: ScaleEstimateRequest) -> dict:
        return scale_estimate(**request.model_dump())

    def bootstrap(self) -> dict:
        config = ProductionModelConfig(vocab_size=self.data.tokenizer.vocab_size)
        model = ProductionMiniLM(config)
        lora_model = copy.deepcopy(model)
        lora = apply_lora(lora_model, 8)
        return {
            "data": self.data.bootstrap(),
            "model": {
                **config.model_dump(),
                "parameter_count": model.parameter_count,
                "kv_cache_128_bytes": model.cache_bytes(128),
                "lora": lora,
            },
            "history": self.history,
            "stages": FACTORY_STAGES,
        }
