import json
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .dataset import ROOT, StoryDataset
from .model import VisualGPT
from .schemas import GenerateRequest, InspectRequest, ModelConfig, TrainingConfig


ARTIFACT_DIR = ROOT / "backend" / "artifacts"
RUNS_DIR = ARTIFACT_DIR / "runs"
HISTORY_PATH = ARTIFACT_DIR / "history.json"
DEMO_PATH = ARTIFACT_DIR / "demo_checkpoint.pt"


PRESETS = {
    "inspect": {
        "label": "한 스텝 해부",
        "description": "한 번의 forward·backward를 느리게 살펴봅니다.",
        "steps": 1,
        "batch_size": 8,
        "learning_rate": 0.0003,
    },
    "balanced": {
        "label": "균형",
        "description": "약 100만 파라미터 모델의 학습 변화를 짧게 관찰합니다.",
        "steps": 500,
        "batch_size": 32,
        "learning_rate": 0.0003,
    },
    "deep": {
        "label": "조금 더 깊게",
        "description": "더 긴 학습으로 손실과 생성 품질 변화를 비교합니다.",
        "steps": 1200,
        "batch_size": 32,
        "learning_rate": 0.0002,
    },
}


@dataclass
class RunState:
    id: str
    config: TrainingConfig
    model: VisualGPT
    optimizer: torch.optim.Optimizer
    generator: torch.Generator
    status: str = "running"
    step: int = 0
    losses: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    event_counter: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    paused_at: float | None = None
    paused_seconds: float = 0.0
    last_metrics: dict = field(default_factory=dict)
    sample: str = ""
    error_message: str | None = None
    one_step: bool = False
    stop_requested: bool = False
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
        elapsed = self.elapsed_seconds()
        speed = self.step / elapsed if elapsed > 0 and self.step else 0.0
        remaining = max(0, self.config.steps - self.step)
        eta = remaining / speed if speed > 0 else None
        return {
            "id": self.id,
            "name": self.config.name,
            "status": self.status,
            "step": self.step,
            "total_steps": self.config.steps,
            "config": self.config.model_dump(),
            "losses": self.losses[-200:],
            "last_metrics": self.last_metrics,
            "sample": self.sample,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parameter_count": self.model.parameter_count,
            "stage": "training",
            "stage_progress": self.step / max(1, self.config.steps),
            "overall_progress": self.step / max(1, self.config.steps),
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "steps_per_second": speed,
            "metric_series": self.losses[-500:],
            "error_message": self.error_message,
        }


class TrainingManager:
    def __init__(self) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.dataset = StoryDataset()
        self.runs: dict[str, RunState] = {}
        self._runs_lock = threading.Lock()
        self.history = self._load_history()
        self.demo_model = self._load_demo_model()

    def _load_demo_model(self) -> VisualGPT:
        config = ModelConfig()
        model = VisualGPT(config)
        if DEMO_PATH.exists():
            payload = torch.load(DEMO_PATH, map_location="cpu", weights_only=True)
            config = ModelConfig.model_validate(payload.get("config", {}))
            model = VisualGPT(config)
            model.load_state_dict(payload["model"])
        model.eval()
        return model

    @staticmethod
    def _load_history() -> list[dict]:
        if not HISTORY_PATH.exists():
            return []
        try:
            return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))[:5]
        except (json.JSONDecodeError, OSError):
            return []

    def _save_history(self) -> None:
        HISTORY_PATH.write_text(
            json.dumps(self.history[:5], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create_run(self, config: TrainingConfig) -> dict:
        with self._runs_lock:
            if self.active_run() is not None:
                raise RuntimeError("학습 현미경에서 이미 학습이 진행 중입니다.")
        torch.manual_seed(config.seed)
        model = VisualGPT(config.model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        run_id = uuid.uuid4().hex[:10]
        state = RunState(
            id=run_id,
            config=config,
            model=model,
            optimizer=optimizer,
            generator=torch.Generator().manual_seed(config.seed),
        )
        state.events.append(self._event(state, "status", {"status": "running"}))
        worker = threading.Thread(
            target=self._train_worker,
            args=(state,),
            name=f"training-{run_id}",
            daemon=True,
        )
        state.worker = worker
        with self._runs_lock:
            if self.active_run() is not None:
                raise RuntimeError("학습 현미경에서 이미 학습이 진행 중입니다.")
            self.runs[run_id] = state
        worker.start()
        return state.summary()

    def _event(self, state: RunState, event_type: str, payload: dict) -> dict:
        event = {
            "index": state.event_counter,
            "type": event_type,
            "run_id": state.id,
            "timestamp": time.time(),
            **payload,
        }
        state.event_counter += 1
        return event

    def _emit(self, state: RunState, event_type: str, payload: dict) -> None:
        with state.lock:
            state.events.append(self._event(state, event_type, payload))
            if len(state.events) > 1000:
                state.events = state.events[-1000:]

    def _train_worker(self, state: RunState) -> None:
        try:
            while state.step < state.config.steps and not state.stop_requested:
                if state.status == "paused":
                    time.sleep(0.05)
                    continue
                metrics = self._train_step(state)
                state.step += 1
                state.updated_at = time.time()
                elapsed = max(state.elapsed_seconds(), 1e-6)
                metrics["steps_per_second"] = state.step / elapsed
                metrics["step"] = state.step
                state.last_metrics = metrics
                state.losses.append(
                    {
                        "step": state.step,
                        "loss": metrics["loss"],
                        "perplexity": metrics["perplexity"],
                    }
                )
                include_trace = state.step == 1 or state.step % 10 == 0
                payload = {"metrics": metrics, "summary": state.summary()}
                if include_trace:
                    payload["trace"] = self.inspect_batch(state)
                self._emit(state, "metrics", payload)
                if state.step == 1 or state.step % 100 == 0:
                    state.sample = self.generate(
                        GenerateRequest(
                            prompt="Once upon a time",
                            max_new_tokens=16,
                            seed=state.config.seed,
                        ),
                        model=state.model,
                    )["text"]
                    self._emit(state, "sample", {"text": state.sample})
                if state.one_step:
                    state.one_step = False
                    state.status = "paused"
                    state.paused_at = time.time()
                    state.updated_at = state.paused_at
                    self._emit(state, "status", {"status": "paused"})
            if state.stop_requested:
                state.status = "stopped"
            else:
                state.status = "completed"
            self._finalize_run(state)
        except Exception as exc:
            state.status = "failed"
            state.error_message = str(exc)
            state.last_metrics["error"] = state.error_message
            self._emit(state, "error", {"message": str(exc)})
            self._finalize_run(state, save_checkpoint=False)

    def _train_step(self, state: RunState) -> dict:
        state.model.train()
        x, y = self.dataset.batch(
            "train",
            state.config.batch_size,
            state.config.model.context_length,
            state.generator,
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in state.model.named_parameters()
        }
        _, loss, _ = state.model(x, y)
        assert loss is not None
        state.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            state.model.parameters(), state.config.grad_clip
        )
        state.optimizer.step()
        delta_sq = sum(
            float((parameter.detach() - before[name]).pow(2).sum())
            for name, parameter in state.model.named_parameters()
        )
        loss_value = float(loss.detach())
        return {
            "loss": loss_value,
            "perplexity": float(math.exp(min(loss_value, 20))),
            "gradient_norm": float(grad_norm),
            "parameter_delta_norm": math.sqrt(delta_sq),
            "learning_rate": state.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def inspect_batch(self, state: RunState) -> dict:
        state.model.eval()
        x, y = self.dataset.batch(
            "valid", 1, state.config.model.context_length, state.generator
        )
        logits, loss, trace = state.model(
            x,
            y,
            capture_layer=0,
            capture_head=0,
            capture_token=x.size(1) - 1,
        )
        top_values, top_ids = torch.topk(logits[0, -1], k=5)
        probabilities = F.softmax(logits[0, -1], dim=-1)
        return {
            "tokens": self.dataset.tokenizer.token_strings(x[0].tolist()),
            "token_ids": x[0].tolist(),
            "target_tokens": self.dataset.tokenizer.token_strings(y[0].tolist()),
            "loss": float(loss) if loss is not None else None,
            "top_logits": [
                {
                    "token": self.dataset.tokenizer.vocab[token_id],
                    "token_id": token_id,
                    "logit": float(value),
                    "probability": float(probabilities[token_id]),
                }
                for value, token_id in zip(top_values.tolist(), top_ids.tolist(), strict=True)
            ],
            **(trace or {}),
        }

    def control(self, run_id: str, action: str) -> dict:
        state = self.get_run(run_id)
        now = time.time()
        if action == "pause" and state.status == "running":
            state.status = "paused"
            state.paused_at = now
        elif action == "resume" and state.status == "paused":
            if state.paused_at is not None:
                state.paused_seconds += now - state.paused_at
            state.paused_at = None
            state.status = "running"
        elif action == "step" and state.status in {"paused", "running"}:
            if state.paused_at is not None:
                state.paused_seconds += now - state.paused_at
            state.paused_at = None
            state.one_step = True
            state.status = "running"
        elif action == "stop" and state.status in {"running", "paused"}:
            state.stop_requested = True
            state.status = "stopping"
        state.updated_at = now
        self._emit(state, "status", {"status": state.status})
        return state.summary()

    def active_run(self) -> RunState | None:
        active = [
            run
            for run in self.runs.values()
            if run.status not in {"completed", "failed", "stopped"}
        ]
        return max(active, key=lambda run: run.created_at) if active else None

    def get_run(self, run_id: str) -> RunState:
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise KeyError(f"실행 {run_id}을 찾을 수 없습니다.") from exc

    def events_after(self, run_id: str, index: int) -> list[dict]:
        state = self.get_run(run_id)
        with state.lock:
            return [event for event in state.events if event["index"] > index]

    def _finalize_run(self, state: RunState, save_checkpoint: bool = True) -> None:
        if state.paused_at is not None:
            state.paused_seconds += time.time() - state.paused_at
            state.paused_at = None
        state.updated_at = time.time()
        if save_checkpoint:
            torch.save(
                {
                    "model": state.model.state_dict(),
                    "config": state.config.model.model_dump(),
                },
                RUNS_DIR / f"{state.id}.pt",
            )
        summary = state.summary()
        self.history = [summary] + [
            item for item in self.history if item.get("id") != state.id
        ]
        self.history = self.history[:5]
        self._save_history()
        self._emit(state, "status", {"status": state.status, "summary": summary})

    @torch.no_grad()
    def generate(
        self, request: GenerateRequest, model: VisualGPT | None = None
    ) -> dict:
        if model is None:
            if request.run_id and request.run_id in self.runs:
                model = self.runs[request.run_id].model
            else:
                model = self.demo_model
        model.eval()
        generator = torch.Generator().manual_seed(request.seed)
        ids = self.dataset.tokenizer.encode(
            request.prompt, add_bos=True, add_eos=False
        )
        steps = []
        latest_trace: dict[str, Any] | None = None
        for _ in range(request.max_new_tokens):
            context = ids[-model.config.context_length :]
            x = torch.tensor([context], dtype=torch.long)
            token_index = len(context) - 1
            logits, _, trace = model(
                x,
                capture_layer=min(request.layer, model.config.n_layers - 1),
                capture_head=min(request.head, model.config.n_heads - 1),
                capture_token=(
                    token_index
                    if request.token_index < 0
                    else min(request.token_index, token_index)
                ),
            )
            next_logits = logits[0, -1] / request.temperature
            k = min(request.top_k, next_logits.numel())
            top_values, top_ids = torch.topk(next_logits, k)
            top_probs = F.softmax(top_values, dim=-1)
            sampled_offset = torch.multinomial(
                top_probs, num_samples=1, generator=generator
            ).item()
            next_id = int(top_ids[sampled_offset])
            ids.append(next_id)
            candidates = [
                {
                    "token": self.dataset.tokenizer.vocab[token_id],
                    "token_id": token_id,
                    "logit": float(next_logits[token_id]),
                    "probability": float(probability),
                }
                for token_id, probability in zip(
                    top_ids[:8].tolist(), top_probs[:8].tolist(), strict=True
                )
            ]
            steps.append(
                {
                    "token": self.dataset.tokenizer.vocab[next_id],
                    "token_id": next_id,
                    "candidates": candidates,
                }
            )
            latest_trace = {
                "tokens": self.dataset.tokenizer.token_strings(context),
                "token_ids": context,
                "top_logits": candidates[:5],
                **(trace or {}),
            }
            if next_id == self.dataset.tokenizer.eos_id:
                break
        return {
            "text": self.dataset.tokenizer.decode(ids),
            "tokens": self.dataset.tokenizer.token_strings(ids),
            "token_ids": ids,
            "steps": steps,
            "trace": latest_trace,
            "model": {
                "parameter_count": model.parameter_count,
                **model.config.model_dump(),
            },
        }

    @torch.no_grad()
    def inspect(self, request: InspectRequest) -> dict:
        model = (
            self.runs[request.run_id].model
            if request.run_id and request.run_id in self.runs
            else self.demo_model
        )
        model.eval()
        ids = request.token_ids[-model.config.context_length :]
        if any(token_id < 0 or token_id >= model.config.vocab_size for token_id in ids):
            raise ValueError("어휘 범위를 벗어난 token id가 있습니다.")
        token_index = len(ids) - 1 if request.token_index < 0 else min(
            request.token_index, len(ids) - 1
        )
        x = torch.tensor([ids], dtype=torch.long)
        logits, _, trace = model(
            x,
            capture_layer=min(request.layer, model.config.n_layers - 1),
            capture_head=min(request.head, model.config.n_heads - 1),
            capture_token=token_index,
        )
        top_values, top_ids = torch.topk(logits[0, token_index], k=5)
        probabilities = F.softmax(logits[0, token_index], dim=-1)
        return {
            "tokens": self.dataset.tokenizer.token_strings(ids),
            "token_ids": ids,
            "top_logits": [
                {
                    "token": self.dataset.tokenizer.vocab[token_id],
                    "token_id": token_id,
                    "logit": float(value),
                    "probability": float(probabilities[token_id]),
                }
                for value, token_id in zip(
                    top_values.tolist(), top_ids.tolist(), strict=True
                )
            ],
            **(trace or {}),
        }

    def bootstrap(self) -> dict:
        return {
            "dataset": {
                "name": "roneneldan/TinyStories",
                "train_count": min(5000, len(self.dataset.stories)),
                "validation_count": min(500, max(0, len(self.dataset.stories) - 5000)),
                "bundled": self.dataset.is_official_subset,
                "samples": self.dataset.sample_rows(),
                "license": "CDLA-Sharing-1.0",
                "source": "https://huggingface.co/datasets/roneneldan/TinyStories",
            },
            "presets": PRESETS,
            "model": {
                **self.demo_model.config.model_dump(),
                "parameter_count": self.demo_model.parameter_count,
                "checkpoint_ready": DEMO_PATH.exists(),
            },
            "history": self.history[:5],
        }
