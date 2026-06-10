from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    vocab_size: Literal[2048] = 2048
    context_length: int = Field(default=32, ge=8, le=64)
    d_model: int = Field(default=128, ge=32, le=256)
    n_layers: int = Field(default=2, ge=1, le=4)
    n_heads: int = Field(default=4, ge=1, le=8)
    d_ff: int = Field(default=512, ge=64, le=1024)
    dropout: float = Field(default=0.0, ge=0.0, le=0.3)

    @model_validator(mode="after")
    def validate_heads(self) -> "ModelConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model은 n_heads로 나누어 떨어져야 합니다.")
        return self


class TrainingConfig(BaseModel):
    name: str = Field(default="새 실험", min_length=1, max_length=50)
    preset: Literal["inspect", "balanced", "deep"] = "balanced"
    steps: int = Field(default=500, ge=1, le=3000)
    batch_size: int = Field(default=32, ge=1, le=64)
    learning_rate: float = Field(default=3e-4, ge=1e-5, le=3e-3)
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.2)
    grad_clip: float = Field(default=1.0, ge=0.1, le=5.0)
    seed: int = Field(default=42, ge=0, le=999_999)
    model: ModelConfig = Field(default_factory=ModelConfig)


class RunControl(BaseModel):
    action: Literal["step", "pause", "resume", "stop"]


class GenerateRequest(BaseModel):
    prompt: str = Field(default="Once upon a time", min_length=1, max_length=500)
    max_new_tokens: int = Field(default=24, ge=1, le=80)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=10, ge=1, le=100)
    seed: int = Field(default=42, ge=0, le=999_999)
    run_id: str | None = None
    layer: int = Field(default=0, ge=0, le=3)
    head: int = Field(default=0, ge=0, le=7)
    token_index: int = Field(default=-1, ge=-1, le=63)


class InspectRequest(BaseModel):
    token_ids: list[int] = Field(min_length=1, max_length=64)
    run_id: str | None = None
    layer: int = Field(default=0, ge=0, le=3)
    head: int = Field(default=0, ge=0, le=7)
    token_index: int = Field(default=-1, ge=-1, le=63)


FactoryStage = Literal[
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


class FactoryRunConfig(BaseModel):
    name: str = Field(default="상용 미니 모델 공장", min_length=1, max_length=60)
    stages: list[FactoryStage] = Field(
        default_factory=lambda: [
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
    )
    pretrain_steps: int = Field(default=1200, ge=1, le=5000)
    sft_steps: int = Field(default=200, ge=1, le=1000)
    reward_steps: int = Field(default=150, ge=1, le=1000)
    dpo_steps: int = Field(default=100, ge=1, le=500)
    ppo_steps: int = Field(default=30, ge=1, le=200)
    grpo_steps: int = Field(default=30, ge=1, le=200)
    micro_batch_size: int = Field(default=16, ge=1, le=16)
    gradient_accumulation: int = Field(default=2, ge=1, le=32)
    learning_rate: float = Field(default=3e-4, ge=1e-6, le=3e-3)
    lora_rank: int = Field(default=8, ge=2, le=32)
    response_tokens: int = Field(default=32, ge=4, le=64)
    group_size: int = Field(default=4, ge=2, le=8)
    seed: int = Field(default=42, ge=0, le=999_999)


class PreferenceVote(BaseModel):
    pair_id: str
    choice: Literal["a", "b", "tie"]


class KVBenchmarkRequest(BaseModel):
    prompt: str = Field(default="Once upon a time", min_length=1, max_length=500)
    max_new_tokens: int = Field(default=24, ge=1, le=64)
    seed: int = Field(default=42, ge=0, le=999_999)


class ScaleEstimateRequest(BaseModel):
    parameters_billions: float = Field(default=7.0, ge=0.01, le=1000)
    tokens_billions: float = Field(default=140.0, ge=0.01, le=100_000)
    context_length: int = Field(default=4096, ge=128, le=1_000_000)
    precision_bytes: int = Field(default=2, ge=1, le=4)
    gpu_memory_gb: float = Field(default=80.0, ge=1, le=500)
    gpu_tflops: float = Field(default=312.0, ge=1, le=10_000)
    utilization: float = Field(default=0.4, ge=0.05, le=0.9)
