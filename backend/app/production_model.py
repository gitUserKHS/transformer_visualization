import math
from dataclasses import dataclass

import torch
from pydantic import BaseModel, Field, model_validator
from torch import nn
from torch.nn import functional as F


class ProductionModelConfig(BaseModel):
    vocab_size: int = Field(default=8192, ge=256, le=16384)
    context_length: int = Field(default=128, ge=16, le=512)
    d_model: int = Field(default=384, ge=64, le=768)
    n_layers: int = Field(default=8, ge=1, le=12)
    n_heads: int = Field(default=8, ge=1, le=16)
    n_kv_heads: int = Field(default=2, ge=1, le=16)
    d_ff: int = Field(default=1024, ge=128, le=3072)
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-5
    dropout: float = Field(default=0.0, ge=0.0, le=0.2)

    @model_validator(mode="after")
    def validate_attention(self) -> "ProductionModelConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model은 n_heads로 나누어 떨어져야 합니다.")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads는 n_kv_heads의 배수여야 합니다.")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE를 위해 head_dim은 짝수여야 합니다.")
        return self


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_length: int, theta: float):
        super().__init__()
        inverse = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos", embedding.cos(), persistent=False)
        self.register_buffer("sin", embedding.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[positions].unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin = self.sin[positions].unsqueeze(0).unsqueeze(0).to(q.dtype)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


@dataclass
class LayerKVCache:
    key: torch.Tensor
    value: torch.Tensor


KVCache = list[LayerKVCache | None]


class GQAAttention(nn.Module):
    def __init__(self, config: ProductionModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.groups = config.n_heads // config.n_kv_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(
            self.head_dim, config.context_length, config.rope_theta
        )
        self.dropout = config.dropout

    @staticmethod
    def _heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, heads, head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past: LayerKVCache | None = None,
        use_cache: bool = False,
        capture: bool = False,
    ) -> tuple[torch.Tensor, LayerKVCache | None, dict | None]:
        batch, query_length, _ = x.shape
        q = self._heads(self.q_proj(x), self.n_heads, self.head_dim)
        k = self._heads(self.k_proj(x), self.n_kv_heads, self.head_dim)
        v = self._heads(self.v_proj(x), self.n_kv_heads, self.head_dim)
        q, k = self.rope(q, k, positions)
        if past is not None:
            k = torch.cat((past.key, k), dim=-2)
            v = torch.cat((past.value, v), dim=-2)
        present = LayerKVCache(k.detach(), v.detach()) if use_cache else None
        repeated_k = k.repeat_interleave(self.groups, dim=1)
        repeated_v = v.repeat_interleave(self.groups, dim=1)
        key_length = repeated_k.size(-2)
        scores = q @ repeated_k.transpose(-2, -1) / math.sqrt(self.head_dim)
        past_length = key_length - query_length
        query_positions = torch.arange(
            past_length, past_length + query_length, device=x.device
        )
        key_positions = torch.arange(key_length, device=x.device)
        mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        weights = F.dropout(weights, self.dropout, self.training)
        attended = weights @ repeated_v
        output = attended.transpose(1, 2).contiguous().view(batch, query_length, -1)
        trace = None
        if capture:
            trace = {
                "q_shape": list(q.shape),
                "kv_shape": list(k.shape),
                "expanded_kv_shape": list(repeated_k.shape),
                "cache_length": key_length,
                "new_tokens": query_length,
                "attention": weights[0, 0].detach().float().cpu().tolist(),
            }
        return self.o_proj(output), present, trace


class SwiGLU(nn.Module):
    def __init__(self, config: ProductionModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ProductionBlock(nn.Module):
    def __init__(self, config: ProductionModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_eps)
        self.attention = GQAAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past: LayerKVCache | None,
        use_cache: bool,
        capture: bool,
    ) -> tuple[torch.Tensor, LayerKVCache | None, dict | None]:
        attention, present, trace = self.attention(
            self.attention_norm(x), positions, past, use_cache, capture
        )
        x = x + attention
        return x + self.ffn(self.ffn_norm(x)), present, trace


class ProductionMiniLM(nn.Module):
    def __init__(self, config: ProductionModelConfig | None = None):
        super().__init__()
        self.config = config or ProductionModelConfig()
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.blocks = nn.ModuleList(
            [ProductionBlock(self.config) for _ in range(self.config.n_layers)]
        )
        self.norm = RMSNorm(self.config.d_model, self.config.rms_eps)
        self.lm_head = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )
        self.lm_head.weight = self.embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _tensor_stats(name: str, tensor: torch.Tensor) -> dict:
        detached = tensor.detach().flatten()
        stride = max(1, detached.numel() // 65_536)
        sample = detached[::stride][:65_536].float()
        return {
            "name": name,
            "shape": list(tensor.shape),
            "sampled_values": int(sample.numel()),
            "mean": float(sample.mean()),
            "std": float(sample.std(unbiased=False)),
            "rms": float(sample.square().mean().sqrt()),
            "max_abs": float(sample.abs().max()),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
        capture_layer: int | None = None,
        capture_flow: bool = False,
    ) -> dict:
        _, length = input_ids.shape
        past_length = (
            0
            if not past_key_values or past_key_values[0] is None
            else past_key_values[0].key.size(-2)
        )
        if past_length + length > self.config.context_length:
            raise ValueError("입력과 KV cache 길이가 context_length를 넘습니다.")
        positions = torch.arange(
            past_length, past_length + length, device=input_ids.device
        )
        x = self.embedding(input_ids)
        flow = [self._tensor_stats("Token embedding", x)] if capture_flow else []
        presents: KVCache = []
        captured = None
        for index, block in enumerate(self.blocks):
            past = past_key_values[index] if past_key_values else None
            x, present, trace = block(
                x, positions, past, use_cache, capture_layer == index
            )
            presents.append(present)
            if capture_flow:
                flow.append(self._tensor_stats(f"Transformer block {index + 1}", x))
            if trace is not None:
                captured = {"layer": index, **trace}
        hidden = self.norm(x)
        if capture_flow:
            flow.append(self._tensor_stats("Final RMSNorm", hidden))
        logits = self.lm_head(hidden)
        if capture_flow:
            flow.append(self._tensor_stats("Vocabulary logits", logits))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().reshape(-1, logits.size(-1)),
                labels[:, 1:].contiguous().reshape(-1),
                ignore_index=-100,
            )
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden,
            "past_key_values": presents if use_cache else None,
            "trace": captured,
            "flow": flow,
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def cache_bytes(
        self, tokens: int, batch_size: int = 1, bytes_per_value: int = 2
    ) -> int:
        return (
            2
            * self.config.n_layers
            * batch_size
            * self.config.n_kv_heads
            * tokens
            * (self.config.d_model // self.config.n_heads)
            * bytes_per_value
        )


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        factory_kwargs = {
            "device": base.weight.device,
            "dtype": base.weight.dtype,
        }
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, **factory_kwargs)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, **factory_kwargs)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling


def apply_lora(model: ProductionMiniLM, rank: int = 8) -> dict:
    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for parameter in model.parameters():
        parameter.requires_grad = False
    replaced = []
    for module_name, module in list(model.named_modules()):
        for target in targets:
            child = getattr(module, target, None)
            if isinstance(child, nn.Linear):
                setattr(module, target, LoRALinear(child, rank=rank))
                replaced.append(f"{module_name}.{target}".strip("."))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "rank": rank,
        "modules": replaced,
        "trainable_parameters": trainable,
        "total_parameters": model.parameter_count,
        "trainable_percent": trainable / model.parameter_count * 100,
    }
