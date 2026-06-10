import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .schemas import ModelConfig


@dataclass
class AttentionTrace:
    layer: int
    head: int
    token_index: int
    attention: list[list[float]]
    q: list[float]
    keys: list[list[float]]
    values: list[list[float]]
    raw_scores: list[float]
    scaled_scores: list[float | None]
    weights: list[float]
    weighted_value: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, config.d_model * 3)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        capture: bool = False,
        head_index: int = 0,
        token_index: int = -1,
    ) -> tuple[torch.Tensor, dict | None]:
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        raw_scores = q @ k.transpose(-2, -1)
        scaled_scores = raw_scores / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len]
        masked_scores = scaled_scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(masked_scores, dim=-1)
        attended = weights @ v
        merged = attended.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        output = self.proj(self.dropout(merged))

        trace = None
        if capture:
            h = min(max(head_index, 0), self.n_heads - 1)
            t = seq_len - 1 if token_index < 0 else min(token_index, seq_len - 1)
            row_mask = mask[t].detach().cpu()
            scaled_row = scaled_scores[0, h, t].detach().cpu()
            trace = {
                "head": h,
                "token_index": t,
                "attention": weights[0, h].detach().cpu().tolist(),
                "q": q[0, h, t].detach().cpu().tolist(),
                "keys": k[0, h].detach().cpu().tolist(),
                "values": v[0, h].detach().cpu().tolist(),
                "raw_scores": raw_scores[0, h, t].detach().cpu().tolist(),
                "scaled_scores": [
                    float(value) if bool(allowed) else None
                    for value, allowed in zip(scaled_row, row_mask, strict=True)
                ],
                "weights": weights[0, h, t].detach().cpu().tolist(),
                "weighted_value": attended[0, h, t].detach().cpu().tolist(),
            }
        return output, trace


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        capture: bool = False,
        head_index: int = 0,
        token_index: int = -1,
    ) -> tuple[torch.Tensor, dict | None]:
        attention_output, trace = self.attention(
            self.ln1(x), capture, head_index, token_index
        )
        x = x + attention_output
        x = x + self.ffn(self.ln2(x))
        return x, trace


class VisualGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        capture_layer: int | None = None,
        capture_head: int = 0,
        capture_token: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        _, seq_len = input_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError("입력 길이가 context_length보다 큽니다.")
        positions = torch.arange(seq_len, device=input_ids.device)
        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)
        x = token_vectors + position_vectors
        layer_stats = []
        captured = None
        for layer_index, block in enumerate(self.blocks):
            x, trace = block(
                x,
                capture=capture_layer == layer_index,
                head_index=capture_head,
                token_index=capture_token,
            )
            layer_stats.append(
                {
                    "layer": layer_index,
                    "mean": float(x.detach().mean()),
                    "std": float(x.detach().std()),
                    "norm": float(x.detach().norm(dim=-1).mean()),
                }
            )
            if trace is not None:
                captured = AttentionTrace(layer=layer_index, **trace).to_dict()
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        trace_payload = None
        if capture_layer is not None:
            trace_payload = {
                "attention": captured,
                "embedding": {
                    "token_vector": token_vectors[0, capture_token].detach().cpu().tolist(),
                    "position_vector": position_vectors[capture_token].detach().cpu().tolist(),
                    "sum_vector": (
                        token_vectors[0, capture_token] + position_vectors[capture_token]
                    )
                    .detach()
                    .cpu()
                    .tolist(),
                },
                "layer_stats": layer_stats,
            }
        return logits, loss, trace_payload

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def causal_mask(self, seq_len: int) -> torch.Tensor:
        return self.blocks[0].attention.causal_mask[:seq_len, :seq_len]

