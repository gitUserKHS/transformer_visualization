import math

import torch

from backend.app.model import VisualGPT
from backend.app.schemas import ModelConfig


def tiny_config() -> ModelConfig:
    return ModelConfig(
        context_length=8,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
    )


def test_model_shapes_mask_and_attention_probability():
    model = VisualGPT(tiny_config())
    x = torch.randint(0, 2048, (2, 8))
    logits, loss, trace = model(x, x, capture_layer=0, capture_head=1, capture_token=6)
    assert logits.shape == (2, 8, 2048)
    assert loss is not None and math.isfinite(float(loss.detach()))
    assert trace is not None
    weights = trace["attention"]["attention"]
    assert all(abs(sum(row) - 1.0) < 1e-5 for row in weights)
    assert all(weights[row][column] == 0 for row in range(8) for column in range(row + 1, 8))
    assert torch.equal(model.causal_mask(4), torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_optimizer_step_changes_parameters():
    torch.manual_seed(7)
    model = VisualGPT(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 2048, (2, 8))
    before = model.lm_head.weight.detach().clone()
    _, loss, _ = model(x, x)
    assert loss is not None
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.lm_head.weight)

