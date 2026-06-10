import math

import torch

from backend.app.inference_tools import (
    cache_comparison,
    continuous_batch_schedule,
    quantize_tensor_int4,
    quantize_tensor_int8,
)
from backend.app.production_model import ProductionMiniLM, ProductionModelConfig


def test_quantization_reports_finite_reconstruction_error():
    tensor = torch.randn(17, 33)
    restored8, int8 = quantize_tensor_int8(tensor)
    restored4, int4 = quantize_tensor_int4(tensor, group_size=16)

    assert restored8.shape == tensor.shape
    assert restored4.shape == tensor.shape
    assert math.isfinite(int8["mse"]) and int8["mse"] < int4["mse"]
    assert math.isfinite(int4["mse"])


def test_gqa_cache_is_smaller_than_mha_and_scheduler_finishes():
    model = ProductionMiniLM(
        ProductionModelConfig(
            vocab_size=256,
            context_length=16,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
        )
    )
    comparison = {row["mode"]: row["bytes"] for row in cache_comparison(model, 16)}
    assert comparison["MHA"] == comparison["GQA"] * 2
    assert comparison["GQA"] == comparison["MQA"] * 2

    timeline = continuous_batch_schedule([4, 8, 12, 16], decode_steps=3)
    assert timeline[0]["batch_size"] == 3
    assert timeline[-1]["batch_size"] == 1
