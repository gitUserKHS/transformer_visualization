import torch

from backend.app.production_model import (
    ProductionMiniLM,
    ProductionModelConfig,
    RMSNorm,
    apply_lora,
)


def small_config() -> ProductionModelConfig:
    return ProductionModelConfig(
        vocab_size=256,
        context_length=16,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
    )


def test_rmsnorm_has_unit_root_mean_square():
    norm = RMSNorm(8)
    output = norm(torch.randn(3, 4, 8))
    rms = output.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_gqa_shapes_and_kv_cache_match_full_forward():
    torch.manual_seed(3)
    model = ProductionMiniLM(small_config()).eval()
    ids = torch.randint(0, 256, (1, 7))
    full = model(ids, capture_layer=0)
    prefill = model(ids[:, :-1], use_cache=True, capture_layer=0)
    decode = model(
        ids[:, -1:],
        past_key_values=prefill["past_key_values"],
        use_cache=True,
        capture_layer=0,
    )
    assert full["trace"]["q_shape"] == [1, 4, 7, 16]
    assert full["trace"]["kv_shape"] == [1, 2, 7, 16]
    assert decode["trace"]["cache_length"] == 7
    assert torch.allclose(
        full["logits"][:, -1], decode["logits"][:, -1], atol=2e-6
    )


def test_lora_freezes_base_and_keeps_small_trainable_fraction():
    model = ProductionMiniLM(small_config())
    report = apply_lora(model, rank=4)
    assert report["trainable_parameters"] > 0
    assert report["trainable_percent"] < 10
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )


def test_lora_adapters_follow_model_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ProductionMiniLM(small_config()).to(device)
    apply_lora(model, rank=4)
    assert all(parameter.device.type == device for parameter in model.parameters())


def test_causal_loss_is_finite_with_prompt_masking():
    model = ProductionMiniLM(small_config())
    ids = torch.randint(0, 256, (2, 12))
    labels = ids.clone()
    labels[:, :5] = -100
    loss = model(ids, labels=labels)["loss"]
    assert loss is not None and torch.isfinite(loss)
