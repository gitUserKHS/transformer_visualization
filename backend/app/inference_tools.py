import math
import time

import torch
from torch.nn import functional as F

from .production_model import ProductionMiniLM
from .production_tokenizer import ProductionTokenizer


@torch.no_grad()
def greedy_generate(
    model: ProductionMiniLM,
    tokenizer: ProductionTokenizer,
    prompt: str,
    max_new_tokens: int,
    use_cache: bool,
) -> dict:
    model.eval()
    device = next(model.parameters()).device
    prompt_budget = max(1, model.config.context_length - max_new_tokens)
    ids = tokenizer.encode(prompt, add_eos=False)[-prompt_budget:]
    generated = list(ids)
    cache = None
    steps = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for index in range(max_new_tokens):
        if use_cache and cache is not None:
            current = torch.tensor([[generated[-1]]], device=device)
        else:
            current = torch.tensor(
                [generated[-model.config.context_length :]], device=device
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        before = time.perf_counter()
        output = model(
            current,
            past_key_values=cache,
            use_cache=use_cache,
            capture_layer=0,
        )
        cache = output["past_key_values"] if use_cache else None
        next_id = int(output["logits"][0, -1].argmax())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        generated.append(next_id)
        trace = output["trace"] or {}
        steps.append(
            {
                "index": index,
                "token_id": next_id,
                "token": tokenizer.token_strings([next_id])[0],
                "input_tokens_computed": int(current.size(1)),
                "cache_length": int(trace.get("cache_length", current.size(1))),
                "latency_ms": (time.perf_counter() - before) * 1000,
                "kv_shape": trace.get("kv_shape"),
            }
        )
        if next_id == tokenizer.eos_id:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "text": tokenizer.decode(generated),
        "token_ids": generated,
        "new_token_ids": generated[len(ids) :],
        "steps": steps,
        "elapsed_ms": elapsed * 1000,
        "tokens_per_second": len(steps) / max(elapsed, 1e-9),
    }


def cache_comparison(model: ProductionMiniLM, tokens: int, bytes_per_value: int = 2) -> list[dict]:
    head_dim = model.config.d_model // model.config.n_heads
    return [
        {
            "mode": label,
            "kv_heads": heads,
            "bytes": 2
            * model.config.n_layers
            * heads
            * head_dim
            * tokens
            * bytes_per_value,
        }
        for label, heads in [
            ("MHA", model.config.n_heads),
            ("GQA", model.config.n_kv_heads),
            ("MQA", 1),
        ]
    ]


def quantize_tensor_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
    source = tensor.detach().float()
    if source.ndim < 2:
        maximum = source.abs().max().clamp_min(1e-8)
        scale = maximum / 127
    else:
        scale = source.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
    quantized = torch.round(source / scale).clamp(-127, 127)
    restored = quantized * scale
    return restored, {
        "mse": float(F.mse_loss(restored, source)),
        "max_error": float((restored - source).abs().max()),
        "compression_ratio": 2.0,
    }


def quantize_tensor_int4(
    tensor: torch.Tensor, group_size: int = 64
) -> tuple[torch.Tensor, dict]:
    source = tensor.detach().float().flatten()
    padding = (-source.numel()) % group_size
    padded = F.pad(source, (0, padding))
    groups = padded.view(-1, group_size)
    scale = groups.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7
    quantized = torch.round(groups / scale).clamp(-8, 7)
    restored = (quantized * scale).flatten()[: source.numel()].view(tensor.shape)
    return restored, {
        "mse": float(F.mse_loss(restored, tensor.detach().float())),
        "max_error": float((restored - tensor.detach().float()).abs().max()),
        "compression_ratio": 4.0,
        "group_size": group_size,
    }


def quantization_report(model: ProductionMiniLM) -> dict:
    sample = next(
        parameter
        for name, parameter in model.named_parameters()
        if parameter.ndim == 2 and "embedding" not in name
    )
    _, int8 = quantize_tensor_int8(sample)
    _, int4 = quantize_tensor_int4(sample)
    parameters = model.parameter_count
    return {
        "bf16": {
            "bytes": parameters * 2,
            "mse": 0.0,
            "note": "기준 가중치",
        },
        "int8": {
            "bytes": parameters,
            **int8,
            "note": "실제 per-channel 복원 오차",
        },
        "int4": {
            "bytes": math.ceil(parameters / 2),
            **int4,
            "note": "실제 groupwise 복원 오차, fused kernel 없음",
        },
    }


def continuous_batch_schedule(lengths: list[int], decode_steps: int = 8) -> list[dict]:
    requests = [
        {"id": f"req-{index + 1}", "prompt_tokens": length, "remaining": decode_steps}
        for index, length in enumerate(lengths)
    ]
    timeline = []
    clock = 0
    pending = list(requests)
    active = []
    while pending or active:
        admitted = []
        while pending and len(active) < 3:
            request = pending.pop(0)
            active.append(request)
            admitted.append(request["id"])
        timeline.append(
            {
                "tick": clock,
                "prefill": admitted,
                "decode": [request["id"] for request in active],
                "batch_size": len(active),
            }
        )
        for request in active:
            request["remaining"] -= 1
        active = [request for request in active if request["remaining"] > 0]
        clock += 1
    return timeline


def scale_estimate(
    parameters_billions: float,
    tokens_billions: float,
    context_length: int,
    precision_bytes: int,
    gpu_memory_gb: float,
    gpu_tflops: float,
    utilization: float,
) -> dict:
    parameters = parameters_billions * 1e9
    training_bytes = parameters * (precision_bytes + 12)
    inference_bytes = parameters * precision_bytes
    minimum_training_gpus = math.ceil(training_bytes / (gpu_memory_gb * 1e9 * 0.85))
    minimum_inference_gpus = math.ceil(inference_bytes / (gpu_memory_gb * 1e9 * 0.85))
    training_flops = 6 * parameters * tokens_billions * 1e9
    seconds = training_flops / (gpu_tflops * 1e12 * utilization)
    return {
        "model_weight_gb": inference_bytes / 1e9,
        "training_state_gb": training_bytes / 1e9,
        "minimum_training_gpus": minimum_training_gpus,
        "minimum_inference_gpus": minimum_inference_gpus,
        "estimated_gpu_days": seconds / 86400,
        "training_flops": training_flops,
        "context_length": context_length,
        "parallelism": {
            "data_parallel": "배치 복제",
            "tensor_parallel": "레이어 내부 행렬 분할",
            "pipeline_parallel": "레이어 묶음 분할",
            "zero": "optimizer·gradient·parameter shard",
        },
    }
