import copy

import torch
from torch import nn
from torch.nn import functional as F

from .production_model import ProductionMiniLM


def masked_response_labels(
    prompt_ids: list[int], response_ids: list[int], max_length: int
) -> tuple[list[int], list[int]]:
    ids = (prompt_ids + response_ids)[:max_length]
    prompt_length = min(len(prompt_ids), len(ids))
    labels = [-100] * prompt_length + ids[prompt_length:]
    return ids, labels


def sequence_log_probs(
    model: ProductionMiniLM, input_ids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    output = model(input_ids)
    logits = output["logits"][:, :-1]
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    safe_targets = targets.masked_fill(~mask, 0)
    token_log_probs = F.log_softmax(logits.float(), dim=-1).gather(
        -1, safe_targets.unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * mask).sum(-1) / mask.sum(-1).clamp_min(1)


class RewardModel(nn.Module):
    def __init__(self, backbone: ProductionMiniLM):
        super().__init__()
        self.backbone = backbone
        self.score = nn.Linear(backbone.config.d_model, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids)
        hidden = output["hidden_states"]
        final_index = attention_mask.sum(-1).clamp_min(1) - 1
        pooled = hidden[
            torch.arange(hidden.size(0), device=hidden.device), final_index
        ]
        return self.score(pooled).squeeze(-1)


def reward_pair_loss(
    chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor
) -> torch.Tensor:
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    logits = beta * (policy_margin - reference_margin)
    loss = -F.logsigmoid(logits).mean()
    return loss, {
        "policy_margin": float(policy_margin.detach().mean()),
        "reference_margin": float(reference_margin.detach().mean()),
        "implicit_reward_margin": float(logits.detach().mean()),
    }


def ppo_clipped_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = ratio * advantage
    clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    clipped = clipped_ratio * advantage
    loss = -torch.minimum(unclipped, clipped).mean()
    return loss, {
        "ratio": float(ratio.detach().mean()),
        "clipped_ratio": float(clipped_ratio.detach().mean()),
        "clip_fraction": float((ratio.ne(clipped_ratio)).float().mean()),
    }


def grpo_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)


def clone_model(model: ProductionMiniLM) -> ProductionMiniLM:
    return copy.deepcopy(model)
