import torch

from backend.app.alignment import (
    dpo_loss,
    grpo_advantages,
    masked_response_labels,
    ppo_clipped_loss,
    reward_pair_loss,
)


def test_sft_prompt_is_masked():
    ids, labels = masked_response_labels([2, 7, 8], [9, 10, 3], 10)
    assert ids == [2, 7, 8, 9, 10, 3]
    assert labels == [-100, -100, -100, 9, 10, 3]


def test_reward_pair_loss_prefers_larger_chosen_reward():
    good = reward_pair_loss(torch.tensor([2.0]), torch.tensor([-1.0]))
    bad = reward_pair_loss(torch.tensor([-1.0]), torch.tensor([2.0]))
    assert good < bad


def test_ppo_clips_large_probability_ratio():
    loss, detail = ppo_clipped_loss(
        torch.tensor([0.5]), torch.tensor([0.0]), torch.tensor([1.0])
    )
    assert torch.isfinite(loss)
    assert detail["clipped_ratio"] <= 1.200001
    assert detail["clip_fraction"] == 1.0


def test_dpo_margin_and_grpo_centering():
    loss, detail = dpo_loss(
        torch.tensor([0.0]),
        torch.tensor([-1.0]),
        torch.tensor([-0.2]),
        torch.tensor([-0.7]),
    )
    advantages = grpo_advantages(torch.tensor([[1.0, 2.0, 4.0, 5.0]]))
    assert torch.isfinite(loss)
    assert detail["implicit_reward_margin"] > 0
    assert torch.allclose(advantages.mean(-1), torch.zeros(1), atol=1e-6)

