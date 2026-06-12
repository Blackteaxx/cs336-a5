from typing import Callable, Literal

import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str] Rollouts from the policy. The length of this list is rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str] The ground truths for the examples. The length of this list is rollout_batch_size, because the ground truth for each example is repeated group_size times.
        group_size: int Number of responses per question (group).
        advantage_eps: float Small constant to avoid division by zero in normalization.
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise subtract only the group mean.
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
            advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
            raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
        metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must match")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(rollout_responses) % group_size != 0:
        raise ValueError("rollout batch length must be divisible by group_size")

    reward_dicts = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(
            rollout_responses,
            repeated_ground_truths,
            strict=True,
        )
    ]
    raw_rewards = torch.tensor(
        [float(reward["reward"]) for reward in reward_dicts],
        dtype=torch.float32,
    )
    grouped_rewards = raw_rewards.reshape(-1, group_size)
    grouped_means = grouped_rewards.mean(dim=1, keepdim=True)
    grouped_stds = grouped_rewards.std(dim=1, keepdim=True)

    if normalize_by_std:
        normalized_rewards = (grouped_rewards - grouped_means) / (
            grouped_stds + advantage_eps
        )
    else:
        normalized_rewards = grouped_rewards - grouped_means

    advantages = normalized_rewards.flatten()
    format_rewards = torch.tensor(
        [float(reward.get("format_reward", 0.0)) for reward in reward_dicts],
        dtype=torch.float32,
    )
    answer_rewards = torch.tensor(
        [float(reward.get("answer_reward", 0.0)) for reward in reward_dicts],
        dtype=torch.float32,
    )

    metadata = {
        "reward/mean": float(raw_rewards.mean().item()),
        "reward/std": float(raw_rewards.std(unbiased=False).item()),
        "reward/min": float(raw_rewards.min().item()),
        "reward/max": float(raw_rewards.max().item()),
        "reward/format_mean": float(format_rewards.mean().item()),
        "reward/answer_mean": float(answer_rewards.mean().item()),
        "reward/group_mean_mean": float(grouped_means.mean().item()),
        "reward/group_std_mean": float(grouped_stds.mean().item()),
        "reward/advantage_mean": float(advantages.mean().item()),
        "reward/advantage_std": float(advantages.std(unbiased=False).item()),
        "reward/num_groups": float(grouped_rewards.shape[0]),
        "reward/group_size": float(group_size),
    }

    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for each token.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to be aggregated across the batch and sequence dimensions in the training loop).
        Implementation tips:
            • Broadcast the raw_rewards_or_advantages over the sequence_length dimension
    """
    # Reinforce: pg loss = \sum_N \sum_T A/R * log p_\theta
    # 并且这里的 Adv 应当是最终获取，放在最后一个 token 上，但是每一个 token 都用这个
    
    return -raw_rewards_or_advantages * policy_log_probs
    
def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float, 
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs from the old policy.
        cliprange: float Clip parameter ε (e.g. 0.2).
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss. 
        metadata dict containing whatever you want to log. We suggest logging whether each token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of the min was lower than the LHS.
    Implementation tips:
        • Broadcast advantages over sequence_length.
    """
    # min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)
    r = torch.exp(policy_log_probs - old_log_probs) # [batch_size, sequence_length]
    
    cliped_r = torch.clamp(r, 1 - cliprange, 1 + cliprange) # [batch_size, sequence_length]
    unclipped_loss = r * advantages
    clipped_loss = cliped_r * advantages
    was_clipped = clipped_loss < unclipped_loss
    loss = -torch.min(unclipped_loss, clipped_loss)

    metadata = {
        "grpo_clip/ratio": r.detach(),
        "grpo_clip/clipped_ratio": cliped_r.detach(),
        "grpo_clip/was_clipped": was_clipped.detach(),
        "grpo_clip/clip_fraction": was_clipped.float().mean().detach(),
    }

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
        loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).
        old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange Required for "grpo_clip"; scalar ε used for clipping.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss (batch_size, sequence_length), per-token loss.
        metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    Implementation tips:
        • Delegate to compute_naive_policy_gradient_loss or compute_grpo_clip_loss.
        • Perform argument checks (see assertion pattern above).
        • Aggregate any returned metadata into a single dict.
    """
    assert loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip"]
    assert policy_log_probs.ndim == 2

    metadata = {}

    if loss_type == "no_baseline":
        assert raw_rewards is not None
        assert raw_rewards.shape == policy_log_probs[:, :1].shape
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata["policy_gradient/loss_mean"] = loss.detach().mean()
        metadata["policy_gradient/raw_reward_mean"] = raw_rewards.detach().mean()
        return loss, metadata

    if loss_type == "reinforce_with_baseline":
        assert advantages is not None
        assert advantages.shape == policy_log_probs[:, :1].shape
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata["policy_gradient/loss_mean"] = loss.detach().mean()
        metadata["policy_gradient/advantage_mean"] = advantages.detach().mean()
        return loss, metadata

    assert advantages is not None
    assert old_log_probs is not None
    assert cliprange is not None
    assert advantages.shape == policy_log_probs[:, :1].shape
    assert old_log_probs.shape == policy_log_probs.shape
    assert cliprange >= 0

    loss, grpo_metadata = compute_grpo_clip_loss(
        advantages,
        policy_log_probs,
        old_log_probs,
        cliprange,
    )
    metadata.update(grpo_metadata)
    metadata["policy_gradient/loss_mean"] = loss.detach().mean()
    metadata["policy_gradient/advantage_mean"] = advantages.detach().mean()
    return loss, metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1.
    Args:
        tensor: torch.Tensor The data to be averaged.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
        dim: int | None Dimension over which to average. If None, compute the mean over all masked elements.
    Returns:
        torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    masked_tensor = tensor * mask
    denom = mask.sum(dim=dim)
    return masked_tensor.sum(dim=dim) / denom

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        loss_type One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
        old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
        cliprange Clip parameter ε for GRPO-Clip.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
        metadata Dict with metadata from the underlying loss call, and any other statistics you might want to log.
        Implementation tips:
        • You should call loss.backward() in this function. Make sure to adjust for gradient accumulation.
    """
    
    # compute 没有被归一化的 loss
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    ) # [batch_size, sequence_length]
    # Normalize each response independently before averaging the microbatch.
    loss = masked_mean(loss, response_mask, dim=-1).mean()
    loss /= gradient_accumulation_steps
    
    # 累计梯度
    loss.backward() 
    
    return loss, metadata
    
    
    
