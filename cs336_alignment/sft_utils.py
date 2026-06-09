from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel


def tokenize_prompt_and_output(
    prompt_strs: Sequence[str],
    output_strs: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
    other tokens (prompt or padding).
    Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
    Returns:
        dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of the tokenized prompt and output strings. Then the returned dictionary should have the following keys:
            input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): the tokenized prompt and output strings, with the final token sliced off.
            labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): the input ids without the first token.
            response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): a mask on the response tokens in the labels.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

    # 获取 pad token
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define either pad_token_id or eos_token_id")

    # tokenize
    prompt_token_ids: list[list[int]] = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs
    ]
    output_token_ids: list[list[int]] = [
        tokenizer.encode(output, add_special_tokens=False) for output in output_strs
    ]
    prompt_and_output_token_ids: list[list[int]] = [
        prompt_ids + output_ids
        for prompt_ids, output_ids in zip(prompt_token_ids, output_token_ids, strict=True)
    ]

    if not prompt_and_output_token_ids:
        empty = torch.empty((0, 0), dtype=torch.long)
        return {
            "input_ids": empty,
            "labels": empty,
            "response_mask": torch.empty((0, 0), dtype=torch.bool),
        }

    # 获取最大长度进行 pad
    max_len = max(len(token_ids) for token_ids in prompt_and_output_token_ids)
    if max_len < 2:
        raise ValueError("each prompt + output must tokenize to at least two tokens")

    # 先分配矩阵
    batch_size = len(prompt_and_output_token_ids)
    padded_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool)

    # 往里填
    for row, (prompt_ids, prompt_and_output_ids) in enumerate(
        zip(prompt_token_ids, prompt_and_output_token_ids, strict=True)
    ):
        seq_len = len(prompt_and_output_ids)
        padded_input_ids[row, :seq_len] = torch.tensor(prompt_and_output_ids, dtype=torch.long)

        prompt_len = len(prompt_ids)
        response_start = max(prompt_len - 1, 0)
        response_end = seq_len - 1
        response_mask[row, response_start:response_end] = True

    return {
        "input_ids": padded_input_ids[:, :-1],
        "labels": padded_input_ids[:, 1:],
        "response_mask": response_mask,
    }

def compute_entropy(logits: Tensor) -> Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    H(x) = - \\sum_{x in vocab} p(x) \\log p(x)

    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size) containing unnormalized logits.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token prediction.
    Note: you should use a numerically stable method (e.g., using logsumexp) to avoid overflow
    """
    log_probs = torch.log_softmax(logits, dim=-1) # x_i - logsumexp(x)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool,
) -> dict[str, Tensor]:
    """
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt + response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling
        compute_entropy.
    Returns:
        dict[str, torch.Tensor].
        "log_probs" shape (batch_size, sequence_length), conditional log-probabilities log pθ(xt |x<t).
        "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy for each position (present only if return_token_entropy=True).
    Implementation tips:
        • Obtain logits with model(input_ids).logits.
    """
    logits = model(input_ids).logits # [batch_size, sequence_length, vocab_size]

    result = {}

    result["log_probs"] = torch.log_softmax(logits, dim=-1) # [batch_size, sequence_length, vocab_size]
    result["log_probs"] = torch.gather(
        result["log_probs"], dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result
