from __future__ import annotations

import json
import inspect
import random
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import typer
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo_utils import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    masked_mean,
)
from cs336_alignment.sft_utils import get_response_log_probs, tokenize_prompt_and_output
from sft_math_experiment import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPT_PATH,
    DEFAULT_VAL_PATH,
    evaluate,
    init_vllm,
    load_model_and_tokenizer,
    load_policy_into_vllm_instance,
    mean_or_zero,
    read_json_or_jsonl,
    setup_swanlab,
    swanlab_log,
    write_jsonl,
)


DEFAULT_MATH_TRAIN_PATH = (
    "/workspace/assignment5-alignment/data/sft-cs336-assign5-datasets/"
    "sft-reason/train.jsonl"
)


class LossType(str, Enum):
    no_baseline = "no_baseline"
    reinforce_with_baseline = "reinforce_with_baseline"
    grpo_clip = "grpo_clip"


class DType(str, Enum):
    bf16 = "bf16"
    fp16 = "fp16"


class AttentionImplementation(str, Enum):
    auto = "auto"
    flash_attention_2 = "flash_attention_2"
    sdpa = "sdpa"
    eager = "eager"


class SwanLabMode(str, Enum):
    online = "online"
    local = "local"
    offline = "offline"
    disabled = "disabled"


@dataclass
class RolloutBatch:
    prompts: list[str]
    responses: list[str]
    repeated_ground_truths: list[Any]
    rows: list[dict[str, Any]]
    raw_rewards: torch.Tensor
    advantages: torch.Tensor


def make_optimizer(args: SimpleNamespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    use_fused = (
        "fused" in inspect.signature(torch.optim.AdamW).parameters
        and str(args.train_device).startswith("cuda")
    )
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        fused=use_fused,
    )


def check_args(args: SimpleNamespace) -> None:
    assert args.train_batch_size % args.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    assert args.rollout_batch_size % args.group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    assert args.train_batch_size >= args.group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    assert args.rollout_batch_size % micro_train_batch_size == 0, (
        "rollout_batch_size must be divisible by micro_train_batch_size"
    )
    if args.loss_type == "grpo_clip":
        assert args.epochs_per_rollout_batch > 1 or args.force_grpo_clip_on_policy, (
            "GRPO-Clip is mainly useful off-policy; pass "
            "--force-grpo-clip-on-policy to run it with one epoch."
        )


def sample_prompt_batch(
    examples: list[dict[str, Any]],
    prompt_template: str,
    n_prompts: int,
) -> list[dict[str, Any]]:
    if n_prompts > len(examples):
        raise ValueError(f"n_prompts={n_prompts} exceeds train size {len(examples)}")

    indices = random.sample(range(len(examples)), n_prompts)
    batch = []
    for prompt_batch_index, example_index in enumerate(indices):
        item = dict(examples[example_index])
        item["train_index"] = example_index
        item["prompt_batch_index"] = prompt_batch_index
        item["prompt"] = prompt_template.format(question=item["problem"])
        batch.append(item)
    return batch


def normalize_ground_truth(value: Any) -> str | list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return str(value)


def normalize_math_examples(
    examples: list[dict[str, Any]],
    path: str | Path,
) -> list[dict[str, Any]]:
    normalized = []
    for index, example in enumerate(examples):
        if "problem" not in example:
            raise ValueError(f"Missing problem in {path} at row {index}")
        answer = example.get("expected_answer", example.get("answer"))
        if answer is None:
            raise ValueError(f"Missing expected_answer/answer in {path} at row {index}")
        item = dict(example)
        item["expected_answer"] = normalize_ground_truth(answer)
        normalized.append(item)
    return normalized


def response_length_from_completion(completion, response: str, tokenizer) -> int:
    token_ids = getattr(completion, "token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    return len(tokenizer.encode(response, add_special_tokens=False))


def generate_rollout_batch(
    vllm_model: LLM,
    tokenizer,
    prompt_batch: list[dict[str, Any]],
    args: SimpleNamespace,
    grpo_step: int,
) -> tuple[RolloutBatch, dict[str, Any]]:
    """
    TODO(grpo_train_loop): implement rollout generation and reward computation.

    Expected shape/order contract:
    - `prompt_batch` has `rollout_batch_size // group_size` questions.
    - Repeat each prompt `group_size` times and ask vLLM for one completion per
      request, matching the reference implementation's rollout layout.
    - Flatten outputs in prompt-major order:
      prompt_0 rollout_0, ..., prompt_0 rollout_g-1, prompt_1 rollout_0, ...
    - Score each response with `r1_zero_reward_fn(response, expected_answer)`.
    - Call `compute_group_normalized_rewards(...)`.
    - Return a RolloutBatch whose tensors have shape `(rollout_batch_size, 1)`.
    """
    sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        top_p=args.sampling_top_p,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    request_prompts = []
    request_items = []
    for item in prompt_batch:
        for rollout_index in range(args.group_size):
            request_prompts.append(item["prompt"])
            request_items.append((item, rollout_index))

    # Keep vLLM output order aligned with the advantages.
    outputs = vllm_model.generate(
        prompts=request_prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    rows, repeated_prompts, responses, repeated_ground_truths = [], [], [], []
    for (item, rollout_index), prompt, output in zip(
        request_items,
        request_prompts,
        outputs,
        strict=True,
    ):
        ground_truth = item["expected_answer"]
        completion = output.outputs[0]
        response = completion.text
        reward = r1_zero_reward_fn(response, ground_truth)
        response_length = response_length_from_completion(
            completion,
            response,
            tokenizer,
        )

        responses.append(response)
        repeated_prompts.append(prompt)
        repeated_ground_truths.append(ground_truth)

        rows.append(
            {
                "grpo_step": grpo_step,
                "train_index": item["train_index"],
                "prompt_batch_index": item["prompt_batch_index"],
                "rollout_index": rollout_index,
                "problem": item["problem"],
                "ground_truth": ground_truth,
                "prompt": prompt,
                "response": response,
                "response_length_tokens": response_length,
                "reward": reward,
            }
            )
    
    advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
        reward_fn=r1_zero_reward_fn,
        rollout_responses=responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=args.group_size,
        advantage_eps=args.advantage_eps,
        normalize_by_std=args.use_std_normalization,
    )
    return (
        RolloutBatch(
            prompts=repeated_prompts,
            responses=responses,
            repeated_ground_truths=repeated_ground_truths,
            rows=rows,
            raw_rewards=raw_rewards[:, None],
            advantages=advantages[:, None],
        ),
        rollout_metrics(rows, reward_metadata, args, grpo_step),
    )


def rollout_metrics(
    rows: list[dict[str, Any]],
    reward_metadata: dict[str, float],
    args: SimpleNamespace,
    grpo_step: int,
) -> dict[str, Any]:
    response_lengths = [int(row["response_length_tokens"]) for row in rows]
    correct = [row for row in rows if float(row["reward"]["answer_reward"]) == 1.0]
    incorrect = [row for row in rows if float(row["reward"]["answer_reward"]) != 1.0]
    metrics: dict[str, Any] = {
        "grpo_step": grpo_step,
        "rollout/num_rollouts": float(len(rows)),
        "rollout/group_size": float(args.group_size),
        "rollout/avg_response_length": mean_or_zero(response_lengths),
        "rollout/avg_correct_response_length": mean_or_zero(
            [int(row["response_length_tokens"]) for row in correct]
        ),
        "rollout/avg_incorrect_response_length": mean_or_zero(
            [int(row["response_length_tokens"]) for row in incorrect]
        ),
        "rollout/accuracy": len(correct) / max(1, len(rows)),
    }
    metrics.update({f"rollout/{key}": value for key, value in reward_metadata.items()})
    return metrics


def tokenize_rollout_batch(
    rollout_batch: RolloutBatch,
    tokenizer,
    args: SimpleNamespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    TODO(grpo_train_loop): tokenize prompts and sampled responses for policy updates.

    Use `tokenize_prompt_and_output`, truncate to `args.max_length`, and create
    `attention_mask`. Put all tensors on `device`. The returned dict should include:
    input_ids, labels, response_mask, attention_mask, raw_rewards, advantages.
    """
    batch = tokenize_prompt_and_output(
        rollout_batch.prompts,
        rollout_batch.responses,
        tokenizer,
    )
    input_ids = batch["input_ids"][:, : args.max_length].to(device)
    labels = batch["labels"][:, : args.max_length].to(device)
    response_mask = batch["response_mask"][:, : args.max_length].to(device)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define either pad_token_id or eos_token_id")

    attention_mask = (input_ids != pad_token_id).long()
    raw_rewards = rollout_batch.raw_rewards.to(device)
    advantages = rollout_batch.advantages.to(device)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
        "attention_mask": attention_mask,
        "raw_rewards": raw_rewards,
        "advantages": advantages,
    }


@torch.no_grad()
def compute_old_log_probs_once(
    model: PreTrainedModel,
    tokenized_rollouts: dict[str, torch.Tensor],
    args: SimpleNamespace,
) -> torch.Tensor | None:
    """
    TODO(grpo_train_loop): cache old log-probs for off-policy GRPO-Clip.

    Return None unless `args.loss_type == "grpo_clip"`. When computing old
    log-probs, do not build a gradient graph.
    """
    if args.loss_type != "grpo_clip":
        return None

    output = get_response_log_probs(
        model,
        input_ids=tokenized_rollouts["input_ids"],
        labels=tokenized_rollouts["labels"],
        return_token_entropy=False,
        attention_mask=tokenized_rollouts["attention_mask"],
    )
    return output["log_probs"].detach()


def iter_train_minibatches(
    tokenized_rollouts: dict[str, torch.Tensor],
    old_log_probs: torch.Tensor | None,
    args: SimpleNamespace,
):
    """
    TODO(grpo_train_loop): yield shuffled train batches for each rollout epoch.

    Suggested behavior:
    - For each epoch in `epochs_per_rollout_batch`, shuffle rollout indices.
    - Slice `train_batch_size` examples.
    - Inside each train batch, slice microbatches of
      `train_batch_size // gradient_accumulation_steps`.
    - Include matching old_log_probs slices when present.
    """
    rollout_batch_size = tokenized_rollouts["input_ids"].shape[0]
    micro_train_batch_size = (
        args.train_batch_size // args.gradient_accumulation_steps
    )
    assert micro_train_batch_size > 0
    assert rollout_batch_size == args.rollout_batch_size
    if old_log_probs is not None:
        assert old_log_probs.shape[:1] == (rollout_batch_size,)

    device = tokenized_rollouts["input_ids"].device
    tensor_keys = [
        key for key, value in tokenized_rollouts.items() if torch.is_tensor(value)
    ]

    for train_epoch in range(args.epochs_per_rollout_batch):
        rollout_indices = torch.arange(rollout_batch_size, device=device)

        for train_start in range(0, rollout_batch_size, args.train_batch_size):
            train_end = min(train_start + args.train_batch_size, rollout_batch_size)
            train_indices = rollout_indices[train_start:train_end]

            for micro_start in range(0, train_indices.numel(), micro_train_batch_size):
                micro_end = min(
                    micro_start + micro_train_batch_size,
                    train_indices.numel(),
                )
                micro_indices = train_indices[micro_start:micro_end]
                microbatch = {
                    key: tokenized_rollouts[key][micro_indices]
                    for key in tensor_keys
                }
                if old_log_probs is not None:
                    microbatch["old_log_probs"] = old_log_probs[micro_indices]

                should_step = micro_end == train_indices.numel()
                yield microbatch, should_step, train_epoch


def train_one_rollout_batch(
    model: PreTrainedModel,
    tokenizer,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    rollout_batch: RolloutBatch,
    args: SimpleNamespace,
    global_step: int,
    grpo_step: int,
    record,
) -> int:
    """
    TODO(grpo_train_loop): implement the policy update phase.

    The expected inner loop is:
    1. tokenize rollout prompts/responses.
    2. cache old log-probs once if using GRPO-Clip.
    3. for each epoch/train batch/microbatch:
       - recompute current policy log-probs with gradients.
       - call `grpo_microbatch_train_step`.
       - compute/log token entropy over response tokens.
       - after `gradient_accumulation_steps`, clip gradients and optimizer.step().
    4. log loss, grad norm, entropy, clip fraction if present, and reward stats.
    """
    tokenized_rollouts = tokenize_rollout_batch(rollout_batch, tokenizer, args, device)
    old_log_probs = compute_old_log_probs_once(model, tokenized_rollouts, args)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_updates = args.epochs_per_rollout_batch * (
        (args.rollout_batch_size + args.train_batch_size - 1)
        // args.train_batch_size
    )
    progress = tqdm(
        total=total_updates,
        desc=f"grpo {grpo_step}/{args.n_grpo_steps}",
        unit="update",
    )

    accumulation_count = 0
    accumulation_metrics = {
        "loss": 0.0,
        "scaled_loss": 0.0,
        "response_tokens": 0.0,
        "response_entropy": 0.0,
        "raw_reward": 0.0,
        "advantage": 0.0,
        "clip_fraction": 0.0,
        "clip_fraction_count": 0,
    }

    for microbatch, should_step, train_epoch in iter_train_minibatches(
        tokenized_rollouts,
        old_log_probs,
        args,
    ):
        autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        with torch.autocast(
            device_type=str(device).split(":")[0],
            dtype=autocast_dtype,
        ):
            log_prob_outputs = get_response_log_probs(
                model,
                input_ids=microbatch["input_ids"],
                labels=microbatch["labels"],
                return_token_entropy=True,
            )
        loss, loss_metadata = grpo_microbatch_train_step(
            policy_log_probs=log_prob_outputs["log_probs"],
            response_mask=microbatch["response_mask"],
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            loss_type=args.loss_type,
            raw_rewards=microbatch["raw_rewards"],
            advantages=microbatch["advantages"],
            old_log_probs=microbatch.get("old_log_probs"),
            cliprange=args.cliprange,
        )

        response_tokens = microbatch["response_mask"].sum()
        response_entropy = masked_mean(
            log_prob_outputs["token_entropy"].detach(),
            microbatch["response_mask"],
            dim=None,
        )

        accumulation_count += 1
        unscaled_loss = loss.detach() * args.gradient_accumulation_steps
        accumulation_metrics["loss"] += float(unscaled_loss.cpu())
        accumulation_metrics["scaled_loss"] += float(loss.detach().cpu())
        accumulation_metrics["response_tokens"] += float(response_tokens.cpu())
        accumulation_metrics["response_entropy"] += float(response_entropy.cpu())
        accumulation_metrics["raw_reward"] += float(
            microbatch["raw_rewards"].detach().mean().cpu()
        )
        accumulation_metrics["advantage"] += float(
            microbatch["advantages"].detach().mean().cpu()
        )

        if "grpo_clip/was_clipped" in loss_metadata:
            clip_fraction = masked_mean(
                loss_metadata["grpo_clip/was_clipped"].float().detach(),
                microbatch["response_mask"],
                dim=None,
            )
            accumulation_metrics["clip_fraction"] += float(clip_fraction.cpu())
            accumulation_metrics["clip_fraction_count"] += 1

        if not should_step:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        completed_accumulation_count = accumulation_count

        train_metrics = {
            "grpo_step": grpo_step,
            "train_step": global_step,
            "train/rollout_epoch": train_epoch + 1,
            "train/loss": (
                accumulation_metrics["loss"] / completed_accumulation_count
            ),
            "train/scaled_loss": (
                accumulation_metrics["scaled_loss"]
                / completed_accumulation_count
            ),
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/grad_norm": float(grad_norm.detach().cpu()),
            "train/response_tokens": (
                accumulation_metrics["response_tokens"]
                / completed_accumulation_count
            ),
            "train/response_entropy": (
                accumulation_metrics["response_entropy"]
                / completed_accumulation_count
            ),
            "train/raw_reward": (
                accumulation_metrics["raw_reward"]
                / completed_accumulation_count
            ),
            "train/advantage": (
                accumulation_metrics["advantage"]
                / completed_accumulation_count
            ),
            "train/accumulated_microbatches": completed_accumulation_count,
        }
        if accumulation_metrics["clip_fraction_count"]:
            train_metrics["train/clip_fraction"] = (
                accumulation_metrics["clip_fraction"]
                / accumulation_metrics["clip_fraction_count"]
            )

        accumulation_count = 0
        accumulation_metrics = {
            "loss": 0.0,
            "scaled_loss": 0.0,
            "response_tokens": 0.0,
            "response_entropy": 0.0,
            "raw_reward": 0.0,
            "advantage": 0.0,
            "clip_fraction": 0.0,
            "clip_fraction_count": 0,
        }
        progress.set_postfix(loss=train_metrics["train/loss"])
        progress.update(1)
        if global_step % args.log_every_steps == 0:
            record(train_metrics, step=global_step)

    progress.close()
    return global_step


def train(args: SimpleNamespace) -> dict[str, Any]:
    check_args(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    train_examples = normalize_math_examples(
        read_json_or_jsonl(args.train_path),
        args.train_path,
    )
    val_examples = normalize_math_examples(
        read_json_or_jsonl(args.val_path),
        args.val_path,
    )
    prompt_template = Path(args.prompt_path).read_text()

    if args.run_name is None:
        args.run_name = (
            f"grpo-math-rb{args.rollout_batch_size}-g{args.group_size}-"
            f"{args.loss_type}-seed{args.seed}"
        )

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    model, tokenizer, device = load_model_and_tokenizer(args)
    vllm_model = init_vllm(
        model_id=args.model_name_or_path,
        device=args.eval_device,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
    )
    optimizer = make_optimizer(args, model)
    run = setup_swanlab(args, selected_size=args.rollout_batch_size)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    metrics_file = (output_dir / "metrics.jsonl").open("w")

    def record(metrics: dict[str, Any], step: int | None = None) -> None:
        row = {"time": time.time(), **metrics}
        metrics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics_file.flush()
        swanlab_log(run, metrics, step=step)

    global_step = 0
    record(
        {
            "dataset/train_examples": len(train_examples),
            "dataset/val_examples": len(val_examples),
            "grpo_step": 0,
            "train_step": 0,
        },
        step=0,
    )

    if args.eval_at_start:
        eval_metrics = evaluate(
            model,
            vllm_model,
            tokenizer,
            val_examples,
            prompt_template,
            args,
            device,
            0,
            output_dir / "eval_step_0.jsonl" if args.save_eval_outputs else None,
            output_dir / "validation_samples_step_0.jsonl",
            run,
        )
        record({"grpo_step": 0, "eval_step": 0, **eval_metrics}, step=global_step)

    eval_metrics: dict[str, float] = {}
    n_prompts = args.rollout_batch_size // args.group_size
    for grpo_step in range(1, args.n_grpo_steps + 1):
        prompt_batch = sample_prompt_batch(
            train_examples,
            prompt_template,
            n_prompts,
        )

        load_policy_into_vllm_instance(model, vllm_model)
        rollout_batch, rollout_log = generate_rollout_batch(
            vllm_model,
            tokenizer,
            prompt_batch,
            args,
            grpo_step,
        )
        if args.save_rollouts:
            write_jsonl(output_dir / f"rollouts_grpo_step_{grpo_step}.jsonl", rollout_batch.rows)
        record(rollout_log, step=global_step)

        global_step = train_one_rollout_batch(
            model,
            tokenizer,
            device,
            optimizer,
            rollout_batch,
            args,
            global_step,
            grpo_step,
            record,
        )

        if args.eval_every_steps and grpo_step % args.eval_every_steps == 0:
            load_policy_into_vllm_instance(model, vllm_model)
            eval_metrics = evaluate(
                model,
                vllm_model,
                tokenizer,
                val_examples,
                prompt_template,
                args,
                device,
                grpo_step,
                output_dir / f"eval_step_{grpo_step}.jsonl"
                if args.save_eval_outputs
                else None,
                output_dir / f"validation_samples_step_{grpo_step}.jsonl",
                run,
            )
            record(
                {
                    "grpo_step": grpo_step,
                    "eval_step": grpo_step,
                    "train_step": global_step,
                    **eval_metrics,
                },
                step=global_step,
            )

    load_policy_into_vllm_instance(model, vllm_model)
    eval_metrics = evaluate(
        model,
        vllm_model,
        tokenizer,
        val_examples,
        prompt_template,
        args,
        device,
        args.n_grpo_steps,
        output_dir / f"eval_step_{args.n_grpo_steps}.jsonl"
        if args.save_eval_outputs
        else None,
        output_dir / f"validation_samples_step_{args.n_grpo_steps}.jsonl",
        run,
    )
    record(
        {
            "grpo_step": args.n_grpo_steps,
            "eval_step": args.n_grpo_steps,
            "train_step": global_step,
            **eval_metrics,
        },
        step=global_step,
    )

    summary = {
        "run_name": args.run_name,
        "train_path": args.train_path,
        "val_path": args.val_path,
        "prompt_path": args.prompt_path,
        "n_grpo_steps": args.n_grpo_steps,
        "rollout_batch_size": args.rollout_batch_size,
        "group_size": args.group_size,
        "loss_type": args.loss_type,
        "global_step": global_step,
        **eval_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.save_final_model:
        final_dir = output_dir / "final_model"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

    metrics_file.close()
    if run is not None:
        try:
            import swanlab

            swanlab.finish()
        except Exception:
            pass
    return summary


def main(
    model_name_or_path: str = typer.Option(DEFAULT_MODEL_PATH),
    train_path: str = typer.Option(DEFAULT_MATH_TRAIN_PATH),
    val_path: str = typer.Option(DEFAULT_VAL_PATH),
    prompt_path: str = typer.Option(DEFAULT_PROMPT_PATH),
    output_dir: str = typer.Option("outputs/grpo_math"),
    run_name: str | None = typer.Option(None),
    seed: int = typer.Option(42),
    train_device: str = typer.Option("cuda:0"),
    eval_device: str = typer.Option("cuda:1"),
    n_grpo_steps: int = typer.Option(200),
    learning_rate: float = typer.Option(3e-5),
    advantage_eps: float = typer.Option(1e-6),
    rollout_batch_size: int = typer.Option(256),
    group_size: int = typer.Option(8),
    sampling_temperature: float = typer.Option(1.0),
    sampling_top_p: float = typer.Option(1.0),
    sampling_min_tokens: int = typer.Option(4),
    sampling_max_tokens: int = typer.Option(1024),
    epochs_per_rollout_batch: int = typer.Option(1),
    train_batch_size: int = typer.Option(256),
    gradient_accumulation_steps: int = typer.Option(256),
    loss_type: LossType = typer.Option(LossType.reinforce_with_baseline),
    use_std_normalization: bool = typer.Option(
        True,
        "--use-std-normalization/--no-use-std-normalization",
    ),
    cliprange: float = typer.Option(0.2),
    force_grpo_clip_on_policy: bool = typer.Option(False),
    max_length: int = typer.Option(2048),
    max_grad_norm: float = typer.Option(1.0),
    adam_beta1: float = typer.Option(0.9),
    adam_beta2: float = typer.Option(0.95),
    adam_eps: float = typer.Option(1e-8),
    dtype: DType = typer.Option(DType.bf16),
    attn_implementation: AttentionImplementation = typer.Option(
        AttentionImplementation.sdpa,
    ),
    gradient_checkpointing: bool = typer.Option(True),
    eval_at_start: bool = typer.Option(False),
    eval_every_steps: int = typer.Option(4),
    eval_samples: int = typer.Option(1024),
    eval_random_sample: bool = typer.Option(
        True,
        "--eval-random-sample/--eval-prefix-sample",
    ),
    eval_sample_seed: int = typer.Option(42),
    eval_max_new_tokens: int = typer.Option(1024),
    eval_min_tokens: int = typer.Option(4),
    eval_temperature: float = typer.Option(1.0),
    eval_top_p: float = typer.Option(1.0),
    gpu_memory_utilization: float = typer.Option(0.85),
    vllm_max_model_len: int | None = typer.Option(None),
    num_validation_samples_to_log: int = typer.Option(16),
    validation_sample_seed: int = typer.Option(0),
    validation_sample_max_chars: int = typer.Option(1024),
    validation_sample_entropy_chunk_size: int = typer.Option(8),
    save_eval_outputs: bool = typer.Option(False),
    save_rollouts: bool = typer.Option(
        True,
        "--save-rollouts/--no-save-rollouts",
    ),
    log_every_steps: int = typer.Option(1),
    swanlab_project: str = typer.Option("cs336-a5-grpo-math"),
    swanlab_log_dir: str = typer.Option("outputs/swanlab"),
    swanlab_mode: SwanLabMode = typer.Option(SwanLabMode.local),
    save_final_model: bool = typer.Option(
        True,
        "--save-final-model/--no-save-final-model",
    ),
) -> None:
    values = locals()
    args = SimpleNamespace(
        **{
            key: value.value if isinstance(value, Enum) else value
            for key, value in values.items()
        }
    )
    summary = train(args)
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    typer.run(main)
