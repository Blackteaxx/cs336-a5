from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft_utils import (
    get_response_log_probs,
    masked_normalize,
    sft_microbatch_train_step,
)
from sft_math_experiment import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPT_PATH,
    DEFAULT_VAL_PATH,
    MathSFTDataset,
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


def response_length_from_completion(completion, response: str, tokenizer) -> int:
    token_ids = getattr(completion, "token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    return len(tokenizer.encode(response, add_special_tokens=False))


def sample_train_batch(
    examples: list[dict[str, Any]],
    batch_size: int,
    seed: int,
    ei_step: int,
) -> list[dict[str, Any]]:
    if batch_size > len(examples):
        raise ValueError(f"ei_batch_size={batch_size} exceeds train size {len(examples)}")

    rng = random.Random(seed + ei_step)
    indices = sorted(rng.sample(range(len(examples)), batch_size))
    batch = []
    for index in indices:
        row = dict(examples[index])
        row["train_index"] = index
        batch.append(row)
    return batch


def make_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )


def generate_rollouts(
    vllm_model: LLM,
    tokenizer,
    batch: list[dict[str, Any]],
    prompt_template: str,
    args: argparse.Namespace,
    ei_step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sampling_params = SamplingParams(
        n=args.rollouts_per_question,
        temperature=args.rollout_temperature,
        top_p=args.rollout_top_p,
        max_tokens=args.rollout_max_new_tokens,
        min_tokens=args.rollout_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    prompts = [prompt_template.format(question=item["problem"]) for item in batch]
    outputs = vllm_model.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    rollout_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    for batch_index, (item, prompt, output) in enumerate(
        zip(batch, prompts, outputs, strict=True)
    ):
        question_has_correct = False
        question_start = len(rollout_rows)
        for rollout_index, completion in enumerate(output.outputs):
            response = completion.text
            reward = r1_zero_reward_fn(response, item["expected_answer"])
            response_length = response_length_from_completion(
                completion,
                response,
                tokenizer,
            )
            is_correct = float(reward["answer_reward"]) == 1.0
            question_has_correct = question_has_correct or is_correct
            row = {
                "ei_step": ei_step,
                "train_index": item["train_index"],
                "batch_index": batch_index,
                "rollout_index": rollout_index,
                "problem": item["problem"],
                "ground_truth": item["expected_answer"],
                "prompt": prompt,
                "response": response,
                "response_length_tokens": response_length,
                "reward": reward,
                "is_correct": is_correct,
            }
            rollout_rows.append(row)
            if is_correct:
                sft_rows.append(
                    {
                        "ei_step": ei_step,
                        "train_index": item["train_index"],
                        "rollout_index": rollout_index,
                        "problem": item["problem"],
                        "expected_answer": item["expected_answer"],
                        "reasoning_trace": response,
                    }
                )

        for row in rollout_rows[question_start:]:
            row["question_has_correct_rollout"] = question_has_correct

    return rollout_rows, sft_rows


def rollout_metrics(
    rollout_rows: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    batch_size: int,
    rollouts_per_question: int,
    ei_step: int,
) -> dict[str, Any]:
    total_rollouts = max(1, len(rollout_rows))
    correct_rollouts = len(sft_rows)
    questions_with_correct = len(
        {row["train_index"] for row in rollout_rows if row["is_correct"]}
    )
    response_lengths = [int(row["response_length_tokens"]) for row in rollout_rows]
    return {
        "ei_step": ei_step,
        "ei/batch_size": batch_size,
        "ei/rollouts_per_question": rollouts_per_question,
        "ei/num_rollouts": float(len(rollout_rows)),
        "ei/num_correct_rollouts": float(correct_rollouts),
        "ei/rollout_accuracy": correct_rollouts / total_rollouts,
        "ei/num_questions_with_correct": float(questions_with_correct),
        "ei/question_pass_rate": questions_with_correct / max(1, batch_size),
        "ei/num_sft_examples": float(len(sft_rows)),
        "ei/avg_rollout_response_length": mean_or_zero(response_lengths),
    }


def train_sft_phase(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    prompt_template: str,
    sft_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    ei_step: int,
    record,
) -> int:
    if not sft_rows:
        record(
            {
                "ei_step": ei_step,
                "train_step": global_step,
                "train/skipped_empty_sft_dataset": 1.0,
            },
            step=global_step,
        )
        return global_step

    dataset = MathSFTDataset(
        sft_rows,
        prompt_template=prompt_template,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model.train()
    accumulation_count = 0
    optimizer.zero_grad(set_to_none=True)
    for sft_epoch in range(args.sft_epochs_per_ei_step):
        updates_this_epoch = (
            len(dataloader) + args.gradient_accumulation_steps - 1
        ) // args.gradient_accumulation_steps
        progress = tqdm(
            total=updates_this_epoch,
            desc=f"ei {ei_step}/{args.n_ei_steps} sft {sft_epoch + 1}/"
            f"{args.sft_epochs_per_ei_step}",
            unit="update",
        )
        accumulation_metrics = {
            "loss": 0.0,
            "scaled_loss": 0.0,
            "response_tokens": 0.0,
            "response_entropy": 0.0,
            "normalize_constant": 0.0,
        }
        for batch_idx, batch in enumerate(dataloader):
            batch = {key: value.to(device) for key, value in batch.items()}
            log_prob_outputs = get_response_log_probs(
                model,
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                return_token_entropy=True,
                attention_mask=batch["attention_mask"],
            )
            loss, loss_metadata = sft_microbatch_train_step(
                policy_log_probs=log_prob_outputs["log_probs"],
                response_mask=batch["response_mask"],
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=args.normalize_constant,
            )
            num_response_tokens = loss_metadata["num_response_tokens"].detach()
            normalize_constant = loss_metadata["normalize_constant"].detach()
            entropy = masked_normalize(
                log_prob_outputs["token_entropy"].detach(),
                batch["response_mask"],
                normalize_constant=float(num_response_tokens.clamp_min(1).item()),
            )
            accumulation_count += 1
            unscaled_loss = loss.detach() * args.gradient_accumulation_steps
            accumulation_metrics["loss"] += float(unscaled_loss.cpu())
            accumulation_metrics["scaled_loss"] += float(loss.detach().cpu())
            accumulation_metrics["response_tokens"] += float(num_response_tokens.cpu())
            accumulation_metrics["response_entropy"] += float(entropy.cpu())
            accumulation_metrics["normalize_constant"] += float(normalize_constant.cpu())

            should_step = (
                accumulation_count == args.gradient_accumulation_steps
                or batch_idx == len(dataloader) - 1
            )
            if not should_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            completed_accumulation_count = accumulation_count

            train_metrics = {
                "ei_step": ei_step,
                "train_step": global_step,
                "train/sft_epoch": sft_epoch + 1,
                "train/loss": (
                    accumulation_metrics["loss"] / completed_accumulation_count
                ),
                "train/scaled_loss": (
                    accumulation_metrics["scaled_loss"] / completed_accumulation_count
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
                "train/normalize_constant": (
                    accumulation_metrics["normalize_constant"]
                    / completed_accumulation_count
                ),
                "train/accumulated_microbatches": completed_accumulation_count,
            }
            accumulation_metrics = {
                "loss": 0.0,
                "scaled_loss": 0.0,
                "response_tokens": 0.0,
                "response_entropy": 0.0,
                "normalize_constant": 0.0,
            }
            accumulation_count = 0
            progress.set_postfix(loss=train_metrics["train/loss"])
            progress.update(1)
            if global_step % args.log_every_steps == 0:
                record(train_metrics, step=global_step)
        progress.close()

    return global_step


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_examples = read_json_or_jsonl(args.train_path)
    val_examples = read_json_or_jsonl(args.val_path)
    prompt_template = Path(args.prompt_path).read_text()
    if args.run_name is None:
        args.run_name = (
            f"ei-math-db{args.ei_batch_size}-g{args.rollouts_per_question}-"
            f"ep{args.sft_epochs_per_ei_step}-seed{args.seed}"
        )

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    model, tokenizer, device = load_model_and_tokenizer(args)
    vllm_model = init_vllm(
        model_id=args.model_name_or_path,
        device=args.eval_device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
    )
    run = setup_swanlab(args, selected_size=args.ei_batch_size)
    metrics_file = (output_dir / "metrics.jsonl").open("a")

    def record(metrics: dict[str, Any], step: int | None = None) -> None:
        row = {"time": time.time(), **metrics}
        metrics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics_file.flush()
        swanlab_log(run, metrics, step=step)

    global_step = 0
    record(
        {
            "dataset/total_examples": len(train_examples),
            "dataset/val_examples": len(val_examples),
            "ei_step": 0,
            "train_step": 0,
        },
        step=0,
    )

    if args.eval_at_start:
        load_policy_into_vllm_instance(model, vllm_model)
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
        record({"ei_step": 0, "eval_step": 0, **eval_metrics}, step=global_step)

    eval_metrics: dict[str, float] = {}
    for ei_step in range(1, args.n_ei_steps + 1):
        train_batch = sample_train_batch(
            train_examples,
            args.ei_batch_size,
            args.seed,
            ei_step,
        )
        load_policy_into_vllm_instance(model, vllm_model)
        rollout_rows, sft_rows = generate_rollouts(
            vllm_model,
            tokenizer,
            train_batch,
            prompt_template,
            args,
            ei_step,
        )
        if args.save_rollouts:
            write_jsonl(output_dir / f"rollouts_ei_step_{ei_step}.jsonl", rollout_rows)
        if args.save_ei_datasets:
            write_jsonl(output_dir / f"sft_data_ei_step_{ei_step}.jsonl", sft_rows)

        record(
            rollout_metrics(
                rollout_rows,
                sft_rows,
                args.ei_batch_size,
                args.rollouts_per_question,
                ei_step,
            ),
            step=global_step,
        )

        optimizer = make_optimizer(args, model)
        global_step = train_sft_phase(
            model,
            tokenizer,
            device,
            prompt_template,
            sft_rows,
            args,
            optimizer,
            global_step,
            ei_step,
            record,
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
            ei_step,
            output_dir / f"eval_step_{ei_step}.jsonl" if args.save_eval_outputs else None,
            output_dir / f"validation_samples_step_{ei_step}.jsonl",
            run,
        )
        record(
            {
                "ei_step": ei_step,
                "eval_step": ei_step,
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
        "n_ei_steps": args.n_ei_steps,
        "ei_batch_size": args.ei_batch_size,
        "rollouts_per_question": args.rollouts_per_question,
        "sft_epochs_per_ei_step": args.sft_epochs_per_ei_step,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expert iteration on MATH.")
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-path", default=DEFAULT_MATH_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--prompt-path", default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-dir", default="outputs/expert_iteration")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:1")

    parser.add_argument("--n-ei-steps", type=int, default=5)
    parser.add_argument("--ei-batch-size", type=int, default=512)
    parser.add_argument("--rollouts-per-question", type=int, default=4)
    parser.add_argument("--sft-epochs-per-ei-step", type=int, default=1)

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--normalize-constant", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        default="sdpa",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--rollout-temperature", type=float, default=0.7)
    parser.add_argument("--rollout-top-p", type=float, default=0.95)
    parser.add_argument("--rollout-max-new-tokens", type=int, default=1024)
    parser.add_argument("--rollout-min-tokens", type=int, default=4)

    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--eval-samples", type=int, default=5000)
    parser.add_argument("--eval-max-new-tokens", type=int, default=1024)
    parser.add_argument("--eval-min-tokens", type=int, default=4)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--num-validation-samples-to-log", type=int, default=16)
    parser.add_argument("--validation-sample-seed", type=int, default=0)
    parser.add_argument("--validation-sample-max-chars", type=int, default=1024)
    parser.add_argument("--validation-sample-entropy-chunk-size", type=int, default=8)
    parser.add_argument("--save-eval-outputs", action="store_true")
    parser.add_argument(
        "--save-rollouts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-ei-datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--log-every-steps", type=int, default=1)
    parser.add_argument("--swanlab-project", default="cs336-a5-expert-iteration")
    parser.add_argument("--swanlab-log-dir", default="outputs/swanlab")
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="local",
    )
    parser.add_argument(
        "--save-final-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    summary = train(parse_args())
    print(json.dumps(summary, indent=2))
