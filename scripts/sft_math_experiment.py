from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft_utils import (
    compute_entropy,
    get_response_log_probs,
    masked_normalize,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)


DEFAULT_MODEL_PATH = "/workspace/assignment5-alignment/model/Qwen2.5-Math-1.5B"
DEFAULT_TRAIN_PATH = (
    "/workspace/assignment5-alignment/data/sft-cs336-assign5-datasets/"
    "sft-reason/sft_gpt-oss-120b.jsonl"
)
DEFAULT_VAL_PATH = (
    "/workspace/assignment5-alignment/data/sft-cs336-assign5-datasets/sft-reason/val.jsonl"
)
DEFAULT_PROMPT_PATH = (
    "/workspace/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
)


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text().strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of examples in {path}")
    return data


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_examples(
    examples: list[dict[str, Any]],
    dataset_size: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Choose a subset of examples based on the specified dataset size."""

    if dataset_size == "full":
        return list(examples)

    size = int(dataset_size)
    if size > len(examples):
        raise ValueError(f"dataset_size={size} exceeds dataset length {len(examples)}")

    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    selected = sorted(indices[:size])
    return [examples[i] for i in selected]


def normalize_response(response: str, prompt: str, eos_token: str | None) -> str:
    """Normalize the response by stripping whitespace and adding the EOS token if needed."""
    response = response.strip()
    if prompt.rstrip().endswith("<think>") and response.startswith("<think>"):
        response = response[len("<think>") :].lstrip()
    # 显式 + eos token，SFT 为了让模型也学会停止
    if eos_token and not response.endswith(eos_token):
        response = response + eos_token
    return response


class MathSFTDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        prompt_template: str,
        tokenizer,
        max_length: int,
    ) -> None:
        self.examples = examples
        self.prompt_template = prompt_template
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        prompt = self.prompt_template.format(question=example["problem"])
        response = normalize_response(
            example["reasoning_trace"], prompt, self.tokenizer.eos_token
        )
        return {"prompt": prompt, "response": response}

    def collate_fn(self, items: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        batch = tokenize_prompt_and_output(
            [item["prompt"] for item in items],
            [item["response"] for item in items],
            self.tokenizer,
        )
        for key in ("input_ids", "labels", "response_mask"):
            batch[key] = batch[key][:, : self.max_length]

        pad_id = self.tokenizer.pad_token_id
        attention_mask = (batch["input_ids"] != pad_id).long()
        batch["attention_mask"] = attention_mask
        return batch


def setup_swanlab(args: argparse.Namespace, selected_size: int):
    if args.swanlab_mode == "disabled":
        return None
    try:
        import swanlab

        run = swanlab.init(
            project=args.swanlab_project,
            name=args.run_name,
            mode=args.swanlab_mode,
            log_dir=args.swanlab_log_dir,
            config={**vars(args), "selected_dataset_size": selected_size},
        )
        return run
    except Exception as exc:
        print(f"[warn] SwanLab init failed; continuing without SwanLab: {exc}")
        return None


def swanlab_log(run, payload: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    try:
        if step is None:
            run.log(payload)
        else:
            run.log(payload, step=step)
    except Exception as exc:
        print(f"[warn] SwanLab log failed: {exc}")


def response_token_length(output, response: str, tokenizer) -> int:
    token_ids = getattr(output.outputs[0], "token_ids", None)
    if token_ids is not None:
        return len(token_ids)
    return len(tokenizer.encode(response, add_special_tokens=False))


def mean_or_zero(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def truncate_for_log(text: str, max_chars: int) -> str:
    text = text.replace("\r", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def validation_samples_to_table_data(
    samples: list[dict[str, Any]],
    max_chars: int,
) -> tuple[list[str], list[list[Any]]]:
    headers = [
        "idx",
        "prompt",
        "response",
        "gt",
        "format",
        "answer",
        "reward",
        "avg entropy",
        "response tokens",
    ]
    rows = []
    for sample in samples:
        reward = sample["reward"]
        rows.append(
            [
                sample["eval_index"],
                truncate_for_log(sample["prompt"], max_chars),
                truncate_for_log(sample["response"], max_chars),
                str(sample["ground_truth"]),
                float(reward["format_reward"]),
                float(reward["answer_reward"]),
                float(reward["reward"]),
                round(float(sample["avg_response_entropy"]), 4),
                int(sample["response_length_tokens"]),
            ]
        )
    return headers, rows


def validation_samples_to_swanlab_table(
    samples: list[dict[str, Any]],
    max_chars: int,
):
    import swanlab

    headers, rows = validation_samples_to_table_data(samples, max_chars)
    return swanlab.echarts.Table().add(headers, rows)


def swanlab_log_validation_samples(
    run,
    samples: list[dict[str, Any]],
    step: int,
    max_chars: int,
) -> None:
    if run is None or not samples:
        return
    try:
        table = validation_samples_to_swanlab_table(samples, max_chars)
        run.log(
            {
                "eval/validation_samples_table": table,
            },
            step=step,
        )
    except Exception as exc:
        print(f"[warn] SwanLab validation sample log failed: {exc}")


@torch.no_grad()
def compute_sample_response_entropies(
    model: PreTrainedModel,
    tokenizer,
    device: torch.device,
    samples: list[dict[str, Any]],
    max_length: int,
    token_chunk_size: int,
) -> list[float]:
    if not samples:
        return []

    model.eval()
    entropies: list[float] = []
    for sample in samples:
        batch = tokenize_prompt_and_output(
            [sample["prompt"]],
            [sample["response"]],
            tokenizer,
        )
        for key in ("input_ids", "labels", "response_mask"):
            batch[key] = batch[key][:, :max_length].to(device)
        attention_mask = (batch["input_ids"] != tokenizer.pad_token_id).long()

        logits = model(
            batch["input_ids"],
            attention_mask=attention_mask,
        ).logits[0]
        response_positions = torch.nonzero(
            batch["response_mask"][0],
            as_tuple=False,
        ).flatten()
        if response_positions.numel() == 0:
            entropies.append(0.0)
            continue

        entropy_sum = torch.zeros((), device=device, dtype=torch.float32)
        for start in range(0, response_positions.numel(), token_chunk_size):
            positions = response_positions[start : start + token_chunk_size]
            entropy_sum += compute_entropy(logits[positions]).float().sum()

        entropies.append(float((entropy_sum / response_positions.numel()).cpu()))
        del batch, attention_mask, logits, response_positions, entropy_sum
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return entropies


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int | None = None,
) -> LLM:
    """Start vLLM on the GPU reserved for evaluation."""
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """Copy current policy weights into the standing vLLM evaluation instance."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def load_model_and_tokenizer(args: argparse.Namespace):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    # if args.attn_implementation != "auto":
    #     load_kwargs["attn_implementation"] = args.attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **load_kwargs,
        )
    except Exception:
        if args.attn_implementation == "flash_attention_2":
            print("[warn] flash_attention_2 load failed; retrying with sdpa")
            load_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                **load_kwargs,
            )
        else:
            raise

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    device = torch.device(args.train_device)
    model.to(device)
    return model, tokenizer, device


def evaluate(
    policy_model: PreTrainedModel,
    vllm_model: LLM,
    tokenizer,
    val_examples: list[dict[str, Any]],
    prompt_template: str,
    args: argparse.Namespace,
    device: torch.device,
    eval_step: int,
    output_path: Path | None,
    sample_output_path: Path | None,
    swanlab_run,
) -> dict[str, float]:
    if args.eval_samples and args.eval_samples < len(val_examples):
        val_examples = val_examples[: args.eval_samples]

    sampling_params = SamplingParams(
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        max_tokens=args.eval_max_new_tokens,
        min_tokens=args.eval_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    prompts = [prompt_template.format(question=item["problem"]) for item in val_examples]
    outputs = vllm_model.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    rows: list[dict[str, Any]] = []
    totals = {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}
    response_lengths: list[int] = []
    correct_response_lengths: list[int] = []
    incorrect_response_lengths: list[int] = []

    for eval_index, (item, prompt, output) in enumerate(
        zip(val_examples, prompts, outputs, strict=True)
    ):
        response = output.outputs[0].text
        reward = r1_zero_reward_fn(response, item["expected_answer"])
        length = response_token_length(output, response, tokenizer)
        for key in totals:
            totals[key] += float(reward[key])
        response_lengths.append(length)
        if reward["answer_reward"] == 1.0:
            correct_response_lengths.append(length)
        else:
            incorrect_response_lengths.append(length)
        rows.append(
            {
                "eval_index": eval_index,
                "problem": item["problem"],
                "ground_truth": item["expected_answer"],
                "prompt": prompt,
                "response": response,
                "response_length_tokens": length,
                "reward": reward,
            }
        )

    count = max(1, len(rows))
    metrics = {
        "eval/accuracy": totals["answer_reward"] / count,
        "eval/reward": totals["reward"] / count,
        "eval/format_accuracy": totals["format_reward"] / count,
        "eval/num_examples": float(len(rows)),
        "eval/avg_response_length": mean_or_zero(response_lengths),
        "eval/avg_correct_response_length": mean_or_zero(correct_response_lengths),
        "eval/avg_incorrect_response_length": mean_or_zero(
            incorrect_response_lengths
        ),
        "eval/num_correct": float(len(correct_response_lengths)),
        "eval/num_incorrect": float(len(incorrect_response_lengths)),
    }
    if output_path is not None:
        write_jsonl(output_path, rows)

    if args.num_validation_samples_to_log > 0 and rows:
        rng = random.Random(args.validation_sample_seed + eval_step)
        sample_count = min(args.num_validation_samples_to_log, len(rows))
        sample_indices = sorted(rng.sample(range(len(rows)), sample_count))
        samples = [dict(rows[index]) for index in sample_indices]
        entropies = compute_sample_response_entropies(
            policy_model,
            tokenizer,
            device,
            samples,
            args.max_length,
            args.validation_sample_entropy_chunk_size,
        )
        for sample, entropy in zip(samples, entropies, strict=True):
            sample["avg_response_entropy"] = entropy

        if sample_output_path is not None:
            write_jsonl(sample_output_path, samples)
        swanlab_log_validation_samples(
            swanlab_run,
            samples,
            step=eval_step,
            max_chars=args.validation_sample_max_chars,
        )

    return metrics


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 准备数据
    all_train_examples = read_json_or_jsonl(args.train_path)
    selected_examples = choose_examples(all_train_examples, args.dataset_size, args.seed)
    val_examples = read_json_or_jsonl(args.val_path)
    prompt_template = Path(args.prompt_path).read_text()

    # 准备日志输出
    if args.run_name is None:
        train_name = Path(args.train_path).stem
        args.run_name = f"sft-math-{train_name}-{args.dataset_size}-seed{args.seed}"

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # 准备模型和 dataloader
    model, tokenizer, device = load_model_and_tokenizer(args)
    vllm_model = init_vllm(
        model_id=args.model_name_or_path,
        device=args.eval_device,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
    )
    dataset = MathSFTDataset(
        selected_examples,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    run = setup_swanlab(args, selected_size=len(selected_examples))
    metrics_path = output_dir / "metrics.jsonl"
    metrics_file = metrics_path.open("a")

    def record(metrics: dict[str, Any], step: int | None = None) -> None:
        row = {"time": time.time(), **metrics}
        metrics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics_file.flush()
        swanlab_log(run, metrics, step=step)

    record(
        {
            "dataset/total_examples": len(all_train_examples),
            "dataset/selected_examples": len(selected_examples),
            "dataset/val_examples": len(val_examples),
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
        model.train()
        record({"eval_step": 0, **eval_metrics}, step=0)

    global_step = 0
    accumulation_count = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for epoch in range(args.epochs):
        updates_this_epoch = (
            len(dataloader) + args.gradient_accumulation_steps - 1
        ) // args.gradient_accumulation_steps
        progress = tqdm(
            total=updates_this_epoch,
            desc=f"epoch {epoch + 1}/{args.epochs}",
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
            accumulation_metrics["response_tokens"] += float(
                num_response_tokens.detach().cpu()
            )
            accumulation_metrics["response_entropy"] += float(entropy.detach().cpu())
            accumulation_metrics["normalize_constant"] += float(
                normalize_constant.cpu()
            )

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
                "train_step": global_step,
                "train/epoch": epoch + 1,
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

            if args.eval_every_steps and global_step % args.eval_every_steps == 0:
                load_policy_into_vllm_instance(model, vllm_model)
                eval_metrics = evaluate(
                    model,
                    vllm_model,
                    tokenizer,
                    val_examples,
                    prompt_template,
                    args,
                    device,
                    global_step,
                    output_dir / f"eval_step_{global_step}.jsonl"
                    if args.save_eval_outputs
                    else None,
                    output_dir / f"validation_samples_step_{global_step}.jsonl",
                    run,
                )
                model.train()
                record({"eval_step": global_step, **eval_metrics}, step=global_step)
        progress.close()

    load_policy_into_vllm_instance(model, vllm_model)
    eval_metrics = evaluate(
        model,
        vllm_model,
        tokenizer,
        val_examples,
        prompt_template,
        args,
        device,
        global_step,
        output_dir / f"eval_step_{global_step}.jsonl"
        if args.save_eval_outputs
        else None,
        output_dir / f"validation_samples_step_{global_step}.jsonl",
        run,
    )
    model.train()
    record({"eval_step": global_step, **eval_metrics}, step=global_step)

    summary = {
        "run_name": args.run_name,
        "train_path": args.train_path,
        "val_path": args.val_path,
        "prompt_path": args.prompt_path,
        "dataset_size": args.dataset_size,
        "selected_dataset_size": len(selected_examples),
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
    parser = argparse.ArgumentParser(description="Run SFT on MATH reasoning traces.")
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val-path", default=DEFAULT_VAL_PATH)
    parser.add_argument("--prompt-path", default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-dir", default="outputs/sft_math")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset-size", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--eval-device", default="cuda:1")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--normalize-constant",
        type=float,
        default=0.0,
        help=(
            "Use a positive value for fixed normalization. "
            "Use 0 to divide by the average response-token count in each microbatch."
        ),
    )
    parser.add_argument("--max-length", type=int, default=1024)
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

    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--eval-every-steps", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=5000)
    parser.add_argument("--eval-max-new-tokens", type=int, default=1024)
    parser.add_argument("--eval-min-tokens", type=int, default=4)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--vllm-max-model-len", type=int, default=2048)
    parser.add_argument("--num-validation-samples-to-log", type=int, default=16)
    parser.add_argument("--validation-sample-seed", type=int, default=0)
    parser.add_argument("--validation-sample-max-chars", type=int, default=600)
    parser.add_argument("--validation-sample-entropy-chunk-size", type=int, default=8)
    parser.add_argument("--save-eval-outputs", action="store_true")

    parser.add_argument("--log-every-steps", type=int, default=1)
    parser.add_argument("--swanlab-project", default="cs336-a5-sft-math")
    parser.add_argument("--swanlab-log-dir", default="outputs/swanlab")
    parser.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        default="local",
    )
    parser.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    summary = train(parse_args())
    print(json.dumps(summary, indent=2))
