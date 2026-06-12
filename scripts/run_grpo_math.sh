#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/assignment5-alignment/model/Qwen2.5-Math-1.5B}"
TRAIN_PATH="${TRAIN_PATH:-/workspace/assignment5-alignment/data/sft-cs336-assign5-datasets/sft-reason/train.jsonl}"
VAL_PATH="${VAL_PATH:-/workspace/assignment5-alignment/data/sft-cs336-assign5-datasets/sft-reason/val.jsonl}"
PROMPT_PATH="${PROMPT_PATH:-/workspace/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo_math}"
RUN_NAME="${RUN_NAME:-}"
SWANLAB_MODE="${SWANLAB_MODE:-online}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-cs336-a5-grpo-math}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

N_GRPO_STEPS="${N_GRPO_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
ADVANTAGE_EPS="${ADVANTAGE_EPS:-1e-6}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-256}"
GROUP_SIZE="${GROUP_SIZE:-8}"
SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-1.0}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-1.0}"
SAMPLING_MIN_TOKENS="${SAMPLING_MIN_TOKENS:-4}"
SAMPLING_MAX_TOKENS="${SAMPLING_MAX_TOKENS:-1024}"
EPOCHS_PER_ROLLOUT_BATCH="${EPOCHS_PER_ROLLOUT_BATCH:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-256}"
LOSS_TYPE="${LOSS_TYPE:-reinforce_with_baseline}"
USE_STD_NORMALIZATION="${USE_STD_NORMALIZATION:-true}"
CLIPRANGE="${CLIPRANGE:-0.2}"

MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-4}"
EVAL_SAMPLES="${EVAL_SAMPLES:-1024}"
EVAL_RANDOM_SAMPLE="${EVAL_RANDOM_SAMPLE:-false}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-$SAMPLING_TEMPERATURE}"
EVAL_TOP_P="${EVAL_TOP_P:-$SAMPLING_TOP_P}"
NUM_VALIDATION_SAMPLES_TO_LOG="${NUM_VALIDATION_SAMPLES_TO_LOG:-16}"
SEED="${SEED:-42}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-$SEED}"

ARGS=(
  --model-name-or-path "$MODEL_PATH"
  --train-path "$TRAIN_PATH"
  --val-path "$VAL_PATH"
  --prompt-path "$PROMPT_PATH"
  --output-dir "$OUTPUT_DIR"
  --train-device "$TRAIN_DEVICE"
  --eval-device "$EVAL_DEVICE"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --n-grpo-steps "$N_GRPO_STEPS"
  --learning-rate "$LEARNING_RATE"
  --advantage-eps "$ADVANTAGE_EPS"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --group-size "$GROUP_SIZE"
  --sampling-temperature "$SAMPLING_TEMPERATURE"
  --sampling-top-p "$SAMPLING_TOP_P"
  --sampling-min-tokens "$SAMPLING_MIN_TOKENS"
  --sampling-max-tokens "$SAMPLING_MAX_TOKENS"
  --epochs-per-rollout-batch "$EPOCHS_PER_ROLLOUT_BATCH"
  --train-batch-size "$TRAIN_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --loss-type "$LOSS_TYPE"
  --cliprange "$CLIPRANGE"
  --max-length "$MAX_LENGTH"
  --max-grad-norm "$MAX_GRAD_NORM"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --eval-at-start
  --eval-every-steps "$EVAL_EVERY_STEPS"
  --eval-samples "$EVAL_SAMPLES"
  --eval-sample-seed "$EVAL_SAMPLE_SEED"
  --eval-temperature "$EVAL_TEMPERATURE"
  --eval-top-p "$EVAL_TOP_P"
  --num-validation-samples-to-log "$NUM_VALIDATION_SAMPLES_TO_LOG"
  --seed "$SEED"
  --swanlab-mode "$SWANLAB_MODE"
  --swanlab-project "$SWANLAB_PROJECT"
)

if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
  ARGS+=(--vllm-max-model-len "$VLLM_MAX_MODEL_LEN")
fi

if [[ -n "$RUN_NAME" ]]; then
  ARGS+=(--run-name "$RUN_NAME")
fi

if [[ "$USE_STD_NORMALIZATION" == "true" ]]; then
  ARGS+=(--use-std-normalization)
else
  ARGS+=(--no-use-std-normalization)
fi

if [[ "$EVAL_RANDOM_SAMPLE" == "true" ]]; then
  ARGS+=(--eval-random-sample)
else
  ARGS+=(--eval-prefix-sample)
fi

uv run python scripts/grpo_math_experiment.py "${ARGS[@]}"
uv run python scripts/plot_grpo_curves.py --output-dir "$OUTPUT_DIR"
