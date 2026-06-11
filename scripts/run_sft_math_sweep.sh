#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/assignment5-alignment/model/Qwen2.5-Math-1.5B}"
PROMPT_PATH="${PROMPT_PATH:-/workspace/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt}"
VAL_PATH="${VAL_PATH:-/workspace/assignment5-alignment/data/sft-cs336-assign5-datasets/sft-reason/val.jsonl}"

BASE_TRAIN_PATH="${BASE_TRAIN_PATH:-/workspace/assignment5-alignment/data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b.jsonl}"
FILTERED_TRAIN_PATH="${FILTERED_TRAIN_PATH:-/workspace/assignment5-alignment/data/sft-cs336-assign5-datasets/sft-reason/sft_gpt-oss-120b_filtered.jsonl}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_math}"
SWANLAB_MODE="${SWANLAB_MODE:-online}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-cs336-a5-sft-math}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

SIZES="${SIZES:-128 256 512 1024 full}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-5}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
NORMALIZE_CONSTANT="${NORMALIZE_CONSTANT:-0.0}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
EVAL_SAMPLES="${EVAL_SAMPLES:-5000}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-10}"
EVAL_MIN_TOKENS="${EVAL_MIN_TOKENS:-4}"
NUM_VALIDATION_SAMPLES_TO_LOG="${NUM_VALIDATION_SAMPLES_TO_LOG:-16}"
VALIDATION_SAMPLE_MAX_CHARS="${VALIDATION_SAMPLE_MAX_CHARS:-1024}"
VALIDATION_SAMPLE_ENTROPY_CHUNK_SIZE="${VALIDATION_SAMPLE_ENTROPY_CHUNK_SIZE:-8}"
SEED="${SEED:-42}"

COMMON_ARGS=(
  --model-name-or-path "$MODEL_PATH"
  --val-path "$VAL_PATH"
  --prompt-path "$PROMPT_PATH"
  --output-dir "$OUTPUT_DIR"
  --train-device "$TRAIN_DEVICE"
  --eval-device "$EVAL_DEVICE"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --epochs "$EPOCHS"
  --learning-rate "$LR"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --gradient-accumulation-steps "$GRAD_ACCUM_STEPS"
  --normalize-constant "$NORMALIZE_CONSTANT"
  --max-length "$MAX_LENGTH"
  --eval-samples "$EVAL_SAMPLES"
  --eval-every-steps "$EVAL_EVERY_STEPS"
  --eval-min-tokens "$EVAL_MIN_TOKENS"
  --num-validation-samples-to-log "$NUM_VALIDATION_SAMPLES_TO_LOG"
  --validation-sample-max-chars "$VALIDATION_SAMPLE_MAX_CHARS"
  --validation-sample-entropy-chunk-size "$VALIDATION_SAMPLE_ENTROPY_CHUNK_SIZE"
  --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --vllm-max-model-len "$VLLM_MAX_MODEL_LEN"
  --seed "$SEED"
  --swanlab-mode "$SWANLAB_MODE"
  --swanlab-project "$SWANLAB_PROJECT"
)

for size in $SIZES; do
  uv run python scripts/sft_math_experiment.py \
    "${COMMON_ARGS[@]}" \
    --train-path "$BASE_TRAIN_PATH" \
    --dataset-size "$size" \
    --run-name "sft-math-unfiltered-size-${size}-seed${SEED}"
done

uv run python scripts/sft_math_experiment.py \
  "${COMMON_ARGS[@]}" \
  --train-path "$FILTERED_TRAIN_PATH" \
  --dataset-size full \
  --run-name "sft-math-filtered-full-seed${SEED}"

uv run python scripts/plot_sft_math_curves.py --output-dir "$OUTPUT_DIR"
