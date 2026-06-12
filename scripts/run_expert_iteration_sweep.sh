#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/assignment5-alignment/model/Qwen2.5-Math-1.5B}"
TRAIN_PATH="${TRAIN_PATH:-/workspace/assignment5-alignment/data/sft-cs336-assign5-datasets/sft-reason/train.jsonl}"
VAL_PATH="${VAL_PATH:-/workspace/assignment5-alignment/data/sft-cs336-assign5-datasets/sft-reason/val.jsonl}"
PROMPT_PATH="${PROMPT_PATH:-/workspace/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/expert_iteration}"

SWANLAB_MODE="${SWANLAB_MODE:-online}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-cs336-a5-expert-iteration}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"

N_EI_STEPS="${N_EI_STEPS:-5}"
EI_BATCH_SIZES="${EI_BATCH_SIZES:-512 1024 2048}"
ROLLOUT_COUNTS="${ROLLOUT_COUNTS:-4 8}"
SFT_EPOCH_COUNTS="${SFT_EPOCH_COUNTS:-1 2}"

LR="${LR:-1e-5}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
NORMALIZE_CONSTANT="${NORMALIZE_CONSTANT:-0.0}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-1024}"
EVAL_SAMPLES="${EVAL_SAMPLES:-5000}"
EVAL_MIN_TOKENS="${EVAL_MIN_TOKENS:-4}"
NUM_VALIDATION_SAMPLES_TO_LOG="${NUM_VALIDATION_SAMPLES_TO_LOG:-16}"
VALIDATION_SAMPLE_MAX_CHARS="${VALIDATION_SAMPLE_MAX_CHARS:-1024}"
VALIDATION_SAMPLE_ENTROPY_CHUNK_SIZE="${VALIDATION_SAMPLE_ENTROPY_CHUNK_SIZE:-8}"
SEED="${SEED:-42}"

COMMON_ARGS=(
  --model-name-or-path "$MODEL_PATH"
  --train-path "$TRAIN_PATH"
  --val-path "$VAL_PATH"
  --prompt-path "$PROMPT_PATH"
  --output-dir "$OUTPUT_DIR"
  --train-device "$TRAIN_DEVICE"
  --eval-device "$EVAL_DEVICE"
  --n-ei-steps "$N_EI_STEPS"
  --learning-rate "$LR"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --gradient-accumulation-steps "$GRAD_ACCUM_STEPS"
  --normalize-constant "$NORMALIZE_CONSTANT"
  --max-length "$MAX_LENGTH"
  --rollout-temperature "$ROLLOUT_TEMPERATURE"
  --rollout-top-p "$ROLLOUT_TOP_P"
  --rollout-max-new-tokens "$ROLLOUT_MAX_NEW_TOKENS"
  --eval-at-start
  --eval-samples "$EVAL_SAMPLES"
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

for ei_batch_size in $EI_BATCH_SIZES; do
  for rollouts in $ROLLOUT_COUNTS; do
    for sft_epochs in $SFT_EPOCH_COUNTS; do
      run_name="ei-math-db${ei_batch_size}-g${rollouts}-ep${sft_epochs}-seed${SEED}"
      uv run python scripts/expert_iteration_experiment.py \
        "${COMMON_ARGS[@]}" \
        --ei-batch-size "$ei_batch_size" \
        --rollouts-per-question "$rollouts" \
        --sft-epochs-per-ei-step "$sft_epochs" \
        --run-name "$run_name"
    done
  done
done

uv run python scripts/plot_expert_iteration_curves.py --output-dir "$OUTPUT_DIR"
