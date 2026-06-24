#!/bin/bash
#SBATCH --job-name=VJEPA2-EXP__vlm_probe_eval
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yh6416/VJEPA2-EXP/logs/vlm_probe_eval_%j.out
#SBATCH --error=/scratch/yh6416/VJEPA2-EXP/logs/vlm_probe_eval_%j.err

# Re-evaluates a saved train_vlm_probe_lora.py checkpoint with eval_probe_checkpoint.py
# (adds instance-level top-5 accuracy on top of the original top-3/class-mean-R@5 metrics).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
SHARED_PROJECT_ROOT="${SHARED_PROJECT_ROOT:-/scratch/yh6416/VJEPA2-EXP}"
SIF="${SIF:-/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${SHARED_PROJECT_ROOT}/overlay-15GB-500K.ext3}"
ENV_NAME="${ENV_NAME:-EgoLifeExp}"

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
EVAL_CSV="${EVAL_CSV:?EVAL_CSV is required}"
TRAIN_CSV="${TRAIN_CSV:?TRAIN_CSV is required}"
VAL_CSV="${VAL_CSV:?VAL_CSV is required}"
TEST_CSV="${TEST_CSV:-}"
VIDEO_ROOT="${VIDEO_ROOT:?VIDEO_ROOT is required}"
VERB_CLASSES_CSV="${VERB_CLASSES_CSV:?VERB_CLASSES_CSV is required}"
NOUN_CLASSES_CSV="${NOUN_CLASSES_CSV:?NOUN_CLASSES_CSV is required}"
NUM_FRAMES="${NUM_FRAMES:-32}"
PROBE_NUM_FRAMES="${PROBE_NUM_FRAMES:-8}"
TARGET_FPS="${TARGET_FPS:-8.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

export HF_HOME="${HF_HOME:-/scratch/yh6416/.huggingface}"
export PROJECT_ROOT ENV_NAME CHECKPOINT EVAL_CSV TRAIN_CSV VAL_CSV TEST_CSV VIDEO_ROOT
export VERB_CLASSES_CSV NOUN_CLASSES_CSV NUM_FRAMES PROBE_NUM_FRAMES TARGET_FPS
export BATCH_SIZE NUM_WORKERS TORCH_DTYPE

echo "[probe_eval] checkpoint=${CHECKPOINT} eval_csv=${EVAL_CSV}"

payload=""
read -r -d '' payload <<'BASH' || true
set -euo pipefail
export PYTHONNOUSERSITE=1
source /ext3/env.sh
conda activate /ext3/envs/${ENV_NAME}
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/vjepa2:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

python -m app.hdepic_lora_action_anticipation.eval_probe_checkpoint \
  --checkpoint "${CHECKPOINT}" \
  --eval-csv "${EVAL_CSV}" \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  ${TEST_CSV:+--test-csv "${TEST_CSV}"} \
  --video-root "${VIDEO_ROOT}" \
  --verb-classes-csv "${VERB_CLASSES_CSV}" \
  --noun-classes-csv "${NOUN_CLASSES_CSV}" \
  --num-frames "${NUM_FRAMES}" \
  --probe-num-frames "${PROBE_NUM_FRAMES}" \
  --target-fps "${TARGET_FPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --torch-dtype "${TORCH_DTYPE}"
BASH

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -lc "${payload}"
