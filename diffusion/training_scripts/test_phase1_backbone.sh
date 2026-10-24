#!/bin/bash
# Phase1 official test (testing.batch_size=70, mc_trials=20)
# mc_trials=20 写死在 classification_train_separately.test_atk
# 用法：
#   source $IRTCN/env.sh
#   BACKBONE=vit DEVICE=0 TEST_SEED=42 bash test_phase1_backbone.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IRTCN="${IRTCN:-$ROOT}"
if [ -f "$IRTCN/env.sh" ]; then
  # shellcheck source=/dev/null
  source "$IRTCN/env.sh"
fi
PYTHON="${PYTHON:-python}"
BACKBONE="${BACKBONE:-vit}"
DEVICE_ID="${DEVICE:-0}"
PREPROCESS="${PREPROCESS:-grayscaled}"
N_THREADS="${N_THREADS:-8}"

case "$BACKBONE" in
  transnext)
    CONFIG="${CONFIG:-chest_x_ray_transnext.yml}"
    EXP_DIR="${EXP_DIR:-$IRTCN/diffusion/results_eval_transnext_chest_x_ray}"
    TASK="${TASK:-chest_x_ray_transnext}"
    DOC="${DOC:-${TASK}_clean}"
    ;;
  vit)
    CONFIG="${CONFIG:-chest_x_ray_vit.yml}"
    EXP_DIR="${EXP_DIR:-$IRTCN/diffusion/results_eval_vit_chest_x_ray}"
    TASK="${TASK:-chest_x_ray_vit}"
    DOC="${DOC:-${TASK}_clean}"
    ;;
  swin)
    CONFIG="${CONFIG:-chest_x_ray.yml}"
    EXP_DIR="${EXP_DIR:-$IRTCN/diffusion/results_eval_swin_chest_x_ray}"
    TASK="${TASK:-chest_x_ray}"
    DOC="${DOC:-${TASK}_clean}"
    ;;
  *)
    echo "Unknown BACKBONE=$BACKBONE"
    exit 1
    ;;
esac

# 允许绝对路径 CONFIG
if [[ "$CONFIG" != /* ]]; then
  CONFIG_PATH="configs/${CONFIG}"
else
  CONFIG_PATH="$CONFIG"
fi

SEED_ARGS=()
if [ -n "${TEST_SEED:-}" ]; then
  SEED_ARGS=(--seed "$TEST_SEED")
  echo "Using explicit TEST_SEED=$TEST_SEED"
else
  echo "No TEST_SEED set; main.py will pick random seed in [0, 10000]"
fi

cd "$IRTCN/diffusion"
env -u PYTHONPATH -u LD_LIBRARY_PATH "$PYTHON" main.py --test \
  --config "$CONFIG_PATH" \
  --exp "${EXP_DIR}/${TASK}" \
  --doc "$DOC" \
  --n_splits 1 \
  --split 0 \
  "${SEED_ARGS[@]}" \
  --device "$DEVICE_ID" \
  --thread "$N_THREADS" \
  --loss card_onehot_conditional \
  --noise_perturbation 0 \
  --low_resolution 0 \
  --brightness 0 \
  --contrast 1 \
  --crop 0 \
  --attack_name None \
  --preprocess "$PREPROCESS" \
  --ni

echo "Done. Results under ${EXP_DIR}/${TASK}/logs/${DOC}/"
