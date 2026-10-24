#!/bin/bash
# ViT full-image pipeline (backbone → mapping → diffusion → integrate)
# Usage:
#   source $IRTCN/env.sh
#   BACKBONE_DEVICE=0 MAPPING_GPUS="0 1 2 3" DIFFUSION_GPUS="0 1 2 3" \
#     bash $IRTCN/diffusion/training_scripts/run_vit_full.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IRTCN="${IRTCN:-$ROOT}"
# shellcheck disable=SC1091
if [ -f "$IRTCN/env.sh" ]; then
  # 不重复 echo 时可静默；env 脚本本身会打印
  # shellcheck source=/dev/null
  source "$IRTCN/env.sh"
fi

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-$IRTCN/data/TB_Chest_Radiography_Database}"
SEED="${SEED:-4444}"
PREPROCESS="${PREPROCESS:-grayscaled}"
STAGE="${STAGE:-all}"   # all | backbone | mapping | diffusion | integrate | test
BACKBONE_DEVICE="${BACKBONE_DEVICE:-0}"
MAPPING_GPUS="${MAPPING_GPUS:-0 1 2 3}"
DIFFUSION_GPUS="${DIFFUSION_GPUS:-0 1 2 3}"
# 可选：只跑指定 Diffusion 路，如 DIFFUSION_MLPS=2 或 "0 2"；空=五路全开
DIFFUSION_MLPS="${DIFFUSION_MLPS:-}"
RESUME="${RESUME:-1}"
DEVICE="${DEVICE:-0}"
TEST_SEED="${TEST_SEED:-42}"

echo "===== IRTCN ViT full ====="
echo "STAGE=$STAGE IRTCN=$IRTCN RESUME=$RESUME DIFFUSION_MLPS=${DIFFUSION_MLPS:-all}"
test -x "$PYTHON" || test -f "$PYTHON" || { echo "PYTHON missing: $PYTHON"; exit 1; }
test -d "$DATA_DIR" || { echo "DATA_DIR missing: $DATA_DIR"; exit 1; }

run_train_stages() {
  local st="$1"
  IRTCN="$IRTCN" DATA_DIR="$DATA_DIR" PYTHON="$PYTHON" SEED="$SEED" PREPROCESS="$PREPROCESS" \
    BACKBONE=vit STAGE="$st" RESUME="$RESUME" \
    BACKBONE_DEVICE="$BACKBONE_DEVICE" MAPPING_GPUS="$MAPPING_GPUS" DIFFUSION_GPUS="$DIFFUSION_GPUS" \
    DIFFUSION_MLPS="$DIFFUSION_MLPS" \
    bash "$SCRIPT_DIR/train_phase1_backbone.sh"
}

integrate_yml() {
  export IRTCN DATA_DIR PYTHON
  env -u PYTHONPATH -u LD_LIBRARY_PATH "$PYTHON" - <<'PY'
import glob, os, re, yaml
irtn = os.environ["IRTCN"]
data_dir = os.environ.get(
    "DATA_DIR",
    os.path.join(irtn, "data", "TB_Chest_Radiography_Database"),
)
cfg_path = f"{irtn}/diffusion/configs/chest_x_ray_vit.yml"
cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
aux = f"{irtn}/mapping/models/ChestXRay_vit"
cfg["data"]["dataroot"] = data_dir
cfg["diffusion"]["trained_aux_cls_ckpt_path"] = aux

def acc(p):
    m = re.search(r"_acc([0-9.]+)\.pth$", os.path.basename(p))
    return float(m.group(1)) if m else -1.0

paths = []
base = (
    f"{irtn}/diffusion/results_eval_vit_chest_x_ray/chest_x_ray_vit/"
    "card_onehot_conditional_results/1000steps/nn"
)
for i in range(5):
    pat = (
        f"{base}/run_{i}/f_phi_prior_cat_f_phi/f_phi_supervised/"
        f"logs/chest_x_ray_vit/split_0/diffu{i}_ckpt_best_*.pth"
    )
    ms = glob.glob(pat)
    if not ms:
        raise SystemExit(f"missing: {pat}")
    best = max(ms, key=acc)
    paths.append(best)
    print(f"mlp{i}: {os.path.basename(best)}")
cfg["diffusion"]["trained_diffusion_ckpt_path"] = [paths]
yaml.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
print("wrote", cfg_path)
PY
}

run_test() {
  IRTCN="$IRTCN" PYTHON="$PYTHON" BACKBONE=vit DEVICE="$DEVICE" TEST_SEED="$TEST_SEED" \
    PREPROCESS="$PREPROCESS" \
    bash "$SCRIPT_DIR/test_phase1_backbone.sh"
}

case "$STAGE" in
  all)
    run_train_stages all
    integrate_yml
    echo "Train+integrate done. Run STAGE=test when ready."
    ;;
  backbone|mapping|diffusion)
    run_train_stages "$STAGE"
    ;;
  integrate)
    integrate_yml
    ;;
  test)
    run_test
    ;;
  *)
    echo "Unknown STAGE=$STAGE (all|backbone|mapping|diffusion|integrate|test)"
    exit 1
    ;;
esac
