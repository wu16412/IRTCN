#!/bin/bash
# TransNeXt + HER-Seg crop pipeline (separate weight dirs from full)
# Usage:
#   source $IRTCN/env.sh
#   BACKBONE_DEVICE=0 MAPPING_GPUS="1 2 3 4" DIFFUSION_GPUS="1 2 3 4" \
#     bash $IRTCN/diffusion/training_scripts/run_transnext_crop.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IRTCN="${IRTCN:-$ROOT}"
# shellcheck disable=SC1091
if [ -f "$IRTCN/env.sh" ]; then
  # shellcheck source=/dev/null
  source "$IRTCN/env.sh"
fi

PYTHON="${PYTHON:-python}"
CROP_DIR="${CROP_DIR:-$IRTCN/data/TB_Chest_Radiography_Database_HERSEG_CROP}"
# Force crop even if env.sh set DATA_DIR to the full-image folder.
DATA_DIR="$CROP_DIR"
SEED="${SEED:-4444}"
PREPROCESS="${PREPROCESS:-grayscaled}"
STAGE="${STAGE:-all}"   # all | backbone | mapping | diffusion | integrate | test
BACKBONE_DEVICE="${BACKBONE_DEVICE:-0}"
MAPPING_GPUS="${MAPPING_GPUS:-0 1 2 3}"
DIFFUSION_GPUS="${DIFFUSION_GPUS:-0 1 2 3}"
DIFFUSION_MLPS="${DIFFUSION_MLPS:-}"
MAPPING_BATCH="${MAPPING_BATCH:-6}"
MAPPING_ACCUM="${MAPPING_ACCUM:-5}"
DEVICE="${DEVICE:-0}"
TEST_SEED="${TEST_SEED:-42}"
RESUME="${RESUME:-0}"

echo "===== IRTCN TransNeXt HERSEG crop ====="
echo "STAGE=$STAGE IRTCN=$IRTCN DATA=$DATA_DIR RESUME=$RESUME"
echo "script=$SCRIPT_DIR/run_transnext_crop.sh"
test -f "$SCRIPT_DIR/train_phase1_backbone.sh" || { echo "missing train_phase1_backbone.sh"; exit 1; }
test -f "$IRTCN/diffusion/configs/chest_x_ray_transnext_herseg_crop.yml" || {
  echo "missing configs/chest_x_ray_transnext_herseg_crop.yml"; exit 1;
}
test -x "$PYTHON" || test -f "$PYTHON" || { echo "PYTHON missing: $PYTHON"; exit 1; }
test -d "$DATA_DIR" || { echo "CROP DATA_DIR missing: $DATA_DIR"; exit 1; }
grep -q 'PHASE1_DATA' "$SCRIPT_DIR/train_phase1_backbone.sh" || {
  echo "train_phase1_backbone.sh 无 PHASE1_DATA=crop 支持 — 请同步最新脚本"; exit 1;
}

run_train_stages() {
  local st="$1"
  IRTCN="$IRTCN" DATA_DIR="$DATA_DIR" CROP_DIR="$CROP_DIR" PYTHON="$PYTHON" \
    SEED="$SEED" PREPROCESS="$PREPROCESS" \
    BACKBONE=transnext PHASE1_DATA=crop STAGE="$st" RESUME="$RESUME" \
    BACKBONE_DEVICE="$BACKBONE_DEVICE" MAPPING_GPUS="$MAPPING_GPUS" \
    DIFFUSION_GPUS="$DIFFUSION_GPUS" DIFFUSION_MLPS="$DIFFUSION_MLPS" \
    MAPPING_BATCH="$MAPPING_BATCH" MAPPING_ACCUM="$MAPPING_ACCUM" \
    bash "$SCRIPT_DIR/train_phase1_backbone.sh"
}

integrate_yml() {
  export IRTCN DATA_DIR CROP_DIR PYTHON
  env -u PYTHONPATH -u LD_LIBRARY_PATH "$PYTHON" - <<'PY'
import glob, os, re, yaml
irtn = os.environ["IRTCN"]
data_dir = os.environ.get("DATA_DIR") or os.environ.get(
    "CROP_DIR",
    os.path.join(irtn, "data", "TB_Chest_Radiography_Database_HERSEG_CROP"),
)
cfg_path = f"{irtn}/diffusion/configs/chest_x_ray_transnext_herseg_crop.yml"
cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
aux = f"{irtn}/mapping/models/ChestXRay_transnext_herseg_crop"
cfg["data"]["dataroot"] = data_dir
cfg["diffusion"]["trained_aux_cls_ckpt_path"] = aux

def acc(p):
    m = re.search(r"_acc([0-9.]+)\.pth$", os.path.basename(p))
    return float(m.group(1)) if m else -1.0

paths = []
base = (
    f"{irtn}/diffusion/results_eval_transnext_herseg_crop/"
    "chest_x_ray_transnext_herseg_crop/"
    "card_onehot_conditional_results/1000steps/nn"
)
for i in range(5):
    pat = (
        f"{base}/run_{i}/f_phi_prior_cat_f_phi/f_phi_supervised/"
        f"logs/chest_x_ray_transnext_herseg_crop/split_0/diffu{i}_ckpt_best_*.pth"
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
  IRTCN="$IRTCN" PYTHON="$PYTHON" BACKBONE=transnext DEVICE="$DEVICE" TEST_SEED="$TEST_SEED" \
    PREPROCESS="$PREPROCESS" \
    CONFIG="$IRTCN/diffusion/configs/chest_x_ray_transnext_herseg_crop.yml" \
    EXP_DIR="$IRTCN/diffusion/results_eval_transnext_herseg_crop" \
    TASK=chest_x_ray_transnext_herseg_crop \
    DOC=chest_x_ray_transnext_herseg_crop_clean \
    DATA_DIR="$DATA_DIR" \
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
