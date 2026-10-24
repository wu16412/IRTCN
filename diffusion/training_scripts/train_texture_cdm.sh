#!/usr/bin/env bash
# Texture-only CDM (mlp_idx=3, NAM-CXR-light WLD144 guide)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IRTCN="${IRTCN:-$ROOT}"
if [ -f "$IRTCN/env.sh" ]; then
  # shellcheck source=/dev/null
  source "$IRTCN/env.sh"
fi

PYTHON="${PYTHON:-python}"
CROP_DIR="${CROP_DIR:-$IRTCN/data/TB_Chest_Radiography_Database_HERSEG_CROP}"
SEED="${SEED:-4444}"
TEXTURE_DEVICE="${TEXTURE_DEVICE:-0}"
CONFIG="${CONFIG:-configs/chest_x_ray_herseg_texture.yml}"
EXP_DIR="${EXP_DIR:-$IRTCN/diffusion/results_herseg_texture_only_mlp3}"
DATAROOT="${DATAROOT:-$CROP_DIR}"
AUX="${AUX:-$IRTCN/mapping/models/ChestXRay_transnext_herseg_crop}"
export NAM_FEATURE_DIR="${NAM_FEATURE_DIR:-$IRTCN/mapping/features/nam_cxr_light_wld144}"
export TEXTURE_CKPT="${TEXTURE_CKPT:-$IRTCN/mapping/models/texture/seed_${SEED}/best.pth}"

cd "$IRTCN/diffusion"

test -f "$TEXTURE_CKPT" || { echo "missing TEXTURE_CKPT=$TEXTURE_CKPT"; exit 1; }
test -d "$NAM_FEATURE_DIR" || { echo "missing NAM_FEATURE_DIR=$NAM_FEATURE_DIR"; exit 1; }
test -d "$DATAROOT" || { echo "missing DATAROOT=$DATAROOT"; exit 1; }
test -f "$AUX/MLPs/block_0.pth" && test -f "$AUX/MLPs/block_3.pth" || {
  echo "missing crop Mapping under $AUX/MLPs"; exit 1;
}

TMP_CONFIG="$(mktemp --suffix=.yml)"
trap 'rm -f "$TMP_CONFIG"' EXIT

env -u PYTHONPATH -u LD_LIBRARY_PATH "$PYTHON" - "$CONFIG" "$TMP_CONFIG" "$DATAROOT" "$AUX" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
cfg["data"]["dataroot"] = sys.argv[3]
cfg["diffusion"]["trained_aux_cls_ckpt_path"] = sys.argv[4]
yaml.safe_dump(cfg, open(sys.argv[2], "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
PY

mkdir -p "$EXP_DIR"
echo "[texture-only] DATAROOT=$DATAROOT AUX=$AUX"
echo "[texture-only] NAM_FEATURE_DIR=$NAM_FEATURE_DIR TEXTURE_CKPT=$TEXTURE_CKPT"

CUDA_VISIBLE_DEVICES="$TEXTURE_DEVICE" env -u PYTHONPATH -u LD_LIBRARY_PATH \
  NAM_FEATURE_DIR="$NAM_FEATURE_DIR" TEXTURE_CKPT="$TEXTURE_CKPT" \
  "$PYTHON" main.py \
  --mlp_idx 3 --device 0 --thread "${N_THREADS:-8}" \
  --loss card_onehot_conditional --config "$TMP_CONFIG" \
  --dataroot "$DATAROOT" --seed "$SEED" --preprocess grayscaled \
  --exp "$EXP_DIR" --doc texture_only_mlp3 --n_splits 1 --ni
