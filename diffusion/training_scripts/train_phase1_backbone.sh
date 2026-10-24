#!/bin/bash
# Phase1 visual IRA — TransNeXt / ViT / Swin
# Hyperparameters follow docs/protocol.md.
#
#   source $IRTCN/env.sh
#   BACKBONE=vit STAGE=all bash $IRTCN/diffusion/training_scripts/train_phase1_backbone.sh
#
# STAGE: backbone | mapping | diffusion | all
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IRTCN="${IRTCN:-$ROOT}"
DATA_DIR="${DATA_DIR:-$IRTCN/data/TB_Chest_Radiography_Database}"
PYTHON="${PYTHON:-python}"
BACKBONE="${BACKBONE:-transnext}"   # transnext | vit | swin
STAGE="${STAGE:-all}"
SEED="${SEED:-4444}"
PREPROCESS="${PREPROCESS:-grayscaled}"
# Phase1 固定 ChestXRay；勿继承 shell 里无关的 DATASET（常被占成数据路径）
if [ -n "${DATASET:-}" ] && [[ ! "$DATASET" =~ ^(ChestXRay|ISICSkinCancer|PathMNIST|RotatedMNIST)$ ]]; then
  echo "WARN: ignore invalid DATASET=$DATASET → ChestXRay"
  DATASET=ChestXRay
fi
# 勿继承环境里其它项目的 DATASET（常被设成路径）；请用 IRA_DATASET 覆盖
DATASET="${IRA_DATASET:-ChestXRay}"

BACKBONE_DEVICE="${BACKBONE_DEVICE:-0}"
MAPPING_GPUS="${MAPPING_GPUS:-0 1 2 3}"
DIFFUSION_GPUS="${DIFFUSION_GPUS:-0 1 2 3}"
MAPPING_BATCH="${MAPPING_BATCH:-}"   # 空=用 train_mapping 默认 30；OOM 时可设 6
MAPPING_ACCUM="${MAPPING_ACCUM:-1}"  # 与 BATCH 搭配：6×5=有效 30
MAPPING_BATCH_0="${MAPPING_BATCH_0:-}"   # 仅 block0 覆盖（浅层更易 OOM）
MAPPING_ACCUM_0="${MAPPING_ACCUM_0:-}"
# 不叠卡：最多同时跑 = GPU 数；谁先空闲谁接下一路（含第 5 路）

case "$BACKBONE" in
  transnext)
    MODEL_TYPE=transnext
    BACKBONE_TYPE=transnext
    SAVE_NAME=transnext_tiny_224
    AUX_DIR="$IRTCN/mapping/models/ChestXRay_transnext"
    CONFIG=chest_x_ray_transnext.yml
    EXP_DIR="$IRTCN/diffusion/results_eval_transnext_chest_x_ray"
    TASK=chest_x_ray_transnext
    MN_MAX=3
    ;;
  vit)
    MODEL_TYPE=vit
    BACKBONE_TYPE=vit
    SAVE_NAME=vit_base_patch16_224
    AUX_DIR="$IRTCN/mapping/models/ChestXRay_vit"
    CONFIG=chest_x_ray_vit.yml
    EXP_DIR="$IRTCN/diffusion/results_eval_vit_chest_x_ray"
    TASK=chest_x_ray_vit
    MN_MAX=4
    ;;
  swin)
    MODEL_TYPE=swin
    BACKBONE_TYPE=swin
    SAVE_NAME=swin_tiny_patch4_window7_224
    AUX_DIR="$IRTCN/mapping/models/ChestXRay_swin"
    CONFIG=chest_x_ray.yml
    EXP_DIR="$IRTCN/diffusion/results_eval_swin_chest_x_ray"
    TASK=chest_x_ray
    MN_MAX=3
    ;;
  *)
    echo "Unknown BACKBONE=$BACKBONE (use transnext|vit|swin)"
    exit 1
    ;;
esac

BACKBONE_CKPT="$AUX_DIR/${SAVE_NAME}_${DATASET}.pth"
MLP_DIR="$AUX_DIR/MLPs"
LOG_DIR="$IRTCN/diffusion/logs_phase1_${BACKBONE}"

# PHASE1_DATA=crop：独立目录，不覆盖 full 权重
PHASE1_DATA="${PHASE1_DATA:-full}"   # full | crop
CROP_DIR="${CROP_DIR:-$IRTCN/data/TB_Chest_Radiography_Database_HERSEG_CROP}"
if [ "$PHASE1_DATA" = "crop" ]; then
  if [ "$BACKBONE" != "transnext" ]; then
    echo "PHASE1_DATA=crop 目前仅支持 BACKBONE=transnext" >&2
    exit 1
  fi
  DATA_DIR="$CROP_DIR"
  AUX_DIR="$IRTCN/mapping/models/ChestXRay_transnext_herseg_crop"
  CONFIG=chest_x_ray_transnext_herseg_crop.yml
  EXP_DIR="$IRTCN/diffusion/results_eval_transnext_herseg_crop"
  TASK=chest_x_ray_transnext_herseg_crop
  LOG_DIR="$IRTCN/diffusion/logs_phase1_transnext_crop"
  BACKBONE_CKPT="$AUX_DIR/${SAVE_NAME}_${DATASET}.pth"
  MLP_DIR="$AUX_DIR/MLPs"
fi

mkdir -p "$AUX_DIR" "$MLP_DIR" "$LOG_DIR" "$EXP_DIR"

echo "========================================"
echo "IRTCN Phase1 A1"
echo "BACKBONE : $BACKBONE"
echo "PHASE1   : $PHASE1_DATA"
echo "STAGE    : $STAGE"
echo "SEED     : $SEED"
echo "DATA     : $DATA_DIR"
echo "PYTHON   : $PYTHON"
echo "AUX_DIR  : $AUX_DIR"
echo "========================================"

run_backbone() {
  cd "$IRTCN/mapping"
  # TransNeXt __init__ 里相对位置表写死 device='cuda'（=物理 GPU0）；
  # 必须用 CUDA_VISIBLE_DEVICES 隔离，再 --device 0
  echo "Backbone on physical GPU $BACKBONE_DEVICE (visible as cuda:0)"
  CUDA_VISIBLE_DEVICES="$BACKBONE_DEVICE" env -u PYTHONPATH -u LD_LIBRARY_PATH "$PYTHON" train_transformer.py \
    --dataset "$DATASET" \
    --root_dir "$DATA_DIR" \
    --model_type "$MODEL_TYPE" \
    --preprocess "$PREPROCESS" \
    --seed "$SEED" \
    --device 0 \
    --save_path "$BACKBONE_CKPT" \
    2>&1 | tee "$LOG_DIR/backbone.log"
}

# GPU 池调度：最多同时占满 free 卡；任一进程结束立刻把该卡派给队列下一任务（不叠卡）
run_gpu_pool() {
  # $1 = 标签；其余 = job id 列表。依赖外部定义的 pool_launch_one
  local label="$1"
  shift
  local -a queue=( "$@" )
  local -a free_gpus=( $POOL_GPUS )
  local -A pid2gpu=()
  local -A pid2job=()

  reclaim_finished() {
    local pid gpu job
    for pid in "${!pid2gpu[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || echo "[$label] job ${pid2job[$pid]} (pid=$pid) exited non-zero"
        gpu="${pid2gpu[$pid]}"
        job="${pid2job[$pid]}"
        echo "[$label] job $job finished on GPU $gpu → free"
        free_gpus+=("$gpu")
        unset "pid2gpu[$pid]"
        unset "pid2job[$pid]"
      fi
    done
  }

  while [ "${#queue[@]}" -gt 0 ] || [ "${#pid2gpu[@]}" -gt 0 ]; do
    while [ "${#free_gpus[@]}" -gt 0 ] && [ "${#queue[@]}" -gt 0 ]; do
      local job="${queue[0]}"
      queue=("${queue[@]:1}")
      local gpu="${free_gpus[0]}"
      free_gpus=("${free_gpus[@]:1}")
      POOL_LAUNCH_JOB="$job" POOL_LAUNCH_GPU="$gpu" pool_launch_one
      local pid=$!
      if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        echo "[$label] ERROR: failed to launch job $job on GPU $gpu" >&2
        free_gpus+=("$gpu")
        continue
      fi
      pid2gpu[$pid]="$gpu"
      pid2job[$pid]="$job"
      echo "[$label] launched job $job on GPU $gpu (pid=$pid); queue left=${#queue[@]} running=${#pid2gpu[@]}"
    done
    if [ "${#pid2gpu[@]}" -eq 0 ]; then
      break
    fi
    wait -n || true
    reclaim_finished
  done
  echo "[$label] all jobs done."
}

run_mapping() {
  cd "$IRTCN/mapping"
  POOL_GPUS="$MAPPING_GPUS"

  pool_launch_one() {
    local mn_id="$POOL_LAUNCH_JOB"
    local gpu="$POOL_LAUNCH_GPU"
    local batch_args=()
    local bs="${MAPPING_BATCH}"
    local acc="${MAPPING_ACCUM:-1}"
    if [ "$mn_id" = "0" ]; then
      [ -n "${MAPPING_BATCH_0:-}" ] && bs="$MAPPING_BATCH_0"
      [ -n "${MAPPING_ACCUM_0:-}" ] && acc="$MAPPING_ACCUM_0"
    fi
    if [ -n "$bs" ]; then
      batch_args+=(--batch_size "$bs")
    fi
    if [ -n "$acc" ] && [ "$acc" != "1" ]; then
      batch_args+=(--accum_steps "$acc")
    fi
    echo "Mapping block_$mn_id on GPU $gpu (batch=${bs:-default30} accum=${acc})"
    CUDA_VISIBLE_DEVICES="$gpu" env -u PYTHONPATH -u LD_LIBRARY_PATH -u DATASET "$PYTHON" train_mapping.py \
      --dataset "$DATASET" \
      --root_dir "$DATA_DIR" \
      --backbone_type "$BACKBONE_TYPE" \
      --mn_idx "$mn_id" \
      --preprocess "$PREPROCESS" \
      --seed "$SEED" \
      --device 0 \
      --backbone_ckpt "$BACKBONE_CKPT" \
      --mlp_dir "$MLP_DIR" \
      "${batch_args[@]}" \
      > "$LOG_DIR/mapping_block_${mn_id}.log" 2>&1 &
  }

  local jobs=()
  local mn
  for ((mn=0; mn<=MN_MAX; mn++)); do
    jobs+=("$mn")
  done
  run_gpu_pool "mapping" "${jobs[@]}"
  echo "Mapping done."
}

run_diffusion() {
  cd "$IRTCN/diffusion"
  POOL_GPUS="$DIFFUSION_GPUS"
  local n_routes=5
  # 缓解长时间训练显存碎片导致的 Adam step OOM（不改 batch/协议）
  # 旧版 PyTorch（ira 环境）不支持 expandable_segments，勿写入默认
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

  # 断点：RESUME=1 续训；RESUME=0 从头（默认 0，避免误读损坏 ckpt_last）
  RESUME="${RESUME:-0}"
  local resume_args=(--resume "$RESUME")
  # 兼容旧逻辑：RESUME=1 时也带上 --resume_training
  if [ "$RESUME" = "1" ]; then
    resume_args+=(--resume_training)
  fi
  echo "Diffusion RESUME=$RESUME (1=ckpt_last, 0=from scratch)"

  # 可选只训指定路：DIFFUSION_MLPS="2" 或 "0 2 4"；空=0..4 全开
  local -a jobs=()
  if [ -n "${DIFFUSION_MLPS:-}" ]; then
    # shellcheck disable=SC2206
    jobs=( $DIFFUSION_MLPS )
  else
    local mlp
    for ((mlp=0; mlp<n_routes; mlp++)); do
      jobs+=("$mlp")
    done
  fi
  echo "Diffusion routes: ${jobs[*]}  GPUs: $DIFFUSION_GPUS  RESUME=$RESUME"

  pool_launch_one() {
    local mlp_id="$POOL_LAUNCH_JOB"
    local gpu="$POOL_LAUNCH_GPU"
    local run_name="run_${mlp_id}"
    local model_version_dir="card_onehot_conditional_results/1000steps/nn/${run_name}/f_phi_prior_cat_f_phi/f_phi_supervised"
    echo "Diffusion mlp_idx=$mlp_id on GPU $gpu (ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF RESUME=$RESUME)"
    CUDA_VISIBLE_DEVICES="$gpu" env -u PYTHONPATH -u LD_LIBRARY_PATH "$PYTHON" main.py \
      --mlp_idx "$mlp_id" \
      --device 0 \
      --thread 8 \
      --loss card_onehot_conditional \
      --config "configs/${CONFIG}" \
      --exp "${EXP_DIR}/${TASK}/${model_version_dir}" \
      --doc "$TASK" \
      --n_splits 1 \
      --seed "$SEED" \
      --preprocess "$PREPROCESS" \
      --ni \
      "${resume_args[@]}" \
      > "$LOG_DIR/diffusion_mlp_${mlp_id}.log" 2>&1 &
  }

  run_gpu_pool "diffusion" "${jobs[@]}"
  echo "Diffusion done. Update configs/${CONFIG} trained_diffusion_ckpt_path with best ckpts, then test."
}

case "$STAGE" in
  backbone) run_backbone ;;
  mapping) run_mapping ;;
  diffusion) run_diffusion ;;
  all)
    run_backbone
    run_mapping
    run_diffusion
    ;;
  *)
    echo "Unknown STAGE=$STAGE"
    exit 1
    ;;
esac

echo "IRTCN Phase1 STAGE=$STAGE BACKBONE=$BACKBONE finished: $(date)"
