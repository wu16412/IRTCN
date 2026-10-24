# IRTCN

结核病胸片最小可运行代码：冻结 Mapping → 条件扩散模型（CDM），骨干为 TransNeXt / ViT / Swin，可选 NAM-CXR-light WLD-144 纹理 CDM。不含控制实验、evidence chain、inspect_layer 等附加代码。

Minimum runnable code for **IRTCN**: frozen Mapping → Conditional Diffusion Model (CDM) on tuberculosis chest X-rays, with TransNeXt / ViT / Swin backbones and an optional Texture CDM guided by NAM-CXR-light WLD-144 features.

This repository contains **only the main training / testing pipeline**. Probe, control, evidence-chain, and layer-inspection experiments are not included.

## What this repo trains

1. **Backbone** (`mapping/train_transformer.py`) — TransNeXt-Tiny, ViT-B/16, or Swin-T on 224×224 grayscale CXR.
2. **Mapping MLPs** (`mapping/train_mapping.py`) — one frozen MLP per CDM route.
3. **Visual CDMs** (`diffusion/`) — five CARD-style routes (`mlp_idx` 0–4).
4. **Texture CDM** (optional) — route 3 guided by a frozen `TextureClassifier` on NAM WLD-144.

Official test protocol (do not change): `testing.batch_size=70`, Monte-Carlo trials `mc_trials=20` (hard-coded), train seed `4444`.

## Layout

```
mapping/          backbone, Mapping MLPs, TextureClassifier
diffusion/        CDM train / test, configs, launchers
NAM-CXR-light/    NAMLab Linux sources (WLD-144 extractor)
docs/protocol.md  locked hyperparameters
data/             put ImageFolder splits here (not shipped)
```

## Data

Not included. Use ImageFolder:

```
data/TB_Chest_Radiography_Database/{training,validation,testing}/{Normal,Tuberculosis}/
```

HER-Seg dual-lung crops (Texture / crop arm):

```
data/TB_Chest_Radiography_Database_HERSEG_CROP/{training,validation,testing}/{Normal,Tuberculosis}/
```

Split a flat `Normal/` + `Tuberculosis/` folder with `python split.py --root_dir /path/to/dataset`.

## Environment

Python 3.8+ with CUDA PyTorch. Install:

```bash
pip install -r requirements.txt
```

Edit paths in `env.sh`, then:

```bash
chmod +x env.sh diffusion/training_scripts/*.sh
source env.sh
```

## Phase 1 — visual IRA (full image)

```bash
source env.sh
# TransNeXt full (backbone → mapping → 5 CDMs)
bash $IRTCN/diffusion/training_scripts/run_transnext_full.sh
# or ViT
bash $IRTCN/diffusion/training_scripts/run_vit_full.sh

# write trained_diffusion_ckpt_path into the yml, then test
STAGE=integrate bash $IRTCN/diffusion/training_scripts/run_transnext_full.sh
BACKBONE=transnext DEVICE=0 TEST_SEED=42 \
  bash $IRTCN/diffusion/training_scripts/test_phase1_backbone.sh
```

Crop arm (TransNeXt + HER-Seg crop, separate weight dirs):

```bash
bash $IRTCN/diffusion/training_scripts/run_transnext_crop.sh
```

GPU pools: `BACKBONE_DEVICE`, `MAPPING_GPUS`, `DIFFUSION_GPUS`. Each CDM occupies ~10 GB.

## Texture CDM (optional sixth route)

1. Build NAM-CXR-light (`NAM-CXR-light/README.md`) and export WLD-144 to  
   `mapping/features/nam_cxr_light_wld144/{training,validation,testing}.npz`  
   (keys: `lbp`, `labels`, `paths`).
2. Train the frozen texture MLP:

```bash
$PYTHON mapping/train_texture.py \
  --feature-dir $IRTCN/mapping/features/nam_cxr_light_wld144 \
  --output $IRTCN/mapping/models/texture/seed_4444/best.pth \
  --seed 4444
```

3. Train Texture CDM (`mlp_idx=3`) on crop images:

```bash
source env.sh
bash $IRTCN/diffusion/training_scripts/train_texture_cdm.sh
```

`NAM_FEATURE_DIR` and `TEXTURE_CKPT` override config paths.

## Third-party code

| Path | Source | License |
|------|--------|---------|
| `mapping/models/TransNeXt-main/` | [TransNeXt](https://github.com/DaiShiResearch/TransNeXt) | Apache-2.0 |
| `mapping/models/Swin-Transformer-main/` | [Swin Transformer](https://github.com/microsoft/Swin-Transformer) | MIT |
| `NAM-CXR-light/` | NAMLab (Zheng et al., IEEE TIP 2021) | see its LICENSE |
| CDM / Mapping | CARD / IRA lineage | MIT (`LICENSE`) |
