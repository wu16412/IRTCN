# IRTCN protocol (locked)

Do not change these values for official runs.

## Data

| Item | Value |
|------|-------|
| dataset | ChestXRay |
| preprocess | grayscaled |
| num_classes | 2 |
| input | 224×224×3 |
| train seed | **4444** |
| mc_trials | **20** (hard-coded in `classification_train_separately.py`) |
| early stop | none for backbone / mapping (full epochs) |

## Stage 1 backbone (`mapping/train_transformer.py`)

| Item | Value |
|------|-------|
| batch_size | 30 |
| optimizer | AdamW |
| lr | 1e-4 |
| weight_decay | 0.1 |
| scheduler | StepLR(step=10, gamma=0.5) |
| epochs | **200** |
| criterion | CrossEntropyLoss |

Weights: `transnext_tiny_224_ChestXRay.pth`, `vit_base_patch16_224_ChestXRay.pth`, `swin_tiny_patch4_window7_224_ChestXRay.pth`.

## Stage 2 Mapping (`mapping/train_mapping.py`)

| Item | Value |
|------|-------|
| batch_size | 30 |
| optimizer | Adam |
| lr | 0.001 |
| scheduler | StepLR(step=20, gamma=0.5) |
| epochs | **301** |
| mn_idx | ViT 0–4; Swin/TransNeXt **0–3** |

Saves `MLPs/block_{mn_idx}.pth`.

## Stage 3 Diffusion

| Item | Value |
|------|-------|
| loss | card_onehot_conditional |
| timesteps | 1000 |
| beta | linear 0.0001 → 0.02 |
| train batch_size | **4** |
| n_epochs | **1000** |
| warmup_epochs | 40 |
| ema | True, 0.9999 |
| testing.batch_size | **70** |
| testing.n_samples | 100 |
| visual routes | mlp_idx **0–4** |

## Texture CDM

Frozen `TextureClassifier` (WLD-144 → MLP 256-128) as guide at `mlp_idx=3`, `fusion_guide.type=texture_only`. Train the MLP with `mapping/train_texture.py` (val balanced accuracy, NLL tie-break).
