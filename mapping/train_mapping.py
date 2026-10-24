from pathlib import Path
import sys
from torch import nn, optim
import torch
from data.dataset import data_loader
import random
import numpy as np
import argparse
import os

CUR_DIR = Path(__file__).resolve().parent
SWIN_MODELS_DIR = CUR_DIR / "models" / "Swin-Transformer-main" / "models"
TRANSNEXT_CLASSIFICATION_DIR = CUR_DIR / "models" / "TransNeXt-main" / "classification"
sys.path.insert(0, str(SWIN_MODELS_DIR))
sys.path.insert(0, str(TRANSNEXT_CLASSIFICATION_DIR))
sys.path.append("./models/")

from mlp import Classifier
from swin_transformer import SwinTransformer
from transnext import transnext_tiny


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _transnext_num_classes(dataset):
    if dataset in ['ChestXRay', 'ISICSkinCancer']:
        return 2
    if dataset == 'PathMNIST':
        return 9
    return 10


def _load_transnext_weights(model, raw):
    if isinstance(raw, dict):
        state = raw
        if 'state_dict' in raw and isinstance(raw['state_dict'], dict):
            state = raw['state_dict']
        elif 'model' in raw and isinstance(raw['model'], dict):
            state = raw['model']
    else:
        state = raw.state_dict()
    model_sd = model.state_dict()
    to_load = {}
    for k, v in state.items():
        kk = k[7:] if k.startswith('module.') else k
        if kk not in model_sd:
            continue
        if v.shape != model_sd[kk].shape:
            continue
        to_load[kk] = v
    model.load_state_dict(to_load, strict=False)


def extract_transnext_stage_feature(backbone, x, stage_idx):
    """Run TransNeXt through stages 0..stage_idx; return token features (B, N, C)."""
    B = x.shape[0]
    for i in range(stage_idx + 1):
        patch_embed = getattr(backbone, f"patch_embed{i + 1}")
        block = getattr(backbone, f"block{i + 1}")
        norm = getattr(backbone, f"norm{i + 1}")
        x, H, W = patch_embed(x)
        relative_pos_index = getattr(backbone, f"relative_pos_index{i + 1}")
        relative_coords_table = getattr(backbone, f"relative_coords_table{i + 1}")
        for blk in block:
            x = blk(x, H, W, relative_pos_index, relative_coords_table)
        x = norm(x)
        if i < stage_idx and i != backbone.num_stages - 1:
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    return x


parser = argparse.ArgumentParser(description='Train Mapping Networks for encoder blocks outputs')
parser.add_argument('--seed', type=int, help='Seed value', required=False)
parser.add_argument('--dataset', type=str, help='The dataset name', choices=['ChestXRay', 'ISICSkinCancer', 'PathMNIST',
                                                                             'RotatedMNIST'], required=True)
parser.add_argument('--root_dir', type=str, help='The directory to the data images', required=True)
parser.add_argument('--preprocess', type=str, help='The preprocess method', choices=['grayscaled', 'standardized'],
                    required=False, default='grayscaled')
parser.add_argument('--mn_idx', type=int, help='The index of the Mapping Network', choices=[0, 1, 2, 3, 4], required=True)
parser.add_argument('--backbone_type', type=str, help='Feature backbone for mapping',
                    choices=['vit', 'swin', 'transnext'], default='swin')
parser.add_argument('--device', type=int, default=0, help='GPU device id (default: 0)')
parser.add_argument('--backbone_ckpt', type=str, default='',
                    help='骨干权重路径；默认 models/<dataset>/<backbone>_<dataset>.pth')
parser.add_argument('--mlp_dir', type=str, default='',
                    help='MLP 保存目录；默认 models/<dataset>/MLPs')
parser.add_argument('--batch_size', type=int, default=None,
                    help='物理 batch；ChestXRay 协议目标有效 batch=30')
parser.add_argument('--accum_steps', type=int, default=1,
                    help='梯度累积步数；有效 batch ≈ batch_size * accum_steps（mlp0 OOM 时用 6×5）')
args = parser.parse_args()


if args.seed is None:
    seed = random.randint(0, 10000)
else:
    seed = args.seed
dataset_name = args.dataset
mn_idx = args.mn_idx
root_dir = args.root_dir
preprocess = args.preprocess
backbone_type = args.backbone_type

if backbone_type in ('swin', 'transnext') and mn_idx > 3:
    raise ValueError("Swin / TransNeXt backbone only has 4 stages; please set --mn_idx in [0, 1, 2, 3].")


def default_backbone_ckpt(name, dataset):
    if name == 'vit':
        return f'models/{dataset}/vit_base_patch16_224_{dataset}.pth'
    if name == 'swin':
        return f'models/{dataset}/swin_tiny_patch4_window7_224_{dataset}.pth'
    if name == 'transnext':
        return f'models/{dataset}/transnext_tiny_224_{dataset}.pth'
    raise NotImplementedError


def load_backbone(name, dataset, device, backbone_ckpt=None):
    ckpt_path = backbone_ckpt or default_backbone_ckpt(name, dataset)
    if name == 'vit':
        model = torch.load(ckpt_path, map_location=device)
    elif name == 'swin':
        raw = torch.load(ckpt_path, map_location=device)
        if isinstance(raw, dict):
            model = SwinTransformer(
                img_size=224,
                patch_size=4,
                in_chans=3,
                num_classes=2 if dataset in ['ChestXRay', 'ISICSkinCancer'] else (9 if dataset == 'PathMNIST' else 10),
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                drop_path_rate=0.1,
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                fused_window_process=False
            )
            model.load_state_dict(raw)
        else:
            model = raw
    elif name == 'transnext':
        raw = torch.load(ckpt_path, map_location='cpu')
        nc = _transnext_num_classes(dataset)
        model = transnext_tiny(pretrained=False, num_classes=nc, img_size=224)
        _load_transnext_weights(model, raw)
    else:
        raise NotImplementedError
    return model.to(device)


def extract_backbone_feature(backbone, x, idx, name):
    if name == 'vit':
        x = backbone.patch_embed(x)
        x = backbone.pos_drop(x)
        for i in range(0, idx + 1):
            x = backbone.blocks[i](x)
        return x
    if name == 'swin':
        x = backbone.patch_embed(x)
        if getattr(backbone, "ape", False):
            x = x + backbone.absolute_pos_embed
        x = backbone.pos_drop(x)
        for i in range(0, idx + 1):
            x = backbone.layers[i](x)
        return x
    if name == 'transnext':
        return extract_transnext_stage_feature(backbone, x, idx)
    raise NotImplementedError

backbone_ckpt = args.backbone_ckpt.strip() or None
mlp_dir = args.mlp_dir.strip() or f'./models/{dataset_name}/MLPs'
print('Training Mapping Network {} on {} with backbone {}'.format(mn_idx, root_dir, backbone_type))
print(f'Backbone ckpt: {backbone_ckpt or default_backbone_ckpt(backbone_type, dataset_name)}')
print(f'MLP save dir: {mlp_dir}')
set_seed(seed)
if dataset_name in ['ChestXRay', 'ISICSkinCancer']:
    num_classes = 2
    batch_size = 30  # IRa_swin / nested-diffusion 协议；过小易塌到多数类基线
elif dataset_name == 'PathMNIST':
    num_classes = 9
    batch_size = 256
elif dataset_name == 'RotatedMNIST':
    num_classes = 10
    batch_size = 128
if args.batch_size is not None:
    batch_size = args.batch_size
accum_steps = max(1, int(args.accum_steps))
print(f'Mapping batch_size={batch_size}, accum_steps={accum_steps}, effective_batch≈{batch_size * accum_steps}')
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
backbone = load_backbone(backbone_type, dataset_name, device, backbone_ckpt)
backbone.eval()

with torch.no_grad():
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    feat = extract_backbone_feature(backbone, dummy, mn_idx, backbone_type)
    in_features = feat.flatten(1).shape[1]

model = Classifier(num_classes=num_classes, in_features=in_features).to(device)
print(model)

loader_, dataset_ = data_loader(root_dir=root_dir, dataset_name=dataset_name, preprocess=preprocess, batch_size=batch_size)
train_loader = loader_['train']
valid_loader = loader_['valid']
train_dataset = dataset_['train']
valid_dataset = dataset_['valid']
# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
if dataset_name == 'ChestXRay':
    # Train MLPs_plain
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    epochs = 301
elif dataset_name == 'ISICSkinCancer':
    # Train MLPs_plain
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    epochs = 100
elif dataset_name == 'PathMNIST':
    # Train MLPs_plain
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    epochs = 301
elif dataset_name == 'RotatedMNIST':
    # Train MLPs_plain
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    epochs = 301

# Training loop
best_acc = 0.0
best_train_acc = 0.0
os.makedirs(mlp_dir, exist_ok=True)
for epoch in range(epochs):
    backbone.eval()
    model.train()

    running_loss = 0.0
    running_corrects = 0
    optimizer.zero_grad()
    micro_i = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            x_0 = extract_backbone_feature(backbone, inputs, mn_idx, backbone_type)

        outputs = model(x_0, dataset=dataset_name)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels) / accum_steps
        loss.backward()

        running_loss += loss.item() * accum_steps * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        micro_i += 1

        if micro_i % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    if micro_i % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    print(f'Epoch {epoch} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    train_acc = epoch_acc

    # evaluate on test data
    backbone.eval()
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            x_0 = extract_backbone_feature(backbone, inputs, mn_idx, backbone_type)
            outputs = model(x_0, dataset=dataset_name)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(valid_dataset)
    epoch_acc = running_corrects.double() / len(valid_dataset)

    print(f'Epoch {epoch} Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    if epoch_acc > best_acc:
        best_train_acc = train_acc
        best_acc = epoch_acc
        torch.save(model, os.path.join(mlp_dir, 'block_{}.pth'.format(mn_idx)))

    if scheduler is not None:
        scheduler.step()

print('MLP {} on {} best train Acc: {:.4f} val Acc: {:.4f}'.format(mn_idx, dataset_name, best_train_acc, best_acc))
