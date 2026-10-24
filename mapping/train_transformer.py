from pathlib import Path
import sys
from torch import nn, optim
import torch
from data.dataset import data_loader
import random
import numpy as np
import argparse
import os
import timm

CUR_DIR = Path(__file__).resolve().parent
SWIN_MODELS_DIR = CUR_DIR / "models" / "Swin-Transformer-main" / "models"
TRANSNEXT_CLASSIFICATION_DIR = CUR_DIR / "models" / "TransNeXt-main" / "classification"
sys.path.insert(0, str(SWIN_MODELS_DIR))
sys.path.insert(0, str(TRANSNEXT_CLASSIFICATION_DIR))
sys.path.append("./models/")

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


def _load_transnext_weights(model, raw):
    if isinstance(raw, dict):
        state = raw
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state = raw["state_dict"]
        elif "model" in raw and isinstance(raw["model"], dict):
            state = raw["model"]
    elif isinstance(raw, nn.Module):
        state = raw.state_dict()
    else:
        state = raw
    model_sd = model.state_dict()
    to_load = {}
    for k, v in state.items():
        kk = k[7:] if k.startswith("module.") else k
        if kk in model_sd and v.shape == model_sd[kk].shape:
            to_load[kk] = v
    model.load_state_dict(to_load, strict=False)


parser = argparse.ArgumentParser(description='Train Transformer')
parser.add_argument('--seed', type=int, help='Seed value', required=False)
parser.add_argument('--dataset', type=str, help='The dataset name', choices=['ChestXRay', 'ISICSkinCancer', 'PathMNIST',
                                                                             'RotatedMNIST'],
                    required=True)
parser.add_argument('--root_dir', type=str, help='The directory to the data images', required=True)
parser.add_argument('--preprocess', type=str, help='The preprocess method', choices=['grayscaled', 'standardized'],
                    required=False, default='grayscaled')
parser.add_argument('--model_type', type=str, help='The model type', choices=['resnet18',
                                                                              'resnet50',
                                                                              'efficientnetv2',
                                                                              'deit',
                                                                              'vit',
                                                                              'convit',
                                                                              'swin',
                                                                              'transnext'], required=False, default='vit')
parser.add_argument('--device', type=int, default=0, help='GPU device id (default: 0)')
parser.add_argument('--save_path', type=str, default='',
                    help='最优权重保存路径（.pth 或目录）。不指定则 models/<dataset>/<name>_<dataset>.pth')
parser.add_argument('--resume', type=str, default='', help='可选：从已有骨干权重继续训练')
args = parser.parse_args()
if args.seed is None:
    seed = random.randint(0, 10000)
else:
    seed = args.seed
dataset_name = args.dataset
root_dir = args.root_dir
model_type = args.model_type
preprocess = args.preprocess
if dataset_name in ['ChestXRay', 'ISICSkinCancer']:
    num_classes = 2
    batch_size = 30
elif dataset_name == 'PathMNIST':
    num_classes = 9
    batch_size = 256
elif dataset_name == 'RotatedMNIST':
    num_classes = 10
    batch_size = 64

print('Training {} on {} with seed {}:'.format(model_type, root_dir, seed))
set_seed(seed)
if model_type == 'resnet18':
    model = torch.load('./models/base/resnet18.pth')
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    save_model_name = 'resnet18'
elif model_type == 'resnet50':
    model = torch.load('./models/base/resnet50.pth')
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    save_model_name = 'resnet50'
elif model_type == 'efficientnetv2':
    model = torch.load('./models/base/efficientnetv2_l.pth')
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    save_model_name = 'efficientnetv2_l'
elif model_type == 'deit':
    model = torch.load('./models/base/deit_base_patch16_224.pth')
    model.head = torch.nn.Linear(model.head.in_features, num_classes)
    save_model_name = 'deit_base_patch16_224'
elif model_type == 'vit':
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = torch.nn.Linear(model.head.in_features, num_classes)
    save_model_name = 'vit_base_patch16_224'
elif model_type == 'convit':
    model = torch.load('./models/base/convit_base.pth')
    model.head = torch.nn.Linear(model.head.in_features, num_classes)
    save_model_name = 'convit_base'
elif model_type == 'swin':
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
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
    save_model_name = 'swin_tiny_patch4_window7_224'
elif model_type == 'transnext':
    model = transnext_tiny(pretrained=False, num_classes=num_classes, img_size=224)
    save_model_name = 'transnext_tiny_224'
else:
    raise NotImplementedError

if args.resume:
    raw = torch.load(args.resume, map_location='cpu')
    if model_type == 'transnext':
        _load_transnext_weights(model, raw)
    elif isinstance(raw, nn.Module):
        model = raw
    elif isinstance(raw, dict):
        model.load_state_dict(raw, strict=False)
    print(f'Resumed from {args.resume}')

loader_, dataset_ = data_loader(root_dir=root_dir, dataset_name=dataset_name, preprocess=preprocess,
                                batch_size=batch_size)
train_loader = loader_['train']
valid_loader = loader_['valid']
train_dataset = dataset_['train']
valid_dataset = dataset_['valid']
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
if model_type in ['resnet18', 'resnet50', 'efficientnetv2', 'deit', 'vit', 'convit', 'swin', 'transnext']:
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    epochs = 200
else:
    raise NotImplementedError

if args.save_path:
    save_path = Path(args.save_path)
    if save_path.suffix != '.pth':
        save_path = save_path / f'{save_model_name}_{dataset_name}.pth'
else:
    save_path = Path(f'./models/{dataset_name}/{save_model_name}_{dataset_name}.pth')
save_path.parent.mkdir(parents=True, exist_ok=True)
print(f'Save path: {save_path}')

# Training loop
best_acc = 0.0
best_train_acc = 0.0
for epoch in range(epochs):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backward + optimize
        loss.backward()
        # optimizer.step()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    print(f'Epoch {epoch} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    train_acc = epoch_acc

    # evaluate on test data
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(valid_dataset)
    epoch_acc = running_corrects.double() / len(valid_dataset)

    print(f'Epoch {epoch} Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_train_acc = train_acc
        torch.save(model, str(save_path))

    if scheduler is not None:
        scheduler.step()

print('Model {} on {} best train Acc: {:.4f} val Acc: {:.4f}'.format(model_type, dataset_name, best_train_acc, best_acc))
