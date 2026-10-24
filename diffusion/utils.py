import os
import random
import math
import numpy as np
import argparse
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torchvision import transforms
from dataset_helper.chest_x_ray_dataset import data_loader, data_loader_attacks
from torch.nn.functional import interpolate
from torchvision.transforms import Resize, Lambda


def set_random_seed(seed):
    print(f"\n* Set seed {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def sizeof_fmt(num, suffix='B'):
    """
    https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


# print("Check memory usage of different variables:")
# for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
#                          key=lambda x: -x[1])[:10]:
#     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def get_optimizer(config_optim, parameters):
    if config_optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay,
                          betas=(config_optim.beta1, 0.999), amsgrad=config_optim.amsgrad,
                          eps=config_optim.eps)
    elif config_optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay)
    elif config_optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config_optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config_optim.optimizer))


def get_optimizer_and_scheduler(config, parameters, epochs, init_epoch):
    scheduler = None
    optimizer = get_optimizer(config, parameters)
    if hasattr(config, "T_0"):
        T_0 = config.T_0
    else:
        T_0 = epochs // (config.n_restarts + 1)
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=T_0,
                                                                   T_mult=config.T_mult,
                                                                   eta_min=config.eta_min,
                                                                   last_epoch=-1)
        scheduler.last_epoch = init_epoch - 1
    return optimizer, scheduler


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.training.warmup_epochs:
        lr = config.optim.lr * epoch / config.training.warmup_epochs
    else:
        lr = config.optim.min_lr + (config.optim.lr - config.optim.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - config.training.warmup_epochs) / (
                     config.training.n_epochs - config.training.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def _chest_guide_feature_dirs(config):
    """Resolve optional legacy NAM or fusion-region features for every ChestXRay path."""
    from fusion_guide_runtime import (
        FUSION_GUIDE_TYPES,
        NAM_FEATURE_DIR_ENV,
        env_override_path,
        fusion_guide_type_from_config,
    )

    aux_cls = getattr(config.diffusion, 'aux_cls', None)
    guide_type = fusion_guide_type_from_config(config)
    if guide_type not in FUSION_GUIDE_TYPES:
        raise ValueError(f"unsupported fusion_guide.type: {guide_type!r}")
    needs_nam = guide_type == 'texture_only' or (
        guide_type == 'none' and aux_cls is not None and getattr(aux_cls, 'm1d_enabled', False)
    )
    if not needs_nam:
        return None, None
    nam_dir = env_override_path(
        getattr(aux_cls, 'nam_feature_dir', None) if aux_cls is not None else None,
        NAM_FEATURE_DIR_ENV,
    )
    if not nam_dir:
        aux_root = getattr(config.diffusion, 'trained_aux_cls_ckpt_path', None)
        tip = os.path.join(str(aux_root), 'nam_feature_dir.txt') if aux_root else None
        if tip and os.path.isfile(tip):
            with open(tip, 'r', encoding='utf-8') as file:
                nam_dir = file.read().strip()
    if not nam_dir:
        raise ValueError(
            "NAM-based guide enabled but nam_feature_dir missing "
            "(set aux_cls.nam_feature_dir, NAM_FEATURE_DIR, or AUX/nam_feature_dir.txt)."
        )
    return nam_dir, None


def get_texture_only_datasets(args, config, nam_dir):
    """Dedicated texture-only entry: train+valid for development; test only after confirm."""
    from fusion_guide_runtime import (
        LockedSplit,
        assert_no_test_split_access,
        reset_split_access_log,
        texture_splits_from_args,
    )

    reset_split_access_log()
    splits = texture_splits_from_args(args)
    print(f"[texture-only] dataset splits={list(splits)} nam_feature_dir={nam_dir}")
    _, dataset = data_loader(
        root_dir=config.data.dataroot,
        dataset_name=config.data.dataset,
        preprocess=args.preprocess,
        use_precal_mean_std=True,
        nam_feature_dir=nam_dir,
        region_feature_dir=None,
        splits=splits,
        require_nam_splits=True,
    )
    locked_reason = (
        "texture_only development training must not access the testing split; "
        "final test is independent and requires --confirm-final-test"
    )
    train_dataset = dataset.get("train")
    valid_dataset = dataset.get("valid")
    test_dataset = dataset.get("test")
    if "test" not in splits:
        test_dataset = LockedSplit("test", locked_reason)
        assert_no_test_split_access()
    if "train" not in splits:
        train_dataset = LockedSplit("train", "texture_only --test does not load the training split")
    if "valid" not in splits:
        valid_dataset = LockedSplit("valid", "texture_only --test does not load the validation split")
    return None, train_dataset, valid_dataset, test_dataset


def get_dataset(args, config):
    data_object = None
    if config.data.dataset == 'toy':
        tr_x, tr_y = Gaussians().sample(config.data.dataset_size)
        te_x, te_y = Gaussians().sample(config.data.dataset_size)
        train_dataset = torch.utils.data.TensorDataset(tr_x, tr_y)
        test_dataset = torch.utils.data.TensorDataset(te_x, te_y)
    elif config.data.dataset == 'MNIST':
        if config.data.noisy:
            # noisy MNIST as in Contextual Dropout --  no normalization, add standard Gaussian noise
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)),
                AddGaussianNoise(0., 1.)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_dataset = torchvision.datasets.MNIST(root=config.data.dataroot, train=True, download=True,
                                                   transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=config.data.dataroot, train=False, download=True,
                                                  transform=transform)
    elif config.data.dataset == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.FashionMNIST(root=config.data.dataroot, train=True, download=True,
                                                          transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root=config.data.dataroot, train=False, download=True,
                                                         transform=transform)
    elif config.data.dataset == 'RotatedMNIST':
        if config.diffusion.aux_cls.arch == "vit":
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomRotation(45, fill=(0,)),
                                            transforms.Resize((224, 224)),
                                            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = transforms.Compose([transforms.RandomRotation(45, fill=(0,)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST(root=config.data.dataroot, train=True, download=True,
                                                   transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=config.data.dataroot, train=False, download=True,
                                                  transform=transform)

    elif config.data.dataset in ['ChestXRay', 'ISICSkinCancer']:
        nam_dir = None
        region_dir = None
        if config.data.dataset == 'ChestXRay':
            nam_dir, region_dir = _chest_guide_feature_dirs(config)
            from fusion_guide_runtime import fusion_guide_type_from_config
            if fusion_guide_type_from_config(config) == 'texture_only':
                return get_texture_only_datasets(args, config, nam_dir)
            if nam_dir:
                print(f"[fusion-guide] nam_feature_dir={nam_dir}")
            if region_dir:
                print(f"[fusion-guide] region_feature_dir={region_dir}")
        _, _dataset = data_loader(
            root_dir=config.data.dataroot,
            dataset_name=config.data.dataset,
            preprocess=args.preprocess,
            use_precal_mean_std=True,
            nam_feature_dir=nam_dir,
            region_feature_dir=region_dir,
        )
        train_dataset = _dataset['train']
        valid_dataset = _dataset['valid']
        test_dataset = _dataset['test']

        return data_object, train_dataset, valid_dataset, test_dataset

    elif config.data.dataset in ['ChestXRayValidate', 'ISICSkinCancerValidate']:
        if config.data.dataset == 'ChestXRayValidate':
            dataset_name = 'ChestXRay'
        elif config.data.dataset == 'ISICSkinCancerValidate':
            dataset_name = 'ISICSkinCancer'
        nam_dir, region_dir = _chest_guide_feature_dirs(config) if dataset_name == 'ChestXRay' else (None, None)
        _, _dataset = data_loader(
            root_dir=config.data.dataroot, dataset_name=dataset_name,
            preprocess=args.preprocess, use_precal_mean_std=True,
            nam_feature_dir=nam_dir, region_feature_dir=region_dir,
        )
        train_dataset = None
        test_dataset = _dataset['valid']

    elif config.data.dataset in ['ChestXRayAtkFGSM', 'ChestXRayAtkPGD', 'ChestXRayAtkBIM', 'ChestXRayAtkAUTOPGD',
                                 'ChestXRayAtkCW']:
        attack_name = config.data.dataset.split('ChestXRayAtk')[1]
        nam_dir, region_dir = _chest_guide_feature_dirs(config)
        _, _dataset = data_loader_attacks(
            root_dir=config.data.dataroot, attack_name=attack_name,
            nam_feature_dir=nam_dir, region_feature_dir=region_dir,
        )
        train_dataset = None
        test_dataset = _dataset

    elif config.data.dataset in ['ISICSkinCancerAtkFGSM', 'ISICSkinCancerAtkPGD', 'ISICSkinCancerAtkBIM',
                                 'ISICSkinCancerAtkAUTOPGD', 'ISICSkinCancerAtkCW']:
        attack_name = config.data.dataset.split('ISICSkinCancerAtk')[1]
        _, _dataset = data_loader_attacks(root_dir=config.data.dataroot, attack_name=attack_name)
        train_dataset = None
        test_dataset = _dataset

    elif config.data.dataset == "CIFAR10":
        if config.diffusion.aux_cls.arch == "beit":
            data_norm_mean, data_norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=data_norm_mean, std=data_norm_std)
                 ])
        else:
            data_norm_mean, data_norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            # data_norm_mean, data_norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean=data_norm_mean, std=data_norm_std)
                 ])

        train_dataset = torchvision.datasets.CIFAR10(root=config.data.dataroot, train=True,
                                                     download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=config.data.dataroot, train=False,
                                                    download=True, transform=transform)
    elif config.data.dataset == "CIFAR100":
        data_norm_mean, data_norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        # data_norm_mean, data_norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=data_norm_mean, std=data_norm_std)
             ])
        train_dataset = torchvision.datasets.CIFAR100(root=config.data.dataroot, train=True,
                                                      download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root=config.data.dataroot, train=False,
                                                     download=True, transform=transform)
    elif config.data.dataset == "gaussian_mixture":
        data_object = GaussianMixture(n_samples=config.data.dataset_size,
                                      seed=args.seed,
                                      label_min_max=config.data.label_min_max,
                                      dist_dict=vars(config.data.dist_dict),
                                      normalize_x=config.data.normalize_x,
                                      normalize_y=config.data.normalize_y)
        data_object.create_train_test_dataset(train_ratio=config.data.train_ratio)
        train_dataset, test_dataset = data_object.train_dataset, data_object.test_dataset
    else:
        raise NotImplementedError(
            "Options: toy (classification of two Gaussian), MNIST, FashionMNIST, CIFAR10.")
    return data_object, train_dataset, test_dataset


# ------------------------------------------------------------------------------------
# Revised from timm == 0.3.2
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py
# output: the prediction from diffusion model (B x n_classes)
# target: label indices (B)
# ------------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    maxk = min(max(topk), output.size()[1])
    # output = torch.softmax(-(output - 1)**2,  dim=-1)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def compute_extended_classification_metrics(probs, labels, preds=None, num_classes=2, positive_class=1):
    """
  从概率与标签计算 AUC、AP、Sensitivity、Specificity、F1 等（二分类为主）。
  probs: (N, C); labels/preds: (N,)。positive_class 为阳性类索引（ChestXRay 默认 1=Tuberculosis）。
    """
    if hasattr(probs, "detach"):
        probs = probs.detach().cpu().numpy()
    else:
        probs = np.asarray(probs)
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy().astype(np.int64)
    else:
        labels = np.asarray(labels).astype(np.int64)
    if preds is not None:
        if hasattr(preds, "detach"):
            preds = preds.detach().cpu().numpy().astype(np.int64)
        else:
            preds = np.asarray(preds).astype(np.int64)
    else:
        preds = np.argmax(probs, axis=1).astype(np.int64)

    n_classes = int(probs.shape[1]) if probs.ndim == 2 else int(num_classes)
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {"num_samples": int(len(labels)), "positive_class": int(positive_class)}

    if n_classes == 2:
        y_score = probs[:, positive_class]
        y_true_bin = (labels == positive_class).astype(np.int64)
        try:
            metrics["auc"] = float(roc_auc_score(y_true_bin, y_score))
        except ValueError as exc:
            print(
                f"[metrics] auc failed: {exc}; finite={np.isfinite(y_score).all()} "
                f"min={np.nanmin(y_score) if len(y_score) else 'na'} "
                f"max={np.nanmax(y_score) if len(y_score) else 'na'} "
                f"n_pos={int(y_true_bin.sum())} n={len(y_true_bin)}"
            )
            metrics["auc"] = float("nan")
        try:
            metrics["ap"] = float(average_precision_score(y_true_bin, y_score))
        except ValueError as exc:
            print(f"[metrics] ap failed: {exc}")
            metrics["ap"] = float("nan")

        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = (int(x) for x in cm.ravel())
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        metrics["f1"] = float(
            f1_score(labels, preds, pos_label=positive_class, zero_division=0)
        )
        metrics["precision"] = float(
            precision_score(labels, preds, pos_label=positive_class, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(labels, preds, pos_label=positive_class, zero_division=0)
        )
        per_class_recall = recall_score(
            labels, preds, average=None, zero_division=0, labels=[0, 1]
        )
        metrics["per_class_recall"] = [float(x) for x in per_class_recall]
        metrics["ar"] = float(np.mean(per_class_recall))
        n = max(len(labels), 1)
        metrics["confusion_matrix"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        metrics["confusion_matrix_ratio"] = {
            "tn": tn / n,
            "fp": fp / n,
            "fn": fn / n,
            "tp": tp / n,
        }
    else:
        try:
            metrics["auc"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["auc"] = float("nan")
        y_true_bin = (labels == positive_class).astype(np.int64)
        try:
            metrics["ap"] = float(
                average_precision_score(y_true_bin, probs[:, positive_class])
            )
        except ValueError:
            metrics["ap"] = float("nan")
        metrics["f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
        metrics["precision"] = float(
            precision_score(labels, preds, average="macro", zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(labels, preds, average="macro", zero_division=0)
        )
        metrics["sensitivity"] = metrics["recall"]
        per_class_recall = recall_score(
            labels, preds, average=None, zero_division=0, labels=list(range(n_classes))
        )
        cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
        spec_list = []
        for c in range(n_classes):
            tp_c = cm[c, c]
            fn_c = cm[c, :].sum() - tp_c
            fp_c = cm[:, c].sum() - tp_c
            tn_c = cm.sum() - tp_c - fn_c - fp_c
            denom = tn_c + fp_c
            spec_list.append(float(tn_c / denom) if denom > 0 else float("nan"))
        metrics["specificity"] = float(np.nanmean(spec_list))
        metrics["per_class_recall"] = [float(x) for x in per_class_recall]
        metrics["per_class_specificity"] = spec_list
        metrics["ar"] = float(np.mean(per_class_recall))

    return metrics


CORE_CLASSIFICATION_METRIC_KEYS = (
    "sensitivity",
    "specificity",
    "auc",
    "ap",
    "ar",
)

CORE_CLASSIFICATION_METRIC_LABELS_ZH = {
    "sensitivity": "灵敏度 Sensitivity  TP/(TP+FN)",
    "specificity": "特异度 Specificity  TN/(TN+FP)",
    "auc": "AUC  ROC 曲线下面积",
    "ap": "AP  平均精度 (P-R AUC)",
    "ar": "AR  平均召回 (各类 Recall 均值)",
}


def extract_core_classification_metrics(metrics: dict) -> dict:
    """提取五项核心分类指标。"""
    return {k: metrics.get(k) for k in CORE_CLASSIFICATION_METRIC_KEYS if k in metrics}


def format_classification_metrics(metrics: dict, as_percent: bool = False) -> str:
    """格式化打印五项核心分类指标。"""
    lines = ["========== 分类指标 =========="]
    for key in CORE_CLASSIFICATION_METRIC_KEYS:
        if key not in metrics:
            continue
        value = metrics[key]
        label = CORE_CLASSIFICATION_METRIC_LABELS_ZH.get(key, key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            lines.append(f"  {label}: N/A")
        elif as_percent:
            lines.append(f"  {label}: {float(value) * 100:.2f}%")
        else:
            lines.append(f"  {label}: {float(value):.4f}")
    return "\n".join(lines)


def cast_label_to_one_hot_and_prototype(y_labels_batch, config, return_prototype=True):
    """
    y_labels_batch: a vector of length batch_size.
    """
    y_one_hot_batch = nn.functional.one_hot(y_labels_batch, num_classes=config.data.num_classes).float()
    if return_prototype:
        label_min, label_max = config.data.label_min_max
        y_logits_batch = torch.logit(nn.functional.normalize(
            torch.clip(y_one_hot_batch, min=label_min, max=label_max), p=1.0, dim=1))
        return y_one_hot_batch, y_logits_batch
    else:
        return y_one_hot_batch


def apply_attack(attack_func, images_in, labels_in, attack_name):
    # Make a copy of the images tensor
    images = images_in.clone()
    # Make a copy of the labels tensor
    labels = labels_in.clone()
    if attack_name != 'AUTOPGD':
        adv_img, success = attack_func.generate_attack(images, labels=labels)
    else:
        adv_img = attack_func.run_standard_evaluation(images, labels, bs=labels.shape[0])
    # print(f'{sum(success)} successful attacks out of {len(success)}')

    return adv_img


def add_noise(images_in, noise_std):
    # Copy the images to avoid changing the originals
    images = images_in.clone()

    # Add the noise to the images
    images = images + torch.randn_like(images) * noise_std

    return images


class RandomResizedCrop:
    def __init__(self, size, scale):
        self.size = size
        self.scale = scale
        self.resize = Resize(size)

    def __call__(self, img):
        # Calculate the size of the crop
        crop_size = int(self.size[1] * self.scale)  # assuming height = width

        # Randomly select the top left corner of the square
        left = torch.randint(0, self.size[1] - crop_size + 1, (1,)).item()
        top = torch.randint(0, self.size[1] - crop_size + 1, (1,)).item()

        # Crop the image
        img = img[:, top: top + crop_size, left: left + crop_size]

        # Resize the image back to its original size
        img = self.resize(img)

        return img


def random_crop_and_resize(images_in, k):
    # Copy the images to avoid changing the originals
    images = images_in.clone()
    # Create the transform
    transform = Lambda(lambda x: RandomResizedCrop(x.shape[1:], (1 - k))(x))

    # Apply the transform to each image in the batch
    return torch.stack([transform(img) for img in images])


def random_cover_new(images_in, params):
    k = params[0]
    n = params[1]
    # Copy the images to avoid changing the originals
    images = images_in.clone()
    # Calculate the side of the square that needs to be covered
    cover_side = int((k * images.shape[-1] * images.shape[-2]) ** 0.5)

    # Calculate the range of possible top left corners for the square
    top_range = images.shape[-2] - cover_side
    left_range = images.shape[-1] - cover_side

    for image in images:
        covered_regions = []
        for _ in range(n):
            while True:
                # Randomly select the top left corner of the square
                top = random.randint(0, top_range)
                left = random.randint(0, left_range)
                new_region = (top, left, top + cover_side, left + cover_side)

                # Check for overlap with existing regions
                if any((max(r[0], new_region[0]) < min(r[2], new_region[2]) and
                        max(r[1], new_region[1]) < min(r[3], new_region[3])) for r in covered_regions):
                    continue  # Overlap detected, try again

                # No overlap, add the new region to the list
                covered_regions.append(new_region)
                break

            # Set the pixels in the square to black
            image[:, top:top + cover_side, left:left + cover_side] = 0

    return images


def random_cover(images_in, k):
    # Copy the images to avoid changing the originals
    images = images_in.clone()
    # Calculate the side of the square that needs to be covered
    cover_side = int((k * images.shape[-1] * images.shape[-2]) ** 0.5)

    # Calculate the range of possible top left corners for the square
    top_range = images.shape[-2] - cover_side
    left_range = images.shape[-1] - cover_side

    for image in images:
        # Randomly select the top left corner of the square
        top = random.randint(0, top_range)
        left = random.randint(0, left_range)

        # Set the pixels in the square to black
        image[:, top:top + cover_side, left:left + cover_side] = 0

    return images


def down_up_sample(images_in, k):
    # Copy the images to avoid changing the originals
    images = images_in.clone()
    # Get the original size of the images
    original_size = images.shape[-2:]

    # Calculate the new size after downsampling
    down_size = (original_size[0] // k, original_size[1] // k)

    # Downsample the images
    downsampled_images = interpolate(images, size=down_size, mode='bilinear', align_corners=False)

    # Upsample the images back to the original size
    upsampled_images = interpolate(downsampled_images, size=original_size, mode='bilinear', align_corners=False)

    return upsampled_images


def adjust_brightness(images_in, k):
    # Copy the images to avoid changing the originals
    images = images_in.clone()
    # Add the brightness adjustment factor to all pixels
    images_brightness = images + k

    # Clip the values to the range [0, 1]
    images_brightness = torch.clamp(images_brightness, 0, 1)

    return images_brightness


def adjust_contrast(images_in, k):
    # Copy the images to avoid changing the originals
    images = images_in.clone()
    # Calculate the mean of each image
    means = images.mean(dim=[1,2,3], keepdim=True)

    # Adjust contrast
    images_contrast = means + (images - means) * k

    # Clip the values to the range [0, 1]
    images_contrast = torch.clamp(images_contrast, 0, 1)

    return images_contrast
