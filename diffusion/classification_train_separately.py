import logging
import time
import gc
import sys
import os

# 必须在任意使用 torch 的顶层函数（如 set_seed）之前完成导入，避免合并/重排代码后出现 NameError。
import torch
import torch.nn as nn
import torch.utils.data as data

# 设置 matplotlib 使用非交互式后端（避免 X11 显示错误）
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端，适合无头服务器

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy.stats import ttest_rel
from tqdm import tqdm
from data_loader import *
from ema import EMA
from latent_model import *
from pretraining.encoder import Model as AuxCls
from pretraining.resnet import ResNet18
from utils import *
from diffusion_utils import *
from torchmetrics.classification import MulticlassCalibrationError
from torchmetrics import F1Score
from torch.nn.functional import interpolate
import random
from pathlib import Path
from attack import Attack
from autoattack import AutoAttack
from guide_batch import region_to_device as _region_to_device
from guide_batch import unpack_guide_batch as _unpack_guide_batch

plt.style.use('ggplot')


# Swin 主干：仅在 arch=swin 时使用，延迟导入避免影响 sevit 环境
def _load_swin_backbone(ckpt_path, num_classes, device):
    swin_models_dir = Path(__file__).resolve().parent.parent / "mapping" / "models" / "Swin-Transformer-main" / "models"
    if not swin_models_dir.is_dir():
        raise ImportError(
            "Swin Transformer is not bundled in this NAM copy. "
            "Use aux_cls.arch=transnext, or restore mapping/models/Swin-Transformer-main."
        )
    import sys
    sys.path.insert(0, str(swin_models_dir))
    from swin_transformer import SwinTransformer
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict):
        model = SwinTransformer(
            img_size=224, patch_size=4, in_chans=3, num_classes=num_classes,
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
            window_size=7, mlp_ratio=4.0, qkv_bias=True,
            drop_rate=0.0, drop_path_rate=0.1, ape=False, patch_norm=True,
            use_checkpoint=False, fused_window_process=False
        )
        model.load_state_dict(raw, strict=False)
        return model.to(device)
    return raw.to(device)


def _resolve_path_maybe_relative(p):
    """相对路径一律相对 diffusion/ 目录（本文件所在目录）解析，避免从其它 cwd 启动时找不到 mapping。"""
    if p is None:
        return p
    p = str(p)
    if os.path.isabs(p):
        return p
    return str((Path(__file__).resolve().parent / p).resolve())


def _repo_mapping_models_dir(subdir):
    """本仓库 IRa-transnext/mapping/models/<subdir>/（与 train_mapping 保存位置一致）。"""
    return Path(__file__).resolve().parent.parent / "mapping" / "models" / subdir


def _ensure_aux_weights_root(trained_path_cfg, mapping_subdir, backbone_filename):
    """
    解析 yml 中的 trained_aux_cls_ckpt_path；若该目录下没有骨干权重文件，
    则回退到仓库内 mapping/models/<mapping_subdir>/（避免服务器仍指向旧仓库根路径）。
    """
    trained_path = _resolve_path_maybe_relative(trained_path_cfg)
    ck = os.path.join(trained_path, backbone_filename)
    if os.path.isfile(ck):
        return trained_path
    fb = _repo_mapping_models_dir(mapping_subdir)
    fb_ck = fb / backbone_filename
    if fb_ck.is_file():
        print(
            "[warn] yml 中 trained_aux_cls_ckpt_path 下未找到骨干权重:\n"
            f"  已检查: {ck}\n"
            f"  已回退到仓库路径: {fb}"
        )
        return str(fb)
    return trained_path


def _ensure_mlp_import_path(trained_path=None):
    """train_mapping 用 torch.save(model) 保存整模，反序列化需 import mlp（位于 mapping/models/）。"""
    candidates = []
    if trained_path:
        candidates.append(os.path.abspath(os.path.join(trained_path, os.pardir)))
    candidates.append(str(Path(__file__).resolve().parent.parent / "mapping" / "models"))
    seen = set()
    for directory in candidates:
        if not directory or directory in seen:
            continue
        seen.add(directory)
        if os.path.isfile(os.path.join(directory, "mlp.py")):
            if directory not in sys.path:
                sys.path.insert(0, directory)
            return directory
    raise RuntimeError(
        "cannot find mapping/models/mlp.py; MLP checkpoints require Classifier for unpickling"
    )


def _load_transnext_backbone(ckpt_path, num_classes, device):
    transnext_dir = Path(__file__).resolve().parent.parent / "mapping" / "models" / "TransNeXt-main" / "classification"
    if transnext_dir.exists():
        sys.path.insert(0, str(transnext_dir))
    from transnext import transnext_tiny
    ckpt_path = _resolve_path_maybe_relative(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"TransNeXt 权重不存在: {ckpt_path}\n"
            f"  请确认文件存在，且 yml 中 trained_aux_cls_ckpt_path 正确；从任意目录运行时请使用绝对路径，"
            f"或相对 diffusion/ 的路径（如 ../mapping/models/ChestXRay）。")
    raw = torch.load(ckpt_path, map_location=device)
    model = transnext_tiny(pretrained=False, num_classes=num_classes, img_size=224)
    if isinstance(raw, dict):
        state = raw.get('state_dict', raw.get('model', raw))
        if not isinstance(state, dict):
            state = raw
        model_sd = model.state_dict()
        to_load = {}
        for k, v in state.items():
            kk = k[7:] if k.startswith('module.') else k
            if kk not in model_sd or v.shape != model_sd[kk].shape:
                continue
            to_load[kk] = v
        model.load_state_dict(to_load, strict=False)
        return model.to(device)
    if isinstance(raw, nn.Module):
        return raw.to(device)
    raise TypeError(f'Unexpected checkpoint type: {type(raw)}')


def _extract_transnext_stage_feature(backbone, x, stage_idx):
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


def _load_m1d_for_diffusion(trained_path, device):
    """NAM-FiLM (M1d) was removed from this copy; dual Diffusion uses DualBranchClassifier."""
    raise RuntimeError(
        "m1d_enabled is not supported. Set aux_cls.m1d_enabled=False "
        "and use fusion_guide.type=texture_only."
    )


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def filter_nan(pred, label):
    pred_clone = pred.clone()
    label_clone = label.clone()
    valid_indices = ~torch.isnan(pred_clone).any(dim=1)
    filtered_pred = pred_clone[valid_indices]
    filtered_label = label_clone[valid_indices]

    return filtered_pred, filtered_label


def majority_voting_for_mc_samples(predictions):
    # Convert each tensor of logits to class predictions
    class_predictions = [torch.argmax(pred, dim=1) for pred in predictions]

    # Stack the class predictions into a tensor of shape (num_trials, num_instances)
    stacked_predictions = torch.stack(class_predictions)

    # Transpose the tensor to shape (num_instances, num_trials)
    transposed_predictions = torch.transpose(stacked_predictions, 0, 1)

    # Use a for loop to iterate over instances and find the mode for each one
    majority_votes = []
    for i in range(transposed_predictions.shape[0]):
        labels, counts = torch.unique(transposed_predictions[i], return_counts=True)
        majority_vote = labels[counts.argmax()]
        majority_votes.append(majority_vote)

    return torch.stack(majority_votes)


def compute_mean_piws(prediction_tensors, mv, label):
    """
    Computes the mean PIW for correct and incorrect predictions
    :param prediction_tensors: list of N tensors of shape (num_instances, num_classes)
    :param mv: tensor of shape (num_instances,) containing the majority vote for each instance
    :param label: tensor of shape (num_instances,) containing the ground truth label for each instance
    :return: correct_mean_piws, incorrect_mean_piws
    """
    # stack them along a new dimension
    stacked_predictions = torch.stack(prediction_tensors, dim=0)
    # calculate the 2.5th and 97.5th percentiles along this new dimension
    lower_percentile = torch.quantile(stacked_predictions, q=0.025, dim=0)
    upper_percentile = torch.quantile(stacked_predictions, q=0.975, dim=0)
    # calculate the PIW
    piw = upper_percentile - lower_percentile
    # Get the PIWs corresponding to the predicted classes
    predicted_piw = piw[torch.arange(piw.size(0)), mv]
    # Create a mask where the model's predictions match the ground truth
    correct_prediction_mask = mv == label
    # Split the PIWs into correct and incorrect based on the mask
    correct_piw = predicted_piw[correct_prediction_mask]
    incorrect_piw = predicted_piw[~correct_prediction_mask]
    # Calculate the mean PIW for the correct and incorrect predictions
    # mean_piw_correct = correct_piw.mean()
    # mean_piw_incorrect = incorrect_piw.mean()
    mean_piw_correct = correct_piw
    mean_piw_incorrect = incorrect_piw

    return mean_piw_correct, mean_piw_incorrect


def compute_mean_piws_for_class(prediction_tensors, mv, label):
    # stack them along a new dimension
    stacked_predictions = torch.stack(prediction_tensors, dim=0)
    stacked_predictions = stacked_predictions.detach().cpu()
    mv = mv.detach().cpu()
    label = label.detach().cpu()
    # calculate the 2.5th and 97.5th percentiles along this new dimension
    lower_percentile = torch.quantile(stacked_predictions, q=0.025, dim=0)
    upper_percentile = torch.quantile(stacked_predictions, q=0.975, dim=0)
    # calculate the PIW
    piw = upper_percentile - lower_percentile
    # piw: tensor of shape (batch, C)
    # mv: tensor of shape (batch,) containing predicted class indices
    # label: tensor of shape (batch,) containing ground truth class indices

    # Gather predicted PIWs
    predicted_piw = piw[torch.arange(piw.size(0)), mv]

    C = piw.size(1)

    # Initialize tensors to hold mean PIWs for correct and incorrect predictions
    correct_piw = torch.zeros(C)
    incorrect_piw = torch.zeros(C)

    # Loop over all classes
    for c in range(C):
        # Find the indices of instances that are predicted as class c
        indices = (mv == c)

        # Split these indices into correct and incorrect predictions
        correct_indices = indices & (mv == label)
        incorrect_indices = indices & (mv != label)

        # Compute mean PIW for correct and incorrect predictions
        correct_piw[c] = predicted_piw[correct_indices].mean()
        incorrect_piw[c] = predicted_piw[incorrect_indices].mean()

    # correct_piw and incorrect_piw are tensors of shape (C,)
    return correct_piw, incorrect_piw


def calculate_variances(model_logits, predicted_classes, ground_truth):
    N, C = model_logits[0].shape  # Number of instances and classes

    # Optional: Convert logits to probabilities using softmax
    model_probs = model_logits

    # Create tensors to store variances for correct and incorrect predictions
    correct_variances = torch.zeros(C)
    incorrect_variances = torch.zeros(C)

    for c in range(C):
        # Indices of instances predicted as class c and are actually class c (correct predictions)
        correct_indices = (predicted_classes == c) & (ground_truth == c)

        # Indices of instances predicted as class c but are not actually class c (incorrect predictions)
        incorrect_indices = (predicted_classes == c) & (ground_truth != c)

        # Collect probabilities for the correct instances across all models
        correct_probs = torch.stack([probs[correct_indices, c] for probs in model_probs])

        # Collect probabilities for the incorrect instances across all models
        incorrect_probs = torch.stack([probs[incorrect_indices, c] for probs in model_probs])

        # Calculate variance across models for correct predictions
        if correct_probs.shape[1] > 0:
            correct_variances[c] = correct_probs.var(dim=0).mean()

        # Calculate variance across models for incorrect predictions
        if incorrect_probs.shape[1] > 0:
            incorrect_variances[c] = incorrect_probs.var(dim=0).mean()

    return correct_variances, incorrect_variances


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.low_mem_mode = args.low_mem_mode
        # 0 disables noise perturbation
        self.noise_perturbation = args.noise_perturbation
        # 0 disables low resolution, int
        self.low_resolution = args.low_resolution
        # 0.0 disables brightness changes
        self.brightness = args.brightness
        # 1.0 disables contrast changes
        self.contrast = args.contrast
        # 0 disables random crop
        self.crop = args.crop
        # (0, 0) disables random cover
        self.covered = args.covered
        self.atk_name = args.attack_name
        if args.attack_name == 'None':
            self.atk_name = None
        self.eps = args.eps
        self.seed = args.seed
        print('Low memory mode:', self.low_mem_mode)
        set_seed(self.seed)

        self.args = args
        self.config = config
        # M1d namfilm defaults (overwritten when loading ChestXRay/transnext aux)
        self.m1d_enabled = False
        self.m1d_stage_idx = 3
        self.fusion_guide_type = "none"
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        # 监控初始显存使用（使用正确的设备）
        if torch.cuda.is_available() and device.type == 'cuda':
            device_id = device.index if device.index is not None else 0
            print(
                f"[显存监控] 初始化前 (GPU {device_id}) - Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

        self.model_var_type = config.model.var_type
        self.num_timesteps = config.diffusion.timesteps
        self.vis_step = config.diffusion.vis_step
        self.num_figs = config.diffusion.num_figs

        betas = make_beta_schedule(schedule=config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=config.diffusion.beta_start, end=config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)

        # 监控扩散参数加载后的显存
        if torch.cuda.is_available() and self.device.type == 'cuda':
            device_id = self.device.index if self.device.index is not None else 0
            print(
                f"[显存监控] 扩散参数加载后 (GPU {device_id}) - Allocated: {torch.cuda.memory_allocated(self.device) / 1024 ** 3:.2f} GB, Reserved: {torch.cuda.memory_reserved(self.device) / 1024 ** 3:.2f} GB")
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # initial prediction model as guided condition
        if config.diffusion.apply_aux_cls:
            if config.data.dataset in ['ChestXRay', 'ChestXRayAtkFGSM', 'ChestXRayAtkPGD', 'ChestXRayAtkBIM',
                                       'ChestXRayAtkAUTOPGD', 'ChestXRayAtkCW', 'ChestXRayValidate']:
                if config.diffusion.aux_cls.arch == "sevit":
                    # 监控模型加载前的显存
                    if torch.cuda.is_available() and self.device.type == 'cuda':
                        device_id = self.device.index if self.device.index is not None else 0
                        print(
                            f"[显存监控] ViT 模型加载前 (GPU {device_id}) - Allocated: {torch.cuda.memory_allocated(self.device) / 1024 ** 3:.2f} GB, Reserved: {torch.cuda.memory_reserved(self.device) / 1024 ** 3:.2f} GB")

                    # pwd should be in project_root/classification dir
                    trained_path = _resolve_path_maybe_relative(config.diffusion.trained_aux_cls_ckpt_path)
                    _ensure_mlp_import_path(trained_path)
                    sys.path.append(trained_path)
                    self.cond_pred_model = {}
                    # 使用 map_location 确保加载到正确的设备
                    self.cond_pred_model['vit'] = torch.load(
                        os.path.join(trained_path, 'vit_base_patch16_224_ChestXRay.pth'),
                        map_location=self.device)

                    # 监控 ViT 加载后的显存
                    if torch.cuda.is_available() and self.device.type == 'cuda':
                        device_id = self.device.index if self.device.index is not None else 0
                        print(
                            f"[显存监控] ViT 模型加载后 (GPU {device_id}) - Allocated: {torch.cuda.memory_allocated(self.device) / 1024 ** 3:.2f} GB, Reserved: {torch.cuda.memory_reserved(self.device) / 1024 ** 3:.2f} GB")
                    self.cond_pred_model['vit'].eval()
                    self.cond_pred_model['mlps'] = []
                    mlps_root_dir = os.path.join(trained_path, 'MLPs')
                    mlp_list = sorted(os.listdir(mlps_root_dir))

                    # 优化：只加载需要的 MLP（如果 mlp_idx 已指定），否则加载所有（用于测试）
                    mlp_indices_to_load = []
                    if args.mlp_idx is not None and args.mlp_idx >= 0:
                        # 只加载指定的 MLP（索引从 0 开始，文件从 1 开始）
                        mlp_indices_to_load = [args.mlp_idx]
                        print(f"[显存优化] 只加载 MLP {args.mlp_idx}，节省显存")
                    else:
                        # 加载所有 MLP（用于测试或其他用途）
                        mlp_indices_to_load = list(range(len(mlp_list)))
                        print(f"[显存监控] 加载所有 {len(mlp_list)} 个 MLP")

                    for mlp_idx_to_load in mlp_indices_to_load:
                        mlp_file_idx = mlp_idx_to_load + 1  # 文件索引从 1 开始
                        if mlp_file_idx > len(mlp_list):
                            continue

                        # 监控每个 MLP 加载前的显存
                        if torch.cuda.is_available() and not self.low_mem_mode and self.device.type == 'cuda':
                            device_id = self.device.index if self.device.index is not None else 0
                            print(
                                f"[显存监控] MLP {mlp_file_idx} 加载前 (GPU {device_id}) - Allocated: {torch.cuda.memory_allocated(self.device) / 1024 ** 3:.2f} GB, Reserved: {torch.cuda.memory_reserved(self.device) / 1024 ** 3:.2f} GB")

                        if self.low_mem_mode:
                            # 低显存模式：加载到 CPU
                            mlp_model = torch.load(os.path.join(mlps_root_dir, mlp_list[mlp_file_idx - 1]),
                                                   map_location='cpu')
                            # 确保列表长度足够，用 None 填充未加载的 MLP
                            while len(self.cond_pred_model['mlps']) <= mlp_idx_to_load:
                                self.cond_pred_model['mlps'].append(None)
                            self.cond_pred_model['mlps'][mlp_idx_to_load] = mlp_model
                        else:
                            # 正常模式：使用 map_location 确保加载到正确的 GPU
                            mlp_model = torch.load(os.path.join(mlps_root_dir, mlp_list[mlp_file_idx - 1]),
                                                   map_location=self.device)
                            # 确保列表长度足够，用 None 填充未加载的 MLP
                            while len(self.cond_pred_model['mlps']) <= mlp_idx_to_load:
                                self.cond_pred_model['mlps'].append(None)
                            self.cond_pred_model['mlps'][mlp_idx_to_load] = mlp_model

                            # 监控每个 MLP 加载后的显存
                            if torch.cuda.is_available() and self.device.type == 'cuda':
                                device_id = self.device.index if self.device.index is not None else 0
                                print(
                                    f"[显存监控] MLP {mlp_file_idx} 加载后 (GPU {device_id}) - Allocated: {torch.cuda.memory_allocated(self.device) / 1024 ** 3:.2f} GB, Reserved: {torch.cuda.memory_reserved(self.device) / 1024 ** 3:.2f} GB")

                    # 只对已加载的 MLP 设置 eval 模式
                    for ii in range(len(self.cond_pred_model['mlps'])):
                        if self.cond_pred_model['mlps'][ii] is not None:
                            self.cond_pred_model['mlps'][ii].eval()
                    # add one due to vit final output
                    # self.num_noise_estimators_required = len(self.cond_pred_model['mlps']) + 1
                    self.num_noise_estimators_required = len(self.cond_pred_model['mlps']) + 1
                    self.selected_block_indices = [0, 1, 2, 3, 4]

                    print('pretrained model loaded')

                elif config.diffusion.aux_cls.arch == "swin":
                    trained_path = _resolve_path_maybe_relative(config.diffusion.trained_aux_cls_ckpt_path)
                    _mlp_module_dir = os.path.abspath(os.path.join(trained_path, os.pardir))
                    if _mlp_module_dir not in sys.path:
                        sys.path.insert(0, _mlp_module_dir)
                    self.cond_pred_model = {}
                    ckpt_path = os.path.join(trained_path, 'swin_tiny_patch4_window7_224_ChestXRay.pth')
                    self.cond_pred_model['vit'] = _load_swin_backbone(
                        ckpt_path, num_classes=config.data.num_classes, device=self.device)
                    self.cond_pred_model['vit'].eval()
                    self.cond_pred_model['mlps'] = []
                    mlps_root_dir = os.path.join(trained_path, 'MLPs')
                    # 显存优化：训练时只加载当前 mlp_idx 对应的 MLP，测试时加载全部
                    mlp_indices_to_load = list(range(4))
                    if args.mlp_idx is not None and args.mlp_idx >= 0:
                        mlp_indices_to_load = [args.mlp_idx]
                        print(f"[显存优化] 只加载 MLP {args.mlp_idx}，节省显存")
                    for i in range(4):
                        if i in mlp_indices_to_load:
                            name = 'block_{}.pth'.format(i)
                            path = os.path.join(mlps_root_dir, name)
                            if os.path.exists(path):
                                self.cond_pred_model['mlps'].append(torch.load(path, map_location=self.device))
                            else:
                                self.cond_pred_model['mlps'].append(None)
                        else:
                            self.cond_pred_model['mlps'].append(None)
                    for m in self.cond_pred_model['mlps']:
                        if m is not None:
                            m.eval()
                    self.num_noise_estimators_required = len(
                        [m for m in self.cond_pred_model['mlps'] if m is not None]) + 1
                    self.selected_block_indices = [0, 1, 2, 3, 4]
                    print('pretrained Swin + MLPs (ChestXRay) loaded')

                elif config.diffusion.aux_cls.arch == "transnext":
                    trained_path = _ensure_aux_weights_root(
                        config.diffusion.trained_aux_cls_ckpt_path,
                        "ChestXRay",
                        "transnext_tiny_224_ChestXRay.pth",
                    )
                    # train_mapping 用 torch.save(model) 保存整模，加载时需要 import mlp
                    #（mlp.py 在 mapping/models/，与 trained_path 父目录无关）
                    _mlp_module_dir = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), os.pardir, "mapping", "models")
                    )
                    if _mlp_module_dir not in sys.path:
                        sys.path.insert(0, _mlp_module_dir)
                    self.cond_pred_model = {}
                    ckpt_path = os.path.join(trained_path, 'transnext_tiny_224_ChestXRay.pth')
                    self.cond_pred_model['vit'] = _load_transnext_backbone(
                        ckpt_path, num_classes=config.data.num_classes, device=self.device)
                    self.cond_pred_model['vit'].eval()
                    self.cond_pred_model['mlps'] = []
                    mlps_root_dir = os.path.join(trained_path, 'MLPs')
                    mlp_indices_to_load = list(range(4))
                    if args.mlp_idx is not None and args.mlp_idx >= 0:
                        mlp_indices_to_load = [args.mlp_idx]
                        print(f"[显存优化] 只加载 MLP {args.mlp_idx}，节省显存")
                    for i in range(4):
                        if i in mlp_indices_to_load:
                            name = 'block_{}.pth'.format(i)
                            path = os.path.join(mlps_root_dir, name)
                            if os.path.exists(path):
                                self.cond_pred_model['mlps'].append(torch.load(path, map_location=self.device))
                            else:
                                self.cond_pred_model['mlps'].append(None)
                        else:
                            self.cond_pred_model['mlps'].append(None)
                    for m in self.cond_pred_model['mlps']:
                        if m is not None:
                            m.eval()
                    self.num_noise_estimators_required = len(
                        [m for m in self.cond_pred_model['mlps'] if m is not None]) + 1
                    self.selected_block_indices = [0, 1, 2, 3, 4]
                    # Optional M1d namfilm guide (block_3 = h', mlp_idx=4 = M1d logits)
                    self.m1d_enabled = bool(getattr(config.diffusion.aux_cls, 'm1d_enabled', False))
                    self.m1d_stage_idx = int(getattr(config.diffusion.aux_cls, 'm1d_stage_idx', 3))
                    self.cond_pred_model['m1d'] = None
                    self.cond_pred_model['fusion_guide'] = None
                    fusion_config = getattr(config.diffusion.aux_cls, 'fusion_guide', None)
                    self.fusion_guide_type = str(getattr(fusion_config, 'type', 'none')).strip().lower()
                    from fusion_guide_runtime import FUSION_GUIDE_TYPES, TEXTURE_CKPT_ENV, env_override_path
                    if self.fusion_guide_type not in FUSION_GUIDE_TYPES:
                        raise ValueError(
                            "aux_cls.fusion_guide.type must be none or texture_only"
                        )
                    if self.fusion_guide_type == 'texture_only' and not getattr(args, 'test', False) and not getattr(args, 'calib', False):
                        if args.mlp_idx != 3:
                            raise ValueError("texture_only development training requires --mlp_idx 3")
                    if self.fusion_guide_type != 'none':
                        if self.m1d_enabled:
                            raise ValueError("m1d_enabled and fusion_guide cannot be enabled together")
                        guide_index = int(getattr(fusion_config, 'guide_index', 3))
                        if guide_index != 3:
                            raise ValueError("fusion guide_index must be 3; index 4 is reserved for backbone logits")
                        guide_checkpoint = getattr(fusion_config, 'checkpoint', None)
                        if self.fusion_guide_type == 'texture_only':
                            guide_checkpoint = env_override_path(guide_checkpoint, TEXTURE_CKPT_ENV)
                        if not guide_checkpoint:
                            raise ValueError("fusion_guide.checkpoint is required")
                        from fusion_guide_runtime import load_fusion_guide

                        self.cond_pred_model['fusion_guide'] = load_fusion_guide(
                            self.fusion_guide_type,
                            _resolve_path_maybe_relative(guide_checkpoint),
                            self.device,
                        )
                    if self.m1d_enabled:
                        self.cond_pred_model['m1d'] = _load_m1d_for_diffusion(trained_path, self.device)
                        print('pretrained TransNeXt + MLPs + M1d namfilm (ChestXRay) loaded')
                    elif self.fusion_guide_type != 'none':
                        print(
                            f'pretrained TransNeXt + MLPs + {self.fusion_guide_type} '
                            'guide at index 3 (ChestXRay) loaded'
                        )
                    else:
                        print('pretrained TransNeXt + MLPs (ChestXRay) loaded')

            elif config.data.dataset in ['ISICSkinCancer', 'ISICSkinCancerAtkFGSM', 'ISICSkinCancerAtkPGD',
                                         'ISICSkinCancerAtkBIM', 'ISICSkinCancerAtkAUTOPGD', 'ISICSkinCancerAtkCW',
                                         'ISICSkinCancerValidate']:
                if config.diffusion.aux_cls.arch == "sevit":
                    # pwd should be in project_root/classification dir
                    trained_path = _resolve_path_maybe_relative(config.diffusion.trained_aux_cls_ckpt_path)
                    _ensure_mlp_import_path(trained_path)
                    sys.path.append(trained_path)
                    self.cond_pred_model = {}
                    # 使用 map_location 确保加载到正确的设备
                    self.cond_pred_model['vit'] = torch.load(
                        os.path.join(trained_path, 'vit_base_patch16_224_ISICSkinCancer.pth'),
                        map_location=self.device)
                    self.cond_pred_model['vit'].eval()
                    self.cond_pred_model['mlps'] = []
                    mlps_root_dir = os.path.join(trained_path, 'MLPs')
                    mlp_list = sorted(os.listdir(mlps_root_dir))
                    for mlp in range(1, len(mlp_list) + 1):
                        if self.low_mem_mode:
                            if mlp >= len(mlp_list) - 1:
                                # 最后一个 MLP 加载到 CPU
                                self.cond_pred_model['mlps'].append(
                                    torch.load(os.path.join(mlps_root_dir, mlp_list[mlp - 1]),
                                               map_location='cpu'))
                            else:
                                # 其他 MLP 使用 map_location 确保加载到正确的 GPU
                                self.cond_pred_model['mlps'].append(
                                    torch.load(os.path.join(mlps_root_dir, mlp_list[mlp - 1]),
                                               map_location=self.device))
                        else:
                            # 正常模式：使用 map_location 确保加载到正确的 GPU
                            self.cond_pred_model['mlps'].append(
                                torch.load(os.path.join(mlps_root_dir, mlp_list[mlp - 1]),
                                           map_location=self.device))
                    for ii in range(len(self.cond_pred_model['mlps'])):
                        if self.cond_pred_model['mlps'][ii] is not None:
                            self.cond_pred_model['mlps'][ii].eval()
                    # add one due to vit final output
                    # 计算实际加载的 MLP 数量
                    loaded_mlp_count = len([m for m in self.cond_pred_model['mlps'] if m is not None])
                    self.num_noise_estimators_required = loaded_mlp_count + 1
                    self.selected_block_indices = [0, 1, 2, 3, 4]

                    print('pretrained model loaded')

                elif config.diffusion.aux_cls.arch == "swin":
                    trained_path = _resolve_path_maybe_relative(config.diffusion.trained_aux_cls_ckpt_path)
                    _mlp_module_dir = os.path.abspath(os.path.join(trained_path, os.pardir))
                    if _mlp_module_dir not in sys.path:
                        sys.path.insert(0, _mlp_module_dir)
                    self.cond_pred_model = {}
                    ckpt_path = os.path.join(trained_path, 'swin_tiny_patch4_window7_224_ISICSkinCancer.pth')
                    self.cond_pred_model['vit'] = _load_swin_backbone(
                        ckpt_path, num_classes=config.data.num_classes, device=self.device)
                    self.cond_pred_model['vit'].eval()
                    self.cond_pred_model['mlps'] = []
                    mlps_root_dir = os.path.join(trained_path, 'MLPs')
                    for i in range(4):
                        path = os.path.join(mlps_root_dir, 'block_{}.pth'.format(i))
                        self.cond_pred_model['mlps'].append(
                            torch.load(path, map_location=self.device) if os.path.exists(path) else None)
                    for m in self.cond_pred_model['mlps']:
                        if m is not None:
                            m.eval()
                    self.num_noise_estimators_required = len(
                        [m for m in self.cond_pred_model['mlps'] if m is not None]) + 1
                    self.selected_block_indices = [0, 1, 2, 3, 4]
                    print('pretrained Swin + MLPs (ISICSkinCancer) loaded')

                elif config.diffusion.aux_cls.arch == "transnext":
                    trained_path = _ensure_aux_weights_root(
                        config.diffusion.trained_aux_cls_ckpt_path,
                        "ISICSkinCancer",
                        "transnext_tiny_224_ISICSkinCancer.pth",
                    )
                    _mlp_module_dir = os.path.abspath(os.path.join(trained_path, os.pardir))
                    if _mlp_module_dir not in sys.path:
                        sys.path.insert(0, _mlp_module_dir)
                    self.cond_pred_model = {}
                    ckpt_path = os.path.join(trained_path, 'transnext_tiny_224_ISICSkinCancer.pth')
                    self.cond_pred_model['vit'] = _load_transnext_backbone(
                        ckpt_path, num_classes=config.data.num_classes, device=self.device)
                    self.cond_pred_model['vit'].eval()
                    self.cond_pred_model['mlps'] = []
                    mlps_root_dir = os.path.join(trained_path, 'MLPs')
                    for i in range(4):
                        path = os.path.join(mlps_root_dir, 'block_{}.pth'.format(i))
                        self.cond_pred_model['mlps'].append(
                            torch.load(path, map_location=self.device) if os.path.exists(path) else None)
                    for m in self.cond_pred_model['mlps']:
                        if m is not None:
                            m.eval()
                    self.num_noise_estimators_required = len(
                        [m for m in self.cond_pred_model['mlps'] if m is not None]) + 1
                    self.selected_block_indices = [0, 1, 2, 3, 4]
                    print('pretrained TransNeXt + MLPs (ISICSkinCancer) loaded')

            else:
                raise NotImplementedError
            self.aux_cost_function = nn.CrossEntropyLoss()
        else:
            pass

        # scaling temperature for NLL and ECE computation
        if config.data.dataset in ['ChestXRay', 'ChestXRayAtkFGSM', 'ChestXRayAtkPGD', 'ChestXRayAtkBIM',
                                   'ChestXRayAtkAUTOPGD', 'ChestXRayAtkCW', 'ChestXRayValidate']:
            # self.temperature = 0.1752
            self.temperature = 0.1737
        elif config.data.dataset in ['ISICSkinCancer', 'ISICSkinCancerAtkFGSM', 'ISICSkinCancerAtkPGD',
                                     'ISICSkinCancerAtkBIM', 'ISICSkinCancerAtkAUTOPGD', 'ISICSkinCancerAtkCW',
                                     'ISICSkinCancerValidate']:
            self.temperature = 0.3162
        else:
            raise NotImplementedError

    # Compute guiding prediction as diffusion condition
    def compute_guiding_prediction(self, x, nam=None, region=None):
        """
        Compute y_0_hat, to be used as the Gaussian mean at time step T.
        nam: optional (B, nam_dim) for M1d or dual_branch.
        region: optional padded region dictionary for region_attention.
        """
        if self.config.diffusion.aux_cls.arch == 'swin':
            # Swin: 使用 layers（不是 blocks），保持索引与 mlp_idx 对齐
            backbone = self.cond_pred_model['vit']
            max_mlp_count = 5
            y_0_hat = [None] * max_mlp_count
            for i in range(len(self.cond_pred_model['mlps'])):
                if self.cond_pred_model['mlps'][i] is None:
                    continue
                tmp = backbone.patch_embed(x)
                if getattr(backbone, "ape", False):
                    tmp = tmp + backbone.absolute_pos_embed
                tmp = backbone.pos_drop(tmp)
                for j in range(i + 1):
                    tmp = backbone.layers[j](tmp)
                if tmp.dim() >= 3 and tmp.size(0) == 1:
                    tmp = tmp.squeeze(0)
                mlp = self.cond_pred_model['mlps'][i]
                if next(mlp.parameters()).device != self.device:
                    mlp = mlp.to(self.device)
                y_0_hat[i] = mlp(tmp, dataset=self.config.data.dataset)
            # 第 5 路为骨干整图 logits，对应 mlp_idx=4（勿 append，否则变成索引 5 且 y_0_hat[4] 恒为 None）
            y_0_hat[4] = backbone(x)
            return y_0_hat

        if self.config.diffusion.aux_cls.arch == 'transnext':
            backbone = self.cond_pred_model['vit']
            max_mlp_count = 5
            y_0_hat = [None] * max_mlp_count
            m1d = self.cond_pred_model.get('m1d') if getattr(self, 'm1d_enabled', False) else None
            fusion_guide = self.cond_pred_model.get('fusion_guide')
            fusion_type = getattr(self, 'fusion_guide_type', 'none')
            if m1d is not None and nam is None:
                raise ValueError(
                    "m1d_enabled=True but nam batch is None. "
                    "Use ImageFolderWithNam (set aux_cls.nam_feature_dir)."
                )
            if m1d is not None:
                raise RuntimeError(
                    "m1d_enabled is not supported. Use fusion_guide.type=texture_only."
                )
            for i in range(len(self.cond_pred_model['mlps'])):
                if self.cond_pred_model['mlps'][i] is None:
                    continue
                if fusion_guide is not None and i == 3:
                    continue
                if m1d is not None and i == self.m1d_stage_idx:
                    # Mapping block_3 trained on 576-d h'
                    tmp = compute_hprime(backbone, m1d, x, nam, self.m1d_stage_idx)
                else:
                    tmp = _extract_transnext_stage_feature(backbone, x, i)
                    if tmp.dim() >= 3 and tmp.size(0) == 1:
                        tmp = tmp.squeeze(0)
                mlp = self.cond_pred_model['mlps'][i]
                if next(mlp.parameters()).device != self.device:
                    mlp = mlp.to(self.device)
                y_0_hat[i] = mlp(tmp, dataset=self.config.data.dataset)
            if fusion_guide is not None:
                from fusion_guide_runtime import compute_fusion_guides

                y_0_hat = compute_fusion_guides(
                    y_0_hat,
                    backbone,
                    x,
                    _extract_transnext_stage_feature,
                    fusion_type,
                    fusion_guide,
                    nam=nam,
                    region=region,
                )
            elif m1d is not None:
                y_0_hat[4] = compute_m1d_logits(backbone, m1d, x, nam, self.m1d_stage_idx)
            else:
                y_0_hat[4] = backbone(x)
            return y_0_hat

        if self.config.diffusion.aux_cls.arch == 'sevit':
            # ViT/SeViT: 使用 blocks
            max_mlp_count = 5
            y_0_hat = [None] * max_mlp_count
            for i in range(len(self.cond_pred_model['mlps'])):
                if self.cond_pred_model['mlps'][i] is None:
                    continue
                mlp_idx = i + 1
                tmp = self.cond_pred_model['vit'].patch_embed(x)
                tmp = self.cond_pred_model['vit'].pos_drop(tmp)
                for j in range(mlp_idx):
                    tmp = self.cond_pred_model['vit'].blocks[j](tmp)
                mlp_device = next(self.cond_pred_model['mlps'][i].parameters()).device
                if mlp_device != self.device:
                    self.cond_pred_model['mlps'][i] = self.cond_pred_model['mlps'][i].to(self.device)
                y_0_hat[i] = self.cond_pred_model['mlps'][i](tmp, dataset=self.config.data.dataset)
                if self.low_mem_mode:
                    if i >= len(self.cond_pred_model['mlps']) - 1:
                        self.cond_pred_model['mlps'][i].to(torch.device('cpu'))
            y_0_hat[4] = self.cond_pred_model['vit'](x)
            return y_0_hat

        raise NotImplementedError

    def evaluate_guidance_model(self, dataset_loader):
        """
        Evaluate guidance model by reporting train or test set accuracy.
        """
        y_acc_list = []
        for step, feature_label_set in tqdm(enumerate(dataset_loader)):
            x_batch, y_labels_batch, nam_batch, region_batch = _unpack_guide_batch(feature_label_set)
            y_labels_batch = y_labels_batch.reshape(-1)
            if self.config.diffusion.aux_cls.arch in ('sevit', 'swin', 'transnext'):
                final_prediction = []
                nam_dev = nam_batch.to(self.device) if nam_batch is not None else None
                region_dev = _region_to_device(region_batch, self.device)
                y_pred_prob = self.compute_guiding_prediction(
                    x_batch.to(self.device), nam=nam_dev, region=region_dev
                )
                # majority voting
                for tmp in y_pred_prob:
                    if tmp is None:
                        continue
                    predictions = torch.argmax(tmp.detach().cpu(), dim=-1)
                    final_prediction.append(predictions.detach().cpu())
                stacked_tesnor = torch.stack(final_prediction, dim=1)
                y_pred_label = torch.argmax(torch.nn.functional.one_hot(stacked_tesnor).sum(dim=1), dim=-1)
            else:
                raise NotImplementedError
            y_labels_batch = y_labels_batch.cpu().detach().numpy()
            y_acc = y_pred_label.numpy() == y_labels_batch
            if len(y_acc_list) == 0:
                y_acc_list = y_acc
            else:
                y_acc_list = np.concatenate([y_acc_list, y_acc], axis=0)
        y_acc_all = np.mean(y_acc_list)
        return y_acc_all

    def nonlinear_guidance_model_train_step(self, x_batch, y_batch, aux_optimizer):
        """
        One optimization step of the non-linear guidance model that predicts y_0_hat.
        """
        y_batch_pred = self.compute_guiding_prediction(x_batch)
        aux_cost = self.aux_cost_function(y_batch_pred, y_batch)
        # update non-linear guidance model
        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()
        return aux_cost.cpu().item()

    def convert_to_prob(self, logits):
        """
        Convert logits to probabilities.
        """
        logits = ((logits - 1.0) ** 2) * (- 1.0) / self.temperature

        return torch.softmax(logits, dim=-1)

    def compute_nll(self, y_pred, y, prob_in=False):
        """
        Compute NLL loss.
        """
        if prob_in:
            y_pred_prob_log = torch.log(y_pred)
        else:
            y_pred_prob_log = torch.log(self.convert_to_prob(y_pred))
        with torch.no_grad():
            nll_loss = nn.NLLLoss()(y_pred_prob_log, y)

        return nll_loss.item()

    def compute_ece(self, y_pred, y, prob_in=False):
        """
        Compute ECE. 使用 try-except 规避 torchmetrics 在置信度边界处的分箱越界 bug.
        """
        ece_metric = MulticlassCalibrationError(num_classes=self.config.data.num_classes, n_bins=10, norm='l1')
        if prob_in:
            y_pred_prob = y_pred
        else:
            y_pred_prob = self.convert_to_prob(y_pred)
        eps = 1e-4
        y_pred_prob = y_pred_prob.clamp(eps, 1.0 - eps)
        try:
            return ece_metric(y_pred_prob, y)
        except RuntimeError as e:
            if "index" in str(e) and "out of bounds" in str(e):
                # torchmetrics 分箱边界 bug，用手动 ECE 近似
                return self._compute_ece_manual(y_pred_prob, y)
            raise

    def _compute_ece_manual(self, y_pred_prob, y_true):
        """手动计算 ECE，避免 torchmetrics 分箱边界 bug."""
        n_bins = 10
        confidences = y_pred_prob.max(dim=-1).values.clamp(1e-6, 1.0 - 1e-6)
        pred_labels = y_pred_prob.argmax(dim=-1)
        y_flat = y_true.reshape(-1) if y_true.dim() > 1 else y_true
        accuracies = (pred_labels == y_flat).float()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=confidences.device)
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
        ece = torch.tensor(0.0, device=confidences.device)
        for i in range(n_bins):
            hi = bin_uppers[i] if i < n_bins - 1 else 1.0 + 1e-6
            in_bin = (confidences > bin_lowers[i]) & (confidences <= hi)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
                avg_acc = accuracies[in_bin].mean()
                avg_conf = confidences[in_bin].mean()
                ece = ece + torch.abs(avg_acc - avg_conf) * prop_in_bin
        return ece.item()

    def compute_ensemble_confidence(self, outputs):
        """
        This function takes the outputs from an ensemble of K models,
        and returns the overall confidence of the ensemble.

        Parameters:
        outputs: a list of tensors, each of shape (batch_size, C)
                 where C is the number of classes.

        Returns:
        overall_confidence: a tensor of shape (batch_size, C)
        """

        for i in range(len(outputs)):
            outputs[i] = self.convert_to_prob(outputs[i])

        # Convert the list of outputs to a tensor
        all_outputs = torch.stack(outputs)

        # Compute the mean confidence along the ensemble dimension (dimension 0)
        overall_confidence = torch.mean(all_outputs, dim=0)

        return overall_confidence

    def test_calibrate(self, temp=None):
        if temp is not None:
            self.temperature = torch.tensor(temp).to(torch.device('cpu'))
        print('Using seed: {}'.format(self.seed))
        if self.noise_perturbation == 0:
            print('Noise perturbation: disabled')
        else:
            print('Noise perturbation: {}'.format(self.noise_perturbation))
        if self.low_resolution <= 1:
            print('Low resolution: disabled')
        else:
            print('Low resolution: {}'.format(self.low_resolution))
        if self.brightness == 0.0:
            print('Brightness: disabled')
        else:
            print('Brightness: {}'.format(self.brightness))
        if self.contrast == 1.0:
            print('Contrast: disabled')
        else:
            print('Contrast: {}'.format(self.contrast))
        if self.crop == 0:
            print('Crop: disabled')
        else:
            print('Crop: {}'.format(self.crop))
        if self.covered[0] == 0:
            print('Covered: disabled')
        else:
            print('Covered: {}'.format(self.covered))
        if self.atk_name is None:
            print('Attack: disabled')
        else:
            print('Attack: {} eps {}'.format(self.atk_name, self.eps))
        if self.atk_name is not None:
            self.cond_pred_model['vit'].eval()
            if self.atk_name != 'AUTOPGD':
                attack = Attack(epsilon=self.eps, attack_type=self.atk_name, model=self.cond_pred_model['vit'])
            else:
                attack = AutoAttack(model=self.cond_pred_model['vit'], eps=self.eps, version='custom', norm='Linf',
                                    attacks_to_run=['apgd-ce'])

        print('Testing calibration on validation set {} with temperature {}.'.format(self.config.data.dataroot,
                                                                                     self.temperature))
        # print('Using model {}'.format(self.config.diffusion.trained_diffusion_ckpt_path))
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        _, _, test_dataset = get_dataset(args, config)
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            drop_last=True
        )

        # load trained diffusion model
        # 根据配置中实际提供的 ckpt 路径数量加载，而不是 num_noise_estimators_required
        num_diffusion_models = len(config.diffusion.trained_diffusion_ckpt_path[0])
        print(f"Loading {num_diffusion_models} diffusion models from config...")
        noise_estimators = []
        for i in range(num_diffusion_models):
            # load diffusion models to CPU to save GPU memory
            noise_estimators.append(
                ConditionalModel(config, guidance=config.diffusion.include_guidance).to(torch.device('cpu')))
            state = torch.load(config.diffusion.trained_diffusion_ckpt_path[0][i], map_location='cpu')
            noise_estimators[i].load_state_dict(state['noise_estimator'])
            print('Diffusion model {} loaded'.format(i))
        if self.config.diffusion.aux_cls.arch in ('sevit', 'swin', 'transnext'):
            for ii in range(len(noise_estimators)):
                noise_estimators[ii].eval()
            self.cond_pred_model['vit'].eval()
            for ii in range(len(self.cond_pred_model['mlps'])):
                if self.cond_pred_model['mlps'][ii] is not None:
                    self.cond_pred_model['mlps'][ii].eval()
        else:
            raise NotImplementedError

        # print CUDA memory usage
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

        # evaluate trained diffusion model
        if self.config.diffusion.aux_cls.arch in ('sevit', 'swin', 'transnext'):
            target_class = []
            prob_mc = []
        else:
            raise NotImplementedError

        total_batches = len(test_loader)
        print(f"[test_calibrate] Starting inference on {total_batches} batches...")
        import time as time_module
        start_time = time_module.time()

        for test_batch_idx, feature_label_set in enumerate(test_loader):
            images_raw, target, nam_batch, region_batch = _unpack_guide_batch(feature_label_set)
            batch_start = time_module.time()
            nam_dev = None
            region_dev = None
            if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext") \
                    and self.config.data.dataset in ['ChestXRay', 'ChestXRayAtkFGSM', 'ChestXRayAtkPGD',
                                                     'ChestXRayAtkBIM', 'ChestXRayAtkAUTOPGD', 'ChestXRayAtkCW',
                                                     'ChestXRayValidate', 'ISICSkinCancer', 'ISICSkinCancerAtkFGSM',
                                                     'ISICSkinCancerAtkPGD', 'ISICSkinCancerAtkBIM',
                                                     'ISICSkinCancerAtkAUTOPGD', 'ISICSkinCancerAtkCW',
                                                     'ISICSkinCancerValidate']:
                images_224 = images_raw.clone().to(self.device)
                target = target.to(self.device)
                nam_dev = nam_batch.to(self.device) if nam_batch is not None else None
                region_dev = _region_to_device(region_batch, self.device)

                # perturbations
                if self.noise_perturbation > 0.0:
                    images_224 = add_noise(images_224, self.noise_perturbation)
                if self.low_resolution > 1:
                    images_224 = down_up_sample(images_224, self.low_resolution)
                if self.brightness != 0.0:
                    images_224 = adjust_brightness(images_224, self.brightness)
                if self.contrast != 1.0:
                    images_224 = adjust_contrast(images_224, self.contrast)
                if self.covered[0] > 0:
                    images_224 = random_cover_new(images_224, self.covered)
                if self.crop > 0:
                    images_224 = random_crop_and_resize(images_224, self.crop)
                if self.atk_name is not None:
                    images_224 = apply_attack(attack, images_224, target, self.atk_name)

            else:
                raise NotImplementedError

            if config.data.dataset == "toy" \
                    or config.model.arch == "simple" \
                    or config.model.arch == "linear":
                images_224_flat = torch.flatten(images_224, 1).to(self.device)

                with torch.no_grad():
                    if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext"):
                        # target_pred is a list
                        with torch.no_grad():
                            target_pred = self.compute_guiding_prediction(
                                images_224, nam=nam_dev, region=region_dev
                            )
                        # apply softmax to logits
                        tmp = []
                        for entry in target_pred:
                            if entry is not None:
                                tmp.append(nn.functional.softmax(entry, dim=1).to(self.device))
                            else:
                                tmp.append(None)
                        target_pred = tmp
                    else:
                        raise NotImplementedError

                    y_T_mean = target_pred

                    if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext"):
                        # list of K*N (batch, C) tensors
                        mc_samples = []
                        for ii in range(len(noise_estimators)):
                            with torch.no_grad():
                                if ii in self.selected_block_indices:
                                    mc_trials = 20
                                    for _ in range(mc_trials):
                                        mc_samples.append(
                                            p_sample_loop(noise_estimators[ii].to(self.device), images_224_flat,
                                                          target_pred[ii],
                                                          y_T_mean[ii], self.num_timesteps, self.alphas,
                                                          self.one_minus_alphas_bar_sqrt,
                                                          only_last_sample=True))

                                # move diffusion models back to CPU to save GPU memory
                                noise_estimators[ii].to(torch.device('cpu'))

                        # move all tensors to CPU to save GPU memory
                        for ii in range(len(mc_samples)):
                            mc_samples[ii] = mc_samples[ii].detach().cpu()

                        target_class.append(target.detach().cpu())
                        prob_mc.append(self.compute_ensemble_confidence(mc_samples))

                    else:
                        raise NotImplementedError

            # 打印进度
            batch_time = time_module.time() - batch_start
            elapsed = time_module.time() - start_time
            eta = (elapsed / (test_batch_idx + 1)) * (total_batches - test_batch_idx - 1)
            print(
                f"[Batch {test_batch_idx + 1}/{total_batches}] Time: {batch_time:.1f}s | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

        if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext"):
            # TODO
            ece = self.compute_ece(torch.cat(prob_mc, dim=0), torch.cat(target_class, dim=0))
        else:
            raise NotImplementedError

        print(f"Ours ECE: {ece} \n")

        if not tb_logger is None:
            tb_logger.add_scalar('calibration', ece, global_step=0)
        logging.info(
            (
                f"Ours ECE: {ece} \n"
            )
        )

        # 保存校准指标到 JSON 文件
        import json
        calib_metrics = {
            "ece": float(ece),
            "seed": self.seed,
            "num_diffusion_models": len(noise_estimators),
            "mc_trials": 20,
            "num_timesteps": self.num_timesteps
        }
        metrics_path = os.path.join(self.args.log_path, "calibration_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(calib_metrics, f, indent=4)
        print(f"\n校准指标已保存到: {metrics_path}")

        return ece

    def test_atk(self):
        print('Using seed: {}'.format(self.seed))
        if self.noise_perturbation == 0:
            print('Noise perturbation: disabled')
        else:
            print('Noise perturbation: {}'.format(self.noise_perturbation))
        if self.low_resolution <= 1:
            print('Low resolution: disabled')
        else:
            print('Low resolution: {}'.format(self.low_resolution))
        if self.brightness == 0.0:
            print('Brightness: disabled')
        else:
            print('Brightness: {}'.format(self.brightness))
        if self.contrast == 1.0:
            print('Contrast: disabled')
        else:
            print('Contrast: {}'.format(self.contrast))
        if self.crop == 0:
            print('Crop: disabled')
        else:
            print('Crop: {}'.format(self.crop))
        if self.covered[0] == 0:
            print('Covered: disabled')
        else:
            print('Covered: {}'.format(self.covered))
        if self.atk_name is None:
            print('Attack: disabled')
        else:
            print('Attack: {} eps {}'.format(self.atk_name, self.eps))
        if self.atk_name is not None:
            self.cond_pred_model['vit'].eval()
            if self.atk_name != 'AUTOPGD':
                attack = Attack(epsilon=self.eps, attack_type=self.atk_name, model=self.cond_pred_model['vit'])
            else:
                attack = AutoAttack(model=self.cond_pred_model['vit'], eps=self.eps, version='custom', norm='Linf',
                                    attacks_to_run=['apgd-ce'])

        print('Testing on test set {}'.format(self.config.data.dataroot))
        # print('Using model {}'.format(self.config.diffusion.trained_diffusion_ckpt_path))
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        _, _, _, test_dataset = get_dataset(args, config)
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            drop_last=True
        )

        # load trained diffusion model
        # 根据配置中实际提供的 ckpt 路径数量加载，而不是 num_noise_estimators_required
        num_diffusion_models = len(config.diffusion.trained_diffusion_ckpt_path[0])
        print(f"Loading {num_diffusion_models} diffusion models from config...")
        noise_estimators = []
        for i in range(num_diffusion_models):
            # load diffusion models to CPU to save GPU memory
            noise_estimators.append(
                ConditionalModel(config, guidance=config.diffusion.include_guidance).to(torch.device('cpu')))
            state = torch.load(config.diffusion.trained_diffusion_ckpt_path[0][i], map_location='cpu')
            noise_estimators[i].load_state_dict(state['noise_estimator'])
            print('Diffusion model {} loaded'.format(i))
        if self.config.diffusion.aux_cls.arch in ('sevit', 'swin', 'transnext'):
            for ii in range(len(noise_estimators)):
                noise_estimators[ii].eval()
            self.cond_pred_model['vit'].eval()
            for ii in range(len(self.cond_pred_model['mlps'])):
                if self.cond_pred_model['mlps'][ii] is not None:
                    self.cond_pred_model['mlps'][ii].eval()
        else:
            raise NotImplementedError

        # print CUDA memory usage
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

        # evaluate trained diffusion model
        if self.config.diffusion.aux_cls.arch in ('sevit', 'swin', 'transnext'):
            mv_acc_avg_mc = 0.0
            mv_class = []
            target_class = []
            pred_mc = None
            prob_mc = []
        else:
            raise NotImplementedError

        total_batches = len(test_loader)
        print(f"Starting inference on {total_batches} batches...")
        import time as time_module
        start_time = time_module.time()

        for test_batch_idx, feature_label_set in enumerate(test_loader):
            images_raw, target, nam_batch, region_batch = _unpack_guide_batch(feature_label_set)
            batch_start = time_module.time()
            nam_dev = None
            region_dev = None
            _arch = getattr(getattr(self.config.diffusion, 'aux_cls', None), 'arch', None)
            if (_arch in ("sevit", "swin", "transnext") or (
                    getattr(self, 'cond_pred_model', None) and 'vit' in self.cond_pred_model)) \
                    and self.config.data.dataset in ['ChestXRay', 'ChestXRayAtkFGSM', 'ChestXRayAtkPGD',
                                                     'ChestXRayAtkBIM', 'ChestXRayAtkAUTOPGD', 'ChestXRayAtkCW',
                                                     'ISICSkinCancer', 'ISICSkinCancerAtkFGSM', 'ISICSkinCancerAtkPGD',
                                                     'ISICSkinCancerAtkBIM', 'ISICSkinCancerAtkAUTOPGD',
                                                     'ISICSkinCancerAtkCW']:
                images_224 = images_raw.clone().to(self.device)
                target = target.to(self.device)
                nam_dev = nam_batch.to(self.device) if nam_batch is not None else None
                region_dev = _region_to_device(region_batch, self.device)

                # perturbations
                if self.noise_perturbation > 0.0:
                    images_224 = add_noise(images_224, self.noise_perturbation)
                if self.low_resolution > 1:
                    images_224 = down_up_sample(images_224, self.low_resolution)
                if self.brightness != 0.0:
                    images_224 = adjust_brightness(images_224, self.brightness)
                if self.contrast != 1.0:
                    images_224 = adjust_contrast(images_224, self.contrast)
                if self.covered[0] > 0:
                    images_224 = random_cover_new(images_224, self.covered)
                if self.crop > 0:
                    images_224 = random_crop_and_resize(images_224, self.crop)
                if self.atk_name is not None:
                    images_224 = apply_attack(attack, images_224, target, self.atk_name)

            else:
                raise NotImplementedError

            if config.data.dataset == "toy" \
                    or config.model.arch == "simple" \
                    or config.model.arch == "linear":
                images_224_flat = torch.flatten(images_224, 1).to(self.device)

            with torch.no_grad():
                _arch = getattr(getattr(self.config.diffusion, 'aux_cls', None), 'arch', None)
                if _arch in ("sevit", "swin", "transnext") or (
                        getattr(self, 'cond_pred_model', None) and 'vit' in self.cond_pred_model):
                    with torch.no_grad():
                        target_pred = self.compute_guiding_prediction(
                            images_224, nam=nam_dev, region=region_dev
                        )
                    tmp = []
                    for entry in target_pred:
                        if entry is not None:
                            tmp.append(nn.functional.softmax(entry, dim=1).to(self.device))
                        else:
                            tmp.append(None)
                    target_pred = tmp
                else:
                    raise NotImplementedError

                y_T_mean = target_pred

                if _arch in ("sevit", "swin", "transnext") or (
                        getattr(self, 'cond_pred_model', None) and 'vit' in self.cond_pred_model):
                    # list of K*N (batch, C) tensors
                    mc_samples = []
                    for ii in range(len(noise_estimators)):
                        with torch.no_grad():
                            if ii in self.selected_block_indices:
                                mc_trials = 20
                                for _ in range(mc_trials):
                                    mc_samples.append(
                                        p_sample_loop(noise_estimators[ii].to(self.device), images_224_flat,
                                                      target_pred[ii],
                                                      y_T_mean[ii], self.num_timesteps, self.alphas,
                                                      self.one_minus_alphas_bar_sqrt,
                                                      only_last_sample=True))

                            # move diffusion models back to CPU to save GPU memory
                            noise_estimators[ii].to(torch.device('cpu'))

                    # move all tensors to CPU to save GPU memory
                    for ii in range(len(mc_samples)):
                        mc_samples[ii] = mc_samples[ii].detach().cpu()

                    class_pred_batch = majority_voting_for_mc_samples(mc_samples)
                    mv_class.append(class_pred_batch.detach().cpu())
                    target_class.append(target.detach().cpu())
                    prob_mc.append(self.compute_ensemble_confidence(mc_samples))
                    if pred_mc is None:
                        pred_mc = mc_samples
                    else:
                        for ii in range(len(mc_samples)):
                            pred_mc[ii] = torch.cat((pred_mc[ii], mc_samples[ii]), dim=0)

                else:
                    raise NotImplementedError

            # 打印进度
            batch_time = time_module.time() - batch_start
            elapsed = time_module.time() - start_time
            eta = (elapsed / (test_batch_idx + 1)) * (total_batches - test_batch_idx - 1)
            print(
                f"[Batch {test_batch_idx + 1}/{total_batches}] Time: {batch_time:.1f}s | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

        if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext"):

            def compute_accuracy(predictions, labels):
                predictions = predictions.detach().cpu()
                labels = labels.detach().cpu()
                correct_predictions = torch.sum(predictions == labels)
                total_predictions = predictions.numel()
                accuracy = correct_predictions.float() / total_predictions
                return accuracy

            # TODO
            mv_acc_avg_mc = compute_accuracy(torch.cat(mv_class, dim=0), torch.cat(target_class, dim=0))
            piw_correct, piw_incorrect = compute_mean_piws_for_class(pred_mc, torch.cat(mv_class, dim=0),
                                                                     torch.cat(target_class, dim=0))
            ece = self.compute_ece(torch.cat(prob_mc, dim=0), torch.cat(target_class, dim=0))
            correct_variances, incorrect_variances = calculate_variances(pred_mc, torch.cat(mv_class, dim=0),
                                                                         torch.cat(target_class, dim=0))

            all_probs = torch.cat(prob_mc, dim=0)
            all_labels = torch.cat(target_class, dim=0)
            all_preds = torch.cat(mv_class, dim=0)
            try:
                np.savez_compressed(
                    os.path.join(self.args.log_path, "test_official_probs.npz"),
                    p_off=all_probs.detach().cpu().numpy().astype(np.float32),
                    y=all_labels.detach().cpu().numpy().astype(np.int64),
                    pred=all_preds.detach().cpu().numpy().astype(np.int64),
                )
                print(f"saved {os.path.join(self.args.log_path, 'test_official_probs.npz')}")
            except Exception as exc:
                print(f"failed to save test_official_probs.npz: {exc}")
            cls_metrics = compute_extended_classification_metrics(
                all_probs,
                all_labels,
                preds=all_preds,
                num_classes=config.data.num_classes,
                positive_class=1,
            )
            core_cls_metrics = extract_core_classification_metrics(cls_metrics)

        else:
            raise NotImplementedError

        print(f"Majority voting accuracy for MC: {mv_acc_avg_mc :.4f} \n" +
              f"ECE: {ece :.4f} \n" +
              f"Average correct PIW per class: {piw_correct} \n" +
              f"Average incorrect PIW per class: {piw_incorrect} \n" +
              f"Average correct variances per class: {correct_variances} \n" +
              f"Average incorrect variances per class: {incorrect_variances}")
        print(format_classification_metrics(core_cls_metrics))

        if not tb_logger is None:
            tb_logger.add_scalar('accuracy', mv_acc_avg_mc, global_step=0)
            for metric_name, metric_value in core_cls_metrics.items():
                if metric_value is not None and not (
                        isinstance(metric_value, float) and np.isnan(metric_value)
                ):
                    tb_logger.add_scalar(metric_name, float(metric_value), global_step=0)
        logging.info(
            (
                    f"Majority voting accuracy for MC: {mv_acc_avg_mc :.4f} \n" +
                    f"ECE: {ece :.4f} \n" +
                    f"Average correct PIW per class: {piw_correct} \n" +
                    f"Average incorrect PIW per class: {piw_incorrect} \n" +
                    f"Average correct variances per class: {correct_variances} \n" +
                    f"Average incorrect variances per class: {incorrect_variances} \n" +
                    format_classification_metrics(core_cls_metrics) + "\n"
            )
        )

        # 保存测试指标到 JSON 文件
        import json
        test_metrics = {
            "accuracy": float(mv_acc_avg_mc),
            "ece": float(ece),
            **{k: float(v) for k, v in core_cls_metrics.items()},
            "f1": float(cls_metrics.get("f1", float("nan"))),
            "precision": float(cls_metrics.get("precision", float("nan"))),
            "recall": float(cls_metrics.get("recall", float("nan"))),
            "confusion_matrix": cls_metrics.get("confusion_matrix"),
            "per_class_recall": cls_metrics.get("per_class_recall"),
            "piw_correct": piw_correct.tolist() if hasattr(piw_correct, 'tolist') else piw_correct,
            "piw_incorrect": piw_incorrect.tolist() if hasattr(piw_incorrect, 'tolist') else piw_incorrect,
            "variances_correct": correct_variances.tolist() if hasattr(correct_variances,
                                                                       'tolist') else correct_variances,
            "variances_incorrect": incorrect_variances.tolist() if hasattr(incorrect_variances,
                                                                           'tolist') else incorrect_variances,
            "seed": self.seed,
            "num_diffusion_models": len(noise_estimators),
            "mc_trials": 20,
            "num_timesteps": self.num_timesteps
        }
        metrics_path = os.path.join(self.args.log_path, "test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        print(f"\n测试指标已保存到: {metrics_path}")

        return mv_acc_avg_mc

    def train(self, mlp_idx):
        if mlp_idx is None:
            mlp_idx = 0
        if mlp_idx >= 0:
            print('NOTE: TRAINING DIFFUSION MODEL FOR MLP {}'.format(mlp_idx))
        elif mlp_idx == -1:
            print('NOTE: TRAINING DIFFUSION MODEL FOR ViT OUTPUT')
        else:
            raise NotImplementedError
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        data_object, train_dataset, valid_dataset, _ = get_dataset(args, config)
        if getattr(self, "fusion_guide_type", "none") == "texture_only":
            from fusion_guide_runtime import assert_class_mapping, assert_no_test_split_access

            guide = self.cond_pred_model.get("fusion_guide")
            if guide is None:
                raise RuntimeError("texture_only training requires a loaded fusion_guide")
            assert_class_mapping(guide.guide_class_to_idx, train_dataset.class_to_idx)
            assert_class_mapping(guide.guide_class_to_idx, valid_dataset.class_to_idx)
            assert_no_test_split_access()
            print(
                "[texture-only] selection_split=validation "
                f"guide_index=3 class_to_idx={dict(train_dataset.class_to_idx)}"
            )
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        valid_loader = data.DataLoader(
            valid_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            drop_last=True,
        )

        # 清理显存缓存
        torch.cuda.empty_cache()
        gc.collect()

        # initialize separate noise estimator
        noise_estimator = ConditionalModel(config, guidance=config.diffusion.include_guidance).to(self.device)

        # 在创建优化器前清理显存
        torch.cuda.empty_cache()
        gc.collect()

        optimizer = get_optimizer(self.config.optim, noise_estimator.parameters())
        criterion = nn.CrossEntropyLoss()

        # 详细的显存诊断
        if torch.cuda.is_available() and self.device.type == 'cuda':
            device_id = self.device.index if self.device.index is not None else 0
            allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(self.device) / 1024 ** 3

            # 计算模型参数大小
            model_params = sum(p.numel() for p in noise_estimator.parameters())
            model_size_gb = model_params * 4 / 1024 ** 3  # 假设 float32，4 bytes per param

            # 计算优化器状态大小（Adam 需要 2倍参数大小：momentum + variance）
            optimizer_state_gb = model_params * 8 / 1024 ** 3  # Adam: 2 states * 4 bytes

            # 计算 ViT + MLP 大小
            vit_params = sum(p.numel() for p in self.cond_pred_model['vit'].parameters())
            vit_size_gb = vit_params * 4 / 1024 ** 3
            loaded_mlps = [m for m in self.cond_pred_model['mlps'] if m is not None]
            mlp_size_gb = sum(sum(p.numel() for p in mlp.parameters()) for mlp in loaded_mlps) * 4 / 1024 ** 3

            # 估算 EMA shadow 参数大小（如果启用）
            ema_size_gb = model_size_gb if self.config.model.ema else 0

            # 估算训练时的中间变量（梯度、激活值等）
            # batch_size=4, 假设每个样本需要 ~100MB 的中间激活值
            batch_size = config.training.batch_size
            estimated_activations_gb = batch_size * 0.1  # 粗略估算

            print(f"\n[显存诊断] ========== GPU {device_id} 显存占用详细分析 ==========")
            print(f"[显存诊断] 当前状态: 已分配={allocated:.2f} GB, 已保留={reserved:.2f} GB")
            print(f"[显存诊断] --- 模型参数 ---")
            print(f"[显存诊断]   扩散模型 (ConditionalModel): {model_params:,} 参数 = {model_size_gb:.2f} GB")
            print(f"[显存诊断]   ViT 模型: {vit_params:,} 参数 = {vit_size_gb:.2f} GB")
            print(f"[显存诊断]   MLP 模型 ({len(loaded_mlps)} 个): ~{mlp_size_gb:.2f} GB")
            print(f"[显存诊断]   模型参数总计: ~{model_size_gb + vit_size_gb + mlp_size_gb:.2f} GB")
            print(f"[显存诊断] --- 训练状态 ---")
            print(f"[显存诊断]   优化器状态 (Adam): ~{optimizer_state_gb:.2f} GB (momentum + variance)")
            print(f"[显存诊断]   EMA shadow 参数: ~{ema_size_gb:.2f} GB")
            print(f"[显存诊断]   梯度 (backward时): ~{model_size_gb:.2f} GB")
            print(f"[显存诊断]   中间激活值 (batch={batch_size}): ~{estimated_activations_gb:.2f} GB")
            print(f"[显存诊断] --- 总计估算 ---")
            total_estimated = model_size_gb + optimizer_state_gb + ema_size_gb + vit_size_gb + mlp_size_gb
            print(f"[显存诊断]   模型+优化器+EMA: ~{total_estimated:.2f} GB")
            print(f"[显存诊断]   实际已分配: {allocated:.2f} GB")
            print(f"[显存诊断]   差异: {allocated - total_estimated:.2f} GB (可能是CUDA缓存、数据加载器等)")
            print(f"[显存诊断] ====================================================\n")

        # apply an auxiliary optimizer for the guidance classifier
        # if config.diffusion.apply_aux_cls:
        #     aux_optimizer = get_optimizer(self.config.aux_optim,
        #                                   self.cond_pred_model.parameters())

        print(f"Before EMA - CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"Before EMA - CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

        if self.config.model.ema:
            # 再次清理显存，为 EMA 注册腾出空间
            torch.cuda.empty_cache()
            ema_helper_noise_estimator = EMA(mu=self.config.model.ema_rate)
            ema_helper_noise_estimator.register(noise_estimator)
        else:
            ema_helpers_noise_estimators = None

        print(f"After EMA - CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"After EMA - CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

        if not self.args.train_guidance_only:
            start_epoch, step = 0, 0
            # if self.args.resume_training:
            #     # TODO: include resume training for the attention_classifier
            #     states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
            #                         map_location=self.device)
            #     noise_estimator.load_state_dict(states[0])
            #     states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            #     optimizer.load_state_dict(states[1])
            #     start_epoch = states[2]
            #     step = states[3]
            #     if self.config.model.ema:
            #         ema_helper.load_state_dict(states[4])
            #     # load auxiliary model
            #     if config.diffusion.apply_aux_cls and (
            #             hasattr(config.diffusion, "trained_aux_cls_ckpt_path") is False) and (
            #             hasattr(config.diffusion, "trained_aux_cls_log_path") is False):
            #         aux_states = torch.load(os.path.join(self.args.log_path, "aux_ckpt.pth"),
            #                                 map_location=self.device)
            #         self.cond_pred_model.load_state_dict(aux_states[0])
            #         aux_optimizer.load_state_dict(aux_states[1])

            if self.config.diffusion.aux_cls.arch in ('sevit', 'swin', 'transnext'):
                max_accuracy = 0.0
            else:
                raise NotImplementedError
            for epoch in range(start_epoch, self.config.training.n_epochs):
                data_start = time.time()
                data_time = 0
                last_loss_value = 0.0  # 初始化，防止循环未执行时出错
                for i, feature_label_set in enumerate(train_loader):
                    x_batch_raw, y_labels_batch, nam_batch, region_batch = _unpack_guide_batch(feature_label_set)

                    if self.config.diffusion.aux_cls.arch in ('sevit', 'swin', 'transnext') and self.config.data.dataset in [
                        'ChestXRay',
                        'ISICSkinCancer']:
                        x_batch_224 = x_batch_raw.clone().to(self.device)
                        nam_dev = nam_batch.to(self.device) if nam_batch is not None else None
                        region_dev = _region_to_device(region_batch, self.device)
                    else:
                        raise NotImplementedError
                    y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch, config)

                    if config.optim.lr_schedule:
                        adjust_learning_rate(optimizer, i / len(train_loader) + epoch, config)

                    n = x_batch_224.size(0)
                    if config.data.dataset == "toy" or config.model.arch in ["simple", "linear"]:
                        x_batch_224_flat = torch.flatten(x_batch_224, 1).to(self.device)

                    data_time += time.time() - data_start
                    # sevit / swin 多引导头：noise_estimator train，vit/mlps eval
                    noise_estimator.train()
                    self.cond_pred_model['vit'].eval()
                    for ii in range(len(self.cond_pred_model.get('mlps', []))):
                        if self.cond_pred_model['mlps'][ii] is not None:
                            self.cond_pred_model['mlps'][ii].eval()
                    step += 1

                    # antithetic sampling
                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)
                    t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

                    if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext"):
                        # y_0_hat_batch is a list
                        with torch.no_grad():
                            y_0_hat_batch = self.compute_guiding_prediction(
                                x_batch_224, nam=nam_dev, region=region_dev
                            )
                        # apply softmax to logits（跳过 None 值）
                        tmp = []
                        for entry in y_0_hat_batch:
                            if entry is not None:
                                tmp.append(nn.functional.softmax(entry, dim=1).to(self.device))
                            else:
                                tmp.append(None)
                        y_0_hat_batch = tmp
                    else:
                        raise NotImplementedError

                    # compute outputs
                    y_T_mean = y_0_hat_batch
                    y_0_batch = y_one_hot_batch.to(self.device)
                    e = torch.randn_like(y_0_batch).to(y_0_batch.device)
                    if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext"):
                        # 确保 mlp_idx 对应的值不是 None
                        if y_T_mean[mlp_idx] is None or y_0_hat_batch[mlp_idx] is None:
                            raise ValueError(f"MLP {mlp_idx} 未加载！请确保在初始化时加载了 MLP {mlp_idx}")
                        y_t_batch = q_sample(y_0_batch, y_T_mean[mlp_idx], self.alphas_bar_sqrt,
                                             self.one_minus_alphas_bar_sqrt, t, noise=e).to(self.device)
                        noise_output = noise_estimator(x_batch_224_flat, y_t_batch, t, y_0_hat_batch[mlp_idx])
                    else:
                        raise NotImplementedError

                    # compute loss
                    if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext"):
                        loss = (e - noise_output).square().mean()
                    else:
                        raise NotImplementedError

                    if not tb_logger is None:
                        tb_logger.add_scalar("loss", loss, global_step=step)

                    if step % self.config.training.logging_freq == 0 or step == 1:
                        logging.info(
                            (
                                f"During epoch: {epoch}, step: {step}, "
                                f"Noise Estimation loss: {loss.item()}, "
                                f"data time: {data_time / (i + 1)}"
                            )
                        )

                    # optimize diffusion model that predicts eps_theta
                    optimizer.zero_grad()
                    loss.backward()

                    if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext"):
                        try:
                            torch.nn.utils.clip_grad_norm_(
                                noise_estimator.parameters(),
                                config.optim.grad_clip
                            )
                        except Exception:
                            pass
                    else:
                        raise NotImplementedError

                    # backward 后立即清理激活值的显存
                    torch.cuda.empty_cache()

                    optimizer.step()

                    # optimizer.step() 后清理，释放梯度占用的显存
                    torch.cuda.empty_cache()

                    # 保存 loss 值用于 epoch 结束时的日志（在删除前）
                    last_loss_value = loss.item()

                    # 清理中间变量以节省显存（只删除不再需要的变量）
                    del loss, noise_output, y_t_batch, e
                    if step % 10 == 0:  # 每 10 步清理一次
                        torch.cuda.empty_cache()
                        gc.collect()
                    if self.config.model.ema:
                        if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext"):
                            # EMA update 前清理显存，确保有足够空间
                            torch.cuda.empty_cache()
                            if torch.cuda.is_available() and self.device.type == 'cuda' and step % 100 == 0:
                                device_id = self.device.index if self.device.index is not None else 0
                                allocated_before = torch.cuda.memory_allocated(self.device) / 1024 ** 3
                                print(f"[显存监控] EMA update 前 (step {step}, GPU {device_id}): {allocated_before:.2f} GB")
                            ema_helper_noise_estimator.update(noise_estimator)
                            # EMA update 后再次清理
                            torch.cuda.empty_cache()
                            if torch.cuda.is_available() and self.device.type == 'cuda' and step % 100 == 0:
                                device_id = self.device.index if self.device.index is not None else 0
                                allocated_after = torch.cuda.memory_allocated(self.device) / 1024 ** 3
                                print(f"[显存监控] EMA update 后 (step {step}, GPU {device_id}): {allocated_after:.2f} GB")
                        else:
                            raise NotImplementedError

                    # save diffusion model
                    # if step % self.config.training.snapshot_freq == 0 or step == 1:
                    #     states = [
                    #         model.state_dict(),
                    #         optimizer.state_dict(),
                    #         epoch,
                    #         step,
                    #     ]
                    #     if self.config.model.ema:
                    #         states.append(ema_helper.state_dict())
                    #
                    #     if step > 1:  # skip saving the initial ckpt
                    #         torch.save(
                    #             states,
                    #             os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    #         )
                    #     # save current states
                    #     torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    #
                    #     # save auxiliary model
                    #     if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                    #         aux_states = [
                    #             self.cond_pred_model.state_dict(),
                    #             aux_optimizer.state_dict(),
                    #         ]
                    #         if step > 1:  # skip saving the initial ckpt
                    #             torch.save(
                    #                 aux_states,
                    #                 os.path.join(self.args.log_path, "aux_ckpt_{}.pth".format(step)),
                    #             )
                    #         torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

                    data_start = time.time()

                logging.info(
                    (f"epoch: {epoch}, step: {step}, " +
                     f"Noise Estimation loss: {last_loss_value}, " +
                     f"data time: {data_time / (i + 1)}")
                )

                # Evaluate
                if epoch % self.config.training.validation_freq == 0 \
                        or epoch + 1 == self.config.training.n_epochs:

                    if config.data.dataset in ['ChestXRay', 'ISICSkinCancer']:
                        if self.config.diffusion.aux_cls.arch in ('sevit', 'swin', 'transnext'):
                            noise_estimator.eval()
                            self.cond_pred_model['vit'].eval()
                            for ii in range(len(self.cond_pred_model['mlps'])):
                                if self.cond_pred_model['mlps'][ii] is not None:
                                    self.cond_pred_model['mlps'][ii].eval()
                        else:
                            raise NotImplementedError

                        if self.config.diffusion.aux_cls.arch in ('sevit', 'swin', 'transnext'):
                            acc_avg = 0.0
                        else:
                            raise NotImplementedError

                        for test_batch_idx, feature_label_set in enumerate(valid_loader):
                            images_raw, target, nam_batch, region_batch = _unpack_guide_batch(feature_label_set)
                            if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext") and self.config.data.dataset in [
                                'ChestXRay', 'ISICSkinCancer']:
                                images_224 = images_raw.clone().to(self.device)
                                nam_dev = nam_batch.to(self.device) if nam_batch is not None else None
                                region_dev = _region_to_device(region_batch, self.device)
                            else:
                                raise NotImplementedError

                            if config.data.dataset == "toy" \
                                    or config.model.arch == "simple" \
                                    or config.model.arch == "linear":
                                images_224_flat = torch.flatten(images_224, 1).to(self.device)

                            target = target.to(self.device)

                            with torch.no_grad():
                                if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext"):
                                    # target_pred is a list
                                    with torch.no_grad():
                                        target_pred = self.compute_guiding_prediction(
                                            images_224, nam=nam_dev, region=region_dev
                                        )
                                    # apply softmax to logits（跳过 None 值）
                                    tmp = []
                                    for entry in target_pred:
                                        if entry is not None:
                                            tmp.append(nn.functional.softmax(entry, dim=1).to(self.device))
                                        else:
                                            tmp.append(None)
                                    target_pred = tmp
                                else:
                                    raise NotImplementedError

                                y_T_mean = target_pred

                                if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext"):
                                    label_t_0_seq = []
                                    with torch.no_grad():
                                        label_t_0_seq = p_sample_loop(noise_estimator, images_224_flat,
                                                                      target_pred[mlp_idx],
                                                                      y_T_mean[mlp_idx],
                                                                      self.num_timesteps, self.alphas,
                                                                      self.one_minus_alphas_bar_sqrt,
                                                                      only_last_sample=True)
                                    # accuracy in %, not in decimal
                                    acc_avg += accuracy(label_t_0_seq.detach().cpu(), target.cpu())[0].item()
                                else:
                                    raise NotImplementedError

                        if self.config.diffusion.aux_cls.arch in ("sevit", "swin", "transnext"):
                            acc_avg /= (test_batch_idx + 1)

                            if acc_avg > max_accuracy:
                                logging.info("Update accuracy at Epoch {}.".format(epoch))
                                # save diffusion models
                                states = {'noise_estimator': noise_estimator.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'epoch': epoch}
                                path_tmp = os.path.join(self.args.log_path,
                                                        "diffu{}_ckpt_best_eph{}_acc{:.4f}.pth".format(mlp_idx, epoch,
                                                                                                       acc_avg))
                                torch.save(states, path_tmp)
                                print("Saved diffusion model {} at {}".format(mlp_idx, path_tmp))

                            max_accuracy = max(max_accuracy, acc_avg)
                        else:
                            raise NotImplementedError

                        if not tb_logger is None:
                            tb_logger.add_scalar('accuracy', acc_avg, global_step=step)
                        logging.info(
                            (
                                    f"epoch: {epoch}, step: {step}, " +
                                    f"Average diffusion {mlp_idx} accuracy: {acc_avg}%, " +
                                    f"Max diffusion {mlp_idx} accuracy: {max_accuracy:.2f}%, "
                            )
                        )

            # save the model after training is finished
            # states = [
            #     model.state_dict(),
            #     optimizer.state_dict(),
            #     epoch,
            #     step,
            # ]
            # if self.config.model.ema:
            #     states.append(ema_helper.state_dict())
            # torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

    def test_image_task(self):
        """
        Evaluate model performance on image classification tasks.
        """

        def compute_val_before_softmax(gen_y_0_val):
            """
            Compute raw value before applying Softmax function to obtain prediction in probability scale.
            Corresponding to the part inside the Softmax function of Eq. (10) in paper.
            """
            # TODO: add other ways of computing such raw prob value
            raw_prob_val = -(gen_y_0_val - 1) ** 2
            return raw_prob_val

        #####################################################################################################
        ########################## local functions within the class function scope ##########################
        def compute_and_store_cls_metrics(config, y_labels_batch, generated_y, batch_size, num_t):
            """
            generated_y: y_t in logit, has a shape of (batch_size x n_samples, n_classes)

            For each instance, compute probabilities of prediction of each label, majority voted label and
                its correctness, as well as accuracy of all samples given the instance.
            """
            current_t = self.num_timesteps + 1 - num_t
            gen_y_all_class_raw_probs = generated_y.reshape(batch_size,
                                                            config.testing.n_samples,
                                                            config.data.num_classes).cpu()  # .numpy()
            # compute softmax probabilities of all classes for each sample
            raw_prob_val = compute_val_before_softmax(gen_y_all_class_raw_probs)
            gen_y_all_class_probs = torch.softmax(raw_prob_val / self.tuned_scale_T,
                                                  dim=-1)  # (batch_size, n_samples, n_classes)
            # obtain credible interval of probability predictions in each class for the samples given the same x
            low, high = config.testing.PICP_range
            # use raw predicted probability (right before temperature scaling and Softmax) width
            # to construct prediction interval
            CI_y_pred = raw_prob_val.nanquantile(q=torch.tensor([low / 100, high / 100]),
                                                 dim=1).swapaxes(0, 1)  # (batch_size, 2, n_classes)
            # obtain the predicted label with the largest probability for each sample
            gen_y_labels = torch.argmax(gen_y_all_class_probs, 2, keepdim=True)  # (batch_size, n_samples, 1)
            # convert the predicted label to one-hot format
            gen_y_one_hot = torch.zeros_like(gen_y_all_class_probs).scatter_(
                dim=2, index=gen_y_labels,
                src=torch.ones_like(gen_y_labels.float()))  # (batch_size, n_samples, n_classes)
            # compute proportion of each class as the prediction given the same x
            gen_y_label_probs = gen_y_one_hot.sum(1) / config.testing.n_samples  # (batch_size, n_classes)
            gen_y_all_class_mean_prob = gen_y_all_class_probs.mean(1)  # (batch_size, n_classes)
            # obtain the class being predicted the most given the same x
            gen_y_majority_vote = torch.argmax(gen_y_label_probs, 1, keepdim=True)  # (batch_size, 1)
            # compute the proportion of predictions being the correct label for each x
            gen_y_instance_accuracy = (gen_y_labels == y_labels_batch[:, None]).float().mean(1)  # (batch_size, 1)
            # conduct paired two-sided t-test for the two most predicted classes for each instance
            two_most_probable_classes_idx = gen_y_label_probs.argsort(dim=1, descending=True)[:, :2]
            two_most_probable_classes_idx = torch.repeat_interleave(
                two_most_probable_classes_idx[:, None],
                repeats=config.testing.n_samples, dim=1)  # (batch_size, n_samples, 2)
            gen_y_2_class_probs = torch.gather(gen_y_all_class_probs, dim=2,
                                               index=two_most_probable_classes_idx)  # (batch_size, n_samples, 2)
            # make plots to check normality assumptions (differences btw two most probable classes) for the t-test
            # (check https://www.researchgate.net/post/Paired_t-test_and_normality_test_question)
            if num_t == (self.num_timesteps + 1) and step == 0:
                gen_y_2_class_prob_diff = gen_y_2_class_probs[:, :, 1] \
                                          - gen_y_2_class_probs[:, :, 0]  # (batch_size, n_samples)
                plt.style.use('classic')
                for instance_idx in range(24):
                    fig = sm.qqplot(gen_y_2_class_prob_diff[instance_idx, :],
                                    fit=True, line='45')
                    fig.savefig(os.path.join(args.im_path,
                                             f'qq_plot_instance_{instance_idx}.png'))
                    plt.close()
                plt.style.use('ggplot')
            ttest_pvalues = (ttest_rel(gen_y_2_class_probs[:, :, 0],
                                       gen_y_2_class_probs[:, :, 1],
                                       axis=1, alternative='two-sided')).pvalue  # (batch_size, )
            ttest_reject = (ttest_pvalues < config.testing.ttest_alpha)  # (batch_size, )

            if len(majority_vote_by_batch_list[current_t]) == 0:
                majority_vote_by_batch_list[current_t] = gen_y_majority_vote
            else:
                majority_vote_by_batch_list[current_t] = np.concatenate(
                    [majority_vote_by_batch_list[current_t], gen_y_majority_vote], axis=0)
            if current_t == config.testing.metrics_t:
                gen_y_all_class_probs = gen_y_all_class_probs.reshape(
                    y_labels_batch.shape[0] * config.testing.n_samples, config.data.num_classes)
                if len(label_probs_by_batch_list[current_t]) == 0:
                    all_class_probs_by_batch_list[current_t] = gen_y_all_class_probs
                    label_probs_by_batch_list[current_t] = gen_y_label_probs
                    label_mean_probs_by_batch_list[current_t] = gen_y_all_class_mean_prob
                    instance_accuracy_by_batch_list[current_t] = gen_y_instance_accuracy
                    CI_by_batch_list[current_t] = CI_y_pred
                    ttest_reject_by_batch_list[current_t] = ttest_reject
                else:
                    all_class_probs_by_batch_list[current_t] = np.concatenate(
                        [all_class_probs_by_batch_list[current_t], gen_y_all_class_probs], axis=0)
                    label_probs_by_batch_list[current_t] = np.concatenate(
                        [label_probs_by_batch_list[current_t], gen_y_label_probs], axis=0)
                    label_mean_probs_by_batch_list[current_t] = np.concatenate(
                        [label_mean_probs_by_batch_list[current_t], gen_y_all_class_mean_prob], axis=0)
                    instance_accuracy_by_batch_list[current_t] = np.concatenate(
                        [instance_accuracy_by_batch_list[current_t], gen_y_instance_accuracy], axis=0)
                    CI_by_batch_list[current_t] = np.concatenate(
                        [CI_by_batch_list[current_t], CI_y_pred], axis=0)
                    ttest_reject_by_batch_list[current_t] = np.concatenate(
                        [ttest_reject_by_batch_list[current_t], ttest_reject], axis=0)

        def p_sample_loop_with_eval(model, x, y_0_hat, y_T_mean, n_steps,
                                    alphas, one_minus_alphas_bar_sqrt,
                                    batch_size, config):
            """
            Sample y_{t-1} given y_t on the fly, and evaluate model immediately, to avoid OOM.
            """

            def optional_metric_compute(cur_y, num_t):
                if config.testing.compute_metric_all_steps or \
                        (self.num_timesteps + 1 - num_t == config.testing.metrics_t):
                    compute_and_store_cls_metrics(config, y_labels_batch, cur_y, batch_size, num_t)

            device = next(model.parameters()).device
            z = torch.randn_like(y_T_mean).to(device)  # standard Gaussian
            cur_y = z + y_T_mean  # sampled y_T
            num_t = 1
            optional_metric_compute(cur_y, num_t)
            for i in reversed(range(1, n_steps)):
                y_t = cur_y
                cur_y = p_sample(model, x, y_t, y_0_hat, y_T_mean, i, alphas, one_minus_alphas_bar_sqrt)  # y_{t-1}
                num_t += 1
                optional_metric_compute(cur_y, num_t)
            assert num_t == n_steps
            # obtain y_0 given y_1
            num_t += 1
            y_0 = p_sample_t_1to0(model, x, cur_y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt)
            optional_metric_compute(y_0, num_t)

        def p_sample_loop_only_y_0(model, x, y_0_hat, y_T_mean, n_steps,
                                   alphas, one_minus_alphas_bar_sqrt):
            """
            Only compute y_0 -- no metric evaluation.
            """
            device = next(model.parameters()).device
            z = torch.randn_like(y_T_mean).to(device)  # standard Gaussian
            cur_y = z + y_T_mean  # sampled y_T
            num_t = 1
            for i in reversed(range(1, n_steps)):
                y_t = cur_y
                cur_y = p_sample(model, x, y_t, y_0_hat, y_T_mean, i, alphas, one_minus_alphas_bar_sqrt)  # y_{t-1}
                num_t += 1
            assert num_t == n_steps
            y_0 = p_sample_t_1to0(model, x, cur_y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt)
            return y_0

        def compute_quantile_metrics(config, CI_all_classes, majority_voted_class, true_y_label):
            """
            CI_all_classes: (n_test, 2, n_classes), contains quantiles at (low, high) in config.testing.PICP_range
                for the predicted probabilities of all classes for each test instance.

            majority_voted_class: (n_test, ) contains whether the majority-voted label is correct or not
                for each test instance.

            true_y_label: (n_test, ) contains true label for each test instance.
            """
            # obtain credible interval width by computing high - low
            CI_width_all_classes = torch.tensor(
                CI_all_classes[:, 1] - CI_all_classes[:, 0]).squeeze()  # (n_test, n_classes)
            # predict by the k-th smallest CI width
            for kth_smallest in range(1, config.data.num_classes + 1):
                pred_by_narrowest_CI_width = torch.kthvalue(
                    CI_width_all_classes, k=kth_smallest, dim=1, keepdim=False).indices.numpy()  # (n_test, )
                # logging.info(pred_by_narrowest_CI_width)  #@#
                narrowest_CI_pred_correctness = (pred_by_narrowest_CI_width == true_y_label)  # (n_test, )
                logging.info(("We predict the label by the class with the {}-th narrowest CI width, \n" +
                              "and obtain a test accuracy of {:.4f}% through the entire test set.").format(
                    kth_smallest, np.mean(narrowest_CI_pred_correctness) * 100))
            # check whether the most predicted class is the correct label for each x
            majority_vote_correctness = (majority_voted_class == true_y_label)  # (n_test, )
            # obtain one-hot label
            true_y_one_hot = cast_label_to_one_hot_and_prototype(torch.tensor(true_y_label), config,
                                                                 return_prototype=False)  # (n_test, n_classes)
            # obtain predicted CI width only for the true class
            CI_width_true_class = (CI_width_all_classes * true_y_one_hot).sum(dim=1, keepdim=True)  # (n_test, 1)
            # sanity check
            nan_idx = torch.arange(true_y_label.shape[0])[CI_width_true_class.flatten().isnan()]
            if nan_idx.shape[0] > 0:
                logging.info(("Sanity check: model prediction contains nan " +
                              "for test instances with index {}.").format(nan_idx.numpy()))
            CI_width_true_class_correct_pred = CI_width_true_class[majority_vote_correctness]  # (n_correct, 1)
            CI_width_true_class_incorrect_pred = CI_width_true_class[~majority_vote_correctness]  # (n_incorrect, 1)
            logging.info(("\n\nWe apply the majority-voted class label as our prediction, and achieve {:.4f}% " +
                          "accuracy through the entire test set.\n" +
                          "Out of {} test instances, we made {} correct predictions, with a " +
                          "mean credible interval width of {:.4f} in predicted probability of the true class; \n" +
                          "the remaining {} instances are classified incorrectly, " +
                          "with a mean CI width of {:.4f}.\n").format(
                np.mean(majority_vote_correctness) * 100,
                true_y_label.shape[0],
                majority_vote_correctness.sum(),
                CI_width_true_class_correct_pred.mean().item(),
                true_y_label.shape[0] - majority_vote_correctness.sum(),
                CI_width_true_class_incorrect_pred.mean().item()))

            maj_vote_acc_by_class = []
            CI_w_cor_pred_by_class = []
            CI_w_incor_pred_by_class = []
            # report metrics within each class
            for c in range(config.data.num_classes):
                maj_vote_cor_class_c = majority_vote_correctness[true_y_label == c]  # (n_class_c, 1)
                CI_width_true_class_c = CI_width_true_class[true_y_label == c]  # (n_class_c, 1)
                CI_w_cor_class_c = CI_width_true_class_c[maj_vote_cor_class_c]  # (n_correct_class_c, 1)
                CI_w_incor_class_c = CI_width_true_class_c[~maj_vote_cor_class_c]  # (n_incorrect_class_c, 1)
                logging.info(("\n\tClass {} ({} total instances, {:.4f}% accuracy):" +
                              "\n\t\t{} correct predictions, mean CI width {:.4f}" +
                              "\n\t\t{} incorrect predictions, mean CI width {:.4f}").format(
                    c, maj_vote_cor_class_c.shape[0], np.mean(maj_vote_cor_class_c) * 100,
                    CI_w_cor_class_c.shape[0], CI_w_cor_class_c.mean().item(),
                    CI_w_incor_class_c.shape[0], CI_w_incor_class_c.mean().item()))
                maj_vote_acc_by_class.append(np.mean(maj_vote_cor_class_c))
                CI_w_cor_pred_by_class.append(CI_w_cor_class_c.mean().item())
                CI_w_incor_pred_by_class.append(CI_w_incor_class_c.mean().item())

            return maj_vote_acc_by_class, CI_w_cor_pred_by_class, CI_w_incor_pred_by_class

        def compute_ttest_metrics(config, ttest_reject, majority_voted_class, true_y_label):
            """
            ttest_reject: (n_test, ) contains whether to reject the paired two-sided t-test between
                the two most probable predicted classes for each test instance.

            majority_voted_class: (n_test, ) contains whether the majority-voted label is correct or not
                for each test instance.

            true_y_label: (n_test, ) contains true label for each test instance.
            """
            # check whether the most predicted class is the correct label for each x
            majority_vote_correctness = (majority_voted_class == true_y_label)  # (n_test, )
            # split test instances into correct and incorrect predictions
            ttest_reject_correct_pred = ttest_reject[majority_vote_correctness]  # (n_correct, )
            ttest_reject_incorrect_pred = ttest_reject[~majority_vote_correctness]  # (n_incorrect, )
            logging.info(("\n\nWe apply the majority-voted class label as our prediction, and achieve {:.4f}% " +
                          "accuracy through the entire test set.\n" +
                          "Out of {} test instances, we made {} correct predictions, with a " +
                          "mean rejection rate of {:.4f}% for the paired two-sided t-test; \n" +
                          "the remaining {} instances are classified incorrectly, " +
                          "with a mean rejection rate of {:.4f}%.\n").format(
                np.mean(majority_vote_correctness) * 100,
                true_y_label.shape[0],
                ttest_reject_correct_pred.shape[0],
                ttest_reject_correct_pred.mean().item() * 100,
                ttest_reject_incorrect_pred.shape[0],
                ttest_reject_incorrect_pred.mean().item() * 100))

            # rejection rate by prediction correctness within each class
            maj_vote_acc_by_class = []
            ttest_rej_cor_pred_by_class = []
            ttest_rej_incor_pred_by_class = []
            for c in range(config.data.num_classes):
                maj_vote_cor_class_c = majority_vote_correctness[true_y_label == c]  # (n_class_c, 1)
                ttest_reject_class_c = ttest_reject[true_y_label == c]  # (n_class_c, 1)
                ttest_rej_cor_class_c = ttest_reject_class_c[maj_vote_cor_class_c]  # (n_correct_class_c, 1)
                ttest_rej_incor_class_c = ttest_reject_class_c[~maj_vote_cor_class_c]  # (n_incorrect_class_c, 1)
                logging.info(("\n\tClass {} ({} total instances, {:.4f}% accuracy):" +
                              "\n\t\t{} correct predictions, mean rejection rate {:.4f}%" +
                              "\n\t\t{} incorrect predictions, mean rejection rate {:.4f}%").format(
                    c, maj_vote_cor_class_c.shape[0], np.mean(maj_vote_cor_class_c) * 100,
                    ttest_rej_cor_class_c.shape[0], ttest_rej_cor_class_c.mean().item() * 100,
                    ttest_rej_incor_class_c.shape[0], ttest_rej_incor_class_c.mean().item() * 100))
                maj_vote_acc_by_class.append(np.mean(maj_vote_cor_class_c))
                ttest_rej_cor_pred_by_class.append(ttest_rej_cor_class_c.mean().item())
                ttest_rej_incor_pred_by_class.append(ttest_rej_incor_class_c.mean().item())

            # split test instances into rejected and not-rejected t-tests
            maj_vote_cor_reject = majority_vote_correctness[ttest_reject]  # (n_reject, )
            maj_vote_cor_not_reject = majority_vote_correctness[~ttest_reject]  # (n_not_reject, )
            logging.info(("\n\nFurthermore, among all test instances, " +
                          "we reject {} t-tests, with a " +
                          "mean accuracy of {:.4f}%; \n" +
                          "the remaining {} t-tests are not rejected, " +
                          "with a mean accuracy of {:.4f}%.\n").format(
                maj_vote_cor_reject.shape[0],
                maj_vote_cor_reject.mean().item() * 100,
                maj_vote_cor_not_reject.shape[0],
                maj_vote_cor_not_reject.mean().item() * 100))

            # accuracy by rejection status within each class
            ttest_reject_by_class = []
            maj_vote_cor_reject_by_class = []
            maj_vote_cor_not_reject_by_class = []
            for c in range(config.data.num_classes):
                maj_vote_cor_class_c = majority_vote_correctness[true_y_label == c]  # (n_class_c, 1)
                ttest_reject_class_c = ttest_reject[true_y_label == c]  # (n_class_c, 1)
                maj_vote_cor_rej_class_c = maj_vote_cor_class_c[ttest_reject_class_c]  # (n_rej_class_c, 1)
                maj_vote_cor_not_rej_class_c = maj_vote_cor_class_c[~ttest_reject_class_c]  # (n_not_rej_class_c, 1)
                logging.info(("\n\tClass {} ({} total instances, {:.4f}% rejection rate):" +
                              "\n\t\t{} rejected t-tests, mean accuracy {:.4f}%" +
                              "\n\t\t{} not-rejected t-tests, mean accuracy {:.4f}%").format(
                    c, ttest_reject_class_c.shape[0], np.mean(ttest_reject_class_c) * 100,
                    maj_vote_cor_rej_class_c.shape[0], maj_vote_cor_rej_class_c.mean().item() * 100,
                    maj_vote_cor_not_rej_class_c.shape[0], maj_vote_cor_not_rej_class_c.mean().item() * 100))
                ttest_reject_by_class.append(np.mean(ttest_reject_class_c))
                maj_vote_cor_reject_by_class.append(maj_vote_cor_rej_class_c.mean().item())
                maj_vote_cor_not_reject_by_class.append(maj_vote_cor_not_rej_class_c.mean().item())

            return maj_vote_acc_by_class, ttest_rej_cor_pred_by_class, ttest_rej_incor_pred_by_class, \
                   ttest_reject_by_class, maj_vote_cor_reject_by_class, maj_vote_cor_not_reject_by_class

        #####################################################################################################
        #####################################################################################################

        args = self.args
        config = self.config
        split = args.split
        log_path = os.path.join(self.args.log_path)
        dataset_object, train_dataset, test_dataset = get_dataset(args, config)
        # use test batch size for training set during inference
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        raise NotImplementedError

        # TODO
        model = NewConditionalModel(config, guidance=config.diffusion.include_guidance)
        if getattr(self.config.testing, "ckpt_id", None) is None:
            if args.eval_best:
                ckpt_id = 'best'
                states = torch.load(os.path.join(log_path, f"ckpt_{ckpt_id}.pth"),
                                    map_location=self.device)
            else:
                ckpt_id = 'last'
                states = torch.load(os.path.join(log_path, "ckpt.pth"),
                                    map_location=self.device)
        else:
            states = torch.load(os.path.join(log_path, f"ckpt_{self.config.testing.ckpt_id}.pth"),
                                map_location=self.device)
            ckpt_id = self.config.testing.ckpt_id
        logging.info(f"Loading from: {log_path}/ckpt_{ckpt_id}.pth")
        model = model.to(self.device)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None

        model.eval()
        if args.sanity_check:
            logging.info("Evaluation function implementation sanity check...")
            config.testing.n_samples = 10
        if args.test_sample_seed >= 0:
            logging.info(f"Manually setting seed {args.test_sample_seed} for test time sampling of class prototype...")
            set_random_seed(args.test_sample_seed)

        # load auxiliary model
        if config.diffusion.apply_aux_cls:
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path"):
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                                                     config.diffusion.trained_aux_cls_ckpt_name),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states['state_dict'], strict=True)
            else:
                aux_cls_path = log_path
                if hasattr(config.diffusion, "trained_aux_cls_log_path"):
                    aux_cls_path = config.diffusion.trained_aux_cls_log_path
                aux_states = torch.load(os.path.join(aux_cls_path, "aux_ckpt.pth"),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
            self.cond_pred_model.eval()
        # report test set RMSE if applied joint training
        y_acc_aux_model = self.evaluate_guidance_model(test_loader)
        logging.info("After training, guidance classifier accuracy on the test set is {:.8f}.".format(
            y_acc_aux_model))

        if config.testing.compute_metric_all_steps:
            logging.info("\nWe compute classification task metrics for all steps.\n")
        else:
            logging.info("\nWe pick samples at timestep t={} to compute evaluation metrics.\n".format(
                config.testing.metrics_t))

        # tune the scaling temperature parameter with training set
        T_description = "default"
        if args.tune_T:
            y_0_one_sample_all = []
            n_tune_T_samples = 5  # config.testing.n_samples; 25; 10
            logging.info("Begin generating {} samples for tuning temperature scaling parameter...".format(
                n_tune_T_samples))
            for idx, feature_label_set in tqdm(enumerate(train_loader)):  # test_loader would give oracle hyperparameter
                x_batch, y_labels_batch, nam_batch, region_batch = _unpack_guide_batch(feature_label_set)
                x_batch = x_batch.to(self.device)
                nam_dev = nam_batch.to(self.device) if nam_batch is not None else None
                region_dev = _region_to_device(region_batch, self.device)
                # compute y_0_hat as the initial prediction to guide the reverse diffusion process
                y_0_hat_list = self.compute_guiding_prediction(
                    x_batch, nam=nam_dev, region=region_dev
                )
                # 找到第一个非 None 的 MLP 输出（用于测试阶段）
                y_0_hat_batch = None
                for entry in y_0_hat_list:
                    if entry is not None:
                        y_0_hat_batch = entry
                        break
                if y_0_hat_batch is None:
                    raise ValueError("没有找到已加载的 MLP 输出！")
                y_0_hat_batch = y_0_hat_batch.softmax(dim=1)
                if config.data.dataset == "toy" or config.model.arch in ["simple", "linear"]:
                    x_batch = torch.flatten(x_batch, 1)
                batch_size = x_batch.shape[0]
                if len(x_batch.shape) == 2:
                    # x_batch with shape (batch_size, flattened_image_dim)
                    x_tile = (x_batch.repeat(n_tune_T_samples, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1)
                else:
                    # x_batch with shape (batch_size, 3, 32, 32) for CIFAR10 and CIFAR100 dataset
                    x_tile = (x_batch.repeat(n_tune_T_samples, 1, 1, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1)
                y_0_hat_tile = (y_0_hat_batch.repeat(n_tune_T_samples, 1, 1).transpose(0, 1)).flatten(0, 1)
                y_T_mean_tile = y_0_hat_tile
                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                    y_T_mean_tile = torch.zeros(y_0_hat_tile.shape).to(self.device)
                minibatch_sample_start = time.time()
                y_0_sample_batch = p_sample_loop_only_y_0(model, x_tile, y_0_hat_tile, y_T_mean_tile,
                                                          self.num_timesteps,
                                                          self.alphas, self.one_minus_alphas_bar_sqrt)  # .cpu().numpy()
                # take the mean of n_tune_T_samples reconstructed y prototypes as the raw predicted y for T tuning
                y_0_sample_batch = y_0_sample_batch.reshape(batch_size,
                                                            n_tune_T_samples,
                                                            config.data.num_classes).mean(1)  # (batch_size, n_classes)
                minibatch_sample_end = time.time()
                logging.info("Minibatch {} sampling took {:.4f} seconds.".format(
                    idx, (minibatch_sample_end - minibatch_sample_start)))
                y_0_one_sample_all.append(y_0_sample_batch.detach())
                # only generate a few batches for sanity checking
                if args.sanity_check:
                    if idx > 2:
                        break
            print(len(y_0_one_sample_all), y_0_one_sample_all[0].shape)

            logging.info("Begin optimizing temperature scaling parameter...")
            scale_T_raw = nn.Parameter(torch.zeros(1))
            scale_T_lr = 0.01
            scale_T_optimizer = torch.optim.Adam([scale_T_raw], lr=scale_T_lr)
            nll_losses = []
            scale_T_n_epochs = 10 if args.sanity_check else 1
            for _ in range(scale_T_n_epochs):
                for idx, feature_label_set in tqdm(enumerate(train_loader)):  # test_loader would give oracle value
                    _, y_labels_batch, _, _ = _unpack_guide_batch(feature_label_set)
                    y_one_hot_batch, _ = cast_label_to_one_hot_and_prototype(y_labels_batch, config)
                    y_one_hot_batch = y_one_hot_batch.to(self.device)
                    scale_T = nn.functional.softplus(scale_T_raw).to(self.device)
                    y_0_sample_batch = y_0_one_sample_all[idx]
                    raw_p_val = compute_val_before_softmax(y_0_sample_batch)
                    # Eq. (9) of the Calibration paper (Guo et al. 2017)
                    y_0_scaled_batch = (raw_p_val / scale_T).softmax(1)
                    y_0_prob_batch = (y_0_scaled_batch * y_one_hot_batch).sum(1)  # instance likelihood
                    nll_loss = -torch.log(y_0_prob_batch).mean()
                    nll_losses.append(nll_loss.cpu().item())
                    # optimize scaling temperature T parameter
                    scale_T_optimizer.zero_grad()
                    nll_loss.backward()
                    scale_T_optimizer.step()
                    # only tune a few batches for sanity checking
                    if args.sanity_check:
                        if idx > 2:
                            break
                logging.info("NLL of the last mini-batch: {:.8f}".format(nll_losses[-1]))
            self.tuned_scale_T = nn.functional.softplus(scale_T_raw).detach().item()
            T_description = "tuned"
        else:
            self.tuned_scale_T = 1
        logging.info("Apply {} temperature scaling parameter T with a value of {:.4f}".format(
            T_description, self.tuned_scale_T))

        with torch.no_grad():
            true_y_label_by_batch_list = []
            majority_vote_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            all_class_probs_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            label_probs_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            label_mean_probs_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            instance_accuracy_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            CI_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            ttest_reject_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]

            for step, feature_label_set in tqdm(enumerate(test_loader)):
                x_batch, y_labels_batch, nam_batch, region_batch = _unpack_guide_batch(feature_label_set)
                nam_dev = nam_batch.to(self.device) if nam_batch is not None else None
                region_dev = _region_to_device(region_batch, self.device)
                # compute y_0_hat as the initial prediction to guide the reverse diffusion process
                y_0_hat_list = self.compute_guiding_prediction(
                    x_batch.to(self.device), nam=nam_dev, region=region_dev
                )
                # 找到第一个非 None 的 MLP 输出（用于测试阶段）
                y_0_hat_batch = None
                for entry in y_0_hat_list:
                    if entry is not None:
                        y_0_hat_batch = entry
                        break
                if y_0_hat_batch is None:
                    raise ValueError("没有找到已加载的 MLP 输出！")
                y_0_hat_batch = y_0_hat_batch.softmax(dim=1)
                true_y_label_by_batch_list.append(y_labels_batch.numpy())
                # # record unflattened x as input to guidance aux classifier
                # x_unflat_batch = x_batch.to(self.device)
                if config.data.dataset == "toy" \
                        or config.model.arch == "simple" \
                        or config.model.arch == "linear":
                    x_batch = torch.flatten(x_batch, 1)
                # y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch, config)
                y_labels_batch = y_labels_batch.reshape(-1, 1)
                batch_size = x_batch.shape[0]
                if len(x_batch.shape) == 2:
                    # x_batch with shape (batch_size, flattened_image_dim)
                    x_tile = (x_batch.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1)
                else:
                    # x_batch with shape (batch_size, 3, 32, 32) for CIFAR10 and CIFAR100 dataset
                    x_tile = (x_batch.repeat(config.testing.n_samples, 1, 1, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1)
                y_0_hat_tile = (y_0_hat_batch.repeat(config.testing.n_samples, 1, 1).transpose(0, 1)).flatten(0, 1)
                y_T_mean_tile = y_0_hat_tile
                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                    y_T_mean_tile = torch.zeros(y_0_hat_tile.shape).to(self.device)
                # generate samples from all time steps for the current mini-batch
                p_sample_loop_with_eval(model, x_tile, y_0_hat_tile, y_T_mean_tile, self.num_timesteps,
                                        self.alphas, self.one_minus_alphas_bar_sqrt,
                                        batch_size, config)
                # only compute on a few batches for sanity checking
                if args.sanity_check:
                    if step > 0:  # first two mini-batches
                        break

        ################## compute metrics on test set ##################
        all_true_y_labels = np.concatenate(true_y_label_by_batch_list, axis=0).reshape(-1, 1)
        y_majority_vote_accuracy_all_steps_list = []

        if config.testing.compute_metric_all_steps:
            for idx in range(self.num_timesteps + 1):
                current_t = self.num_timesteps - idx
                # compute accuracy at time step t
                majority_voted_class_t = majority_vote_by_batch_list[current_t]
                majority_vote_correctness = (majority_voted_class_t == all_true_y_labels)  # (n_test, 1)
                y_majority_vote_accuracy = np.mean(majority_vote_correctness)
                y_majority_vote_accuracy_all_steps_list.append(y_majority_vote_accuracy)
            logging.info(
                f"Majority Vote Accuracy across all steps: {y_majority_vote_accuracy_all_steps_list}.\n")
        else:
            # compute accuracy at time step metrics_t
            majority_voted_class_t = majority_vote_by_batch_list[config.testing.metrics_t]
            majority_vote_correctness = (majority_voted_class_t == all_true_y_labels)  # (n_test, 1)
            y_majority_vote_accuracy = np.mean(majority_vote_correctness)
            y_majority_vote_accuracy_all_steps_list.append(y_majority_vote_accuracy)
        instance_accuracy_t = instance_accuracy_by_batch_list[config.testing.metrics_t]
        logging.info("Mean accuracy of all samples at test instance level is {:.4f}%.\n".format(
            np.mean(instance_accuracy_t) * 100))
        logging.info("\nNow we compute metrics related to predicted probability quantiles for all classes...")
        CI_all_classes_t = CI_by_batch_list[config.testing.metrics_t]  # (n_test, 2, n_classes)
        majority_vote_t = majority_vote_by_batch_list[config.testing.metrics_t]  # (n_test, 1)
        majority_vote_accuracy_by_class, \
        CI_width_correct_pred_by_class, \
        CI_width_incorrect_pred_by_class = compute_quantile_metrics(
            config, CI_all_classes_t, majority_vote_t.flatten(), all_true_y_labels.flatten())
        # print(CI_all_classes_t[159])  #@#
        # print(majority_vote_t[159])  #@#
        # print(all_true_y_labels[159])  #@#

        logging.info("\nNow we compute metrics related to paired two sample t-test for all classes...")
        ttest_reject_t = ttest_reject_by_batch_list[config.testing.metrics_t]  # (n_test, )
        _ = compute_ttest_metrics(config, ttest_reject_t, majority_vote_t.flatten(), all_true_y_labels.flatten())

        logging.info("\nNow we compute PAvPU based on paired two sample t-test results...")
        if config.testing.compute_metric_all_steps:
            # compute accuracy at time step metrics_t
            majority_voted_class_t = majority_vote_by_batch_list[config.testing.metrics_t]
            majority_vote_correctness = (majority_voted_class_t == all_true_y_labels)  # (n_test, 1)
        majority_vote_incorrectness = ~majority_vote_correctness  # (n_test, 1)
        n_ac = majority_vote_correctness[ttest_reject_t].sum()
        n_au = majority_vote_correctness[~ttest_reject_t].sum()
        n_ic = majority_vote_incorrectness[ttest_reject_t].sum()
        n_iu = majority_vote_incorrectness[~ttest_reject_t].sum()
        PAvPU = (n_ac + n_iu) / (n_ac + n_au + n_ic + n_iu)
        logging.info("\n\tCount of accurate and certain: {}".format(n_ac))
        logging.info("\n\tCount of accurate and uncertain: {}".format(n_au))
        logging.info("\n\tCount of inaccurate and certain: {}".format(n_ic))
        logging.info("\n\tCount of inaccurate and uncertain: {}".format(n_iu))
        logging.info(("\nWe obtain a PAvPU of {:.8f} with an alpha level of {:.4f} " +
                      "for the test set with size {}\n\n").format(
            PAvPU, config.testing.ttest_alpha, majority_vote_correctness.shape[0]))

        logging.info("\nNow we compute NLL and ECE for the test set...")
        label_probs_t = label_probs_by_batch_list[config.testing.metrics_t]  # (n_test, n_classes)
        label_mean_prob_t = label_mean_probs_by_batch_list[config.testing.metrics_t]  # (n_test, n_classes)
        n_test = label_probs_t.shape[0]
        logging.info("\nTest set size: {}".format(n_test))
        # predicted probability corresponding to the true class
        gen_y_true_label_pred_prob_t = torch.tensor(label_mean_prob_t).gather(
            dim=1, index=torch.tensor(all_true_y_labels))  # label_probs_t could result in inf
        NLL_t = - torch.log(gen_y_true_label_pred_prob_t).mean()
        logging.info("\nWe obtain an NLL of {:.8f} for the test set with size {}\n".format(NLL_t, n_test))
        # confidence (predicted probability corresponding to the predicted class)
        gen_y_max_label_prob_t = torch.tensor(label_mean_prob_t).gather(dim=1,
                                                                        index=torch.tensor(majority_vote_t))
        # gen_y_max_label_prob_t = torch.tensor(label_probs_t).gather(dim=1,
        #                                                             index=torch.tensor(majority_vote_t))
        hist_t = torch.histogram(gen_y_max_label_prob_t.flatten(), bins=10, range=(0, 1))
        bin_weights_t = hist_t.hist / n_test
        bin_edges = hist_t.bin_edges[1:]
        # bin membership based on confidence
        bin_membership_t = ((gen_y_max_label_prob_t - bin_edges) >= 0).sum(dim=1)  # (n_test, )
        # accuracy
        # gen_y_majority_vote_correct = torch.tensor(majority_vote_correctness).float()  # (n_test, 1)
        gen_y_majority_vote_correct = torch.tensor(majority_vote_t == all_true_y_labels).float()  # (n_test, 1)
        # compute ECE (Expected Calibration Error)
        calibration_error_t = []
        for bin_idx in range(config.testing.n_bins):
            confidence_bin_t = (bin_membership_t == bin_idx)
            if confidence_bin_t.sum() > 0:
                bin_accuracy = gen_y_majority_vote_correct[confidence_bin_t].mean()
                bin_confidence = gen_y_max_label_prob_t[confidence_bin_t].mean()
                calibration_error_t.append(torch.abs(bin_accuracy - bin_confidence))
            else:
                calibration_error_t.append(0)
        calibration_error_t = torch.tensor(calibration_error_t)
        ECE_t = (bin_weights_t * calibration_error_t).sum()
        # @#
        # print(gen_y_max_label_prob_t.shape, bin_membership_t.shape,
        #       gen_y_majority_vote_correct.shape, calibration_error_t.shape, bin_weights_t.shape)
        # print(calibration_error_t)
        # print(bin_weights_t)
        # @#
        logging.info("\nWe obtain an ECE of {:.8f} for the test set with size {}\n\n".format(ECE_t, n_test))

        # make plot
        if config.testing.compute_metric_all_steps:
            n_metrics = 1
            fig, axs = plt.subplots(n_metrics, 1, figsize=(8.5, n_metrics * 3.5), clear=True)  # W x H
            plt.subplots_adjust(hspace=0.5)
            # majority vote accuracy
            axs.plot(y_majority_vote_accuracy_all_steps_list)
            axs.set_title('Majority Vote Top 1 Accuracy across All Timesteps (Reversed)', fontsize=14)
            axs.set_xlabel('timestep (reversed)', fontsize=12)
            axs.set_ylabel('majority vote accuracy', fontsize=12)
            fig.savefig(os.path.join(args.im_path,
                                     'top_1_test_accuracy_all_timesteps.pdf'))

        n_metrics = 1
        all_classes = np.arange(config.data.num_classes)
        fig, axs = plt.subplots(n_metrics, 1, figsize=(8, n_metrics * 3.6), clear=True)  # W x H
        plt.subplots_adjust(hspace=0.5)
        # majority vote accuracy for each class
        axs.plot(all_classes, majority_vote_accuracy_by_class, label="class_acc")
        axs.plot(all_classes, CI_width_correct_pred_by_class, label="CI_w_cor")
        axs.plot(all_classes, CI_width_incorrect_pred_by_class, label="CI_w_incor")
        axs.set_title('Majority Vote Top 1 Accuracy with \nCredible Interval Width ' +
                      'of Correct and Incorrect Predictions', fontsize=9)
        axs.set_xlabel('Class Label', fontsize=8)
        axs.set_ylabel('Class Probability', fontsize=8)
        axs.set_ylim([0, 1])
        axs.legend(loc='best')
        fig.savefig(os.path.join(args.im_path,
                                 'accuracy_and_CI_width_by_class.pdf'))

        # clear the memory
        plt.close('all')
        del label_probs_by_batch_list
        del majority_vote_by_batch_list
        del instance_accuracy_by_batch_list
        gc.collect()

        return y_majority_vote_accuracy_all_steps_listvote_accuracy_all_steps_listll_steps_list