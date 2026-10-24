"""Safe runtime adapters for frozen Diffusion fusion guides."""
from __future__ import annotations

import importlib.util
import inspect
import os
import pickle
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent
GUIDE_INDEX = 3
BACKBONE_INDEX = 4
TEXTURE_INPUT_DIM = 144
TEXTURE_NUM_CLASSES = 2
FUSION_GUIDE_TYPES = ("none", "texture_only")
NAM_FEATURE_DIR_ENV = ("NAM_FEATURE_DIR", "IRA_NAM_FEATURE_DIR")
TEXTURE_CKPT_ENV = ("TEXTURE_CKPT", "IRA_TEXTURE_CKPT")


def _load_module(module_name: str, path: Path) -> ModuleType:
    """Import a source file directly, including files below hyphenated directories."""
    if not path.is_file():
        raise FileNotFoundError(f"fusion guide module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import fusion guide module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_FORBIDDEN_TORCH_PREFIXES = (
    "torch.nn",
    "torch.optim",
    "torch.distributed",
    "torch.multiprocessing",
    "torch.utils",
)
_FORBIDDEN_BUILTINS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "getattr",
        "setattr",
        "globals",
        "locals",
        "__import__",
        "breakpoint",
        "input",
        "memoryview",
    }
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Allow tensor rebuilds and primitives; never unpickle nn.Module or OS hooks.

    Used only when this PyTorch build has no torch.load(weights_only=True). It is
    not a fallback to unrestricted pickle.
    """

    def find_class(self, module: str, name: str):
        if module in {"builtins", "__builtin__"} and name in _FORBIDDEN_BUILTINS:
            raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")
        if any(module == prefix or module.startswith(prefix + ".") for prefix in _FORBIDDEN_TORCH_PREFIXES):
            raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")
        if module in {"posix", "nt", "os", "sys", "pathlib", "subprocess", "socket", "ctypes"}:
            raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")
        if module in {"builtins", "__builtin__", "collections", "copyreg", "copy_reg", "_codecs"}:
            return super().find_class(module, name)
        if module == "torch" or module.startswith("torch."):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"global {module}.{name} is forbidden in fusion checkpoints")


class _RestrictedPickleModule:
    Unpickler = _RestrictedUnpickler

    @staticmethod
    def load(*args: Any, **kwargs: Any):
        return _RestrictedUnpickler(*args, **kwargs).load()


def torch_load_supports_weights_only() -> bool:
    try:
        return "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        return False


def _load_tensor_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load tensor/config checkpoints without allowing arbitrary pickle objects."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"fusion guide checkpoint not found: {path}")
    if torch_load_supports_weights_only():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    else:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            pickle_module=_RestrictedPickleModule,
        )
    if isinstance(checkpoint, nn.Module):
        raise TypeError("refusing pickled nn.Module; fusion checkpoints must be state_dict + config")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a state/config dictionary, got {type(checkpoint)!r}")
    return checkpoint


def _constructor_config(model_class: type[nn.Module], raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"model config must be a dictionary, got {type(raw)!r}")
    valid = set(inspect.signature(model_class.__init__).parameters) - {"self"}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"unsupported {model_class.__name__} config fields: {sorted(unknown)}")
    return dict(raw)


def _freeze(model: nn.Module, device: torch.device) -> nn.Module:
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state", checkpoint.get("state_dict"))
    if not isinstance(state, dict) or not state:
        raise KeyError("fusion checkpoint must contain a non-empty model_state dictionary")
    invalid = [key for key, value in state.items() if not isinstance(key, str) or not torch.is_tensor(value)]
    if invalid:
        raise TypeError(f"fusion state_dict contains non-tensor entries: {invalid[:3]}")
    return state


def env_override_path(config_value: str | Path | None, env_names: Iterable[str]) -> str | None:
    """Prefer an environment variable over a config placeholder. Core code stays path-agnostic."""
    for name in env_names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    if config_value is None:
        return None
    text = str(config_value).strip()
    return text or None


def normalize_class_to_idx(mapping: Any) -> dict[str, int]:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("class_to_idx must be a non-empty dictionary")
    normalized = {str(key).strip().lower(): int(value) for key, value in mapping.items()}
    if sorted(normalized.values()) != list(range(len(normalized))):
        raise ValueError(f"class_to_idx indices must be 0..C-1, got {mapping}")
    return normalized


def assert_class_mapping(checkpoint_map: Any, dataset_map: Any) -> dict[str, int]:
    """Require ImageFolder labels to match the frozen TextureClassifier mapping."""
    checkpoint = normalize_class_to_idx(checkpoint_map)
    dataset = normalize_class_to_idx(dataset_map)
    if checkpoint != dataset:
        raise ValueError(
            f"class mapping mismatch: checkpoint={dict(checkpoint_map)} dataset={dict(dataset_map)}"
        )
    if len(checkpoint) != TEXTURE_NUM_CLASSES:
        raise ValueError(f"texture guide requires {TEXTURE_NUM_CLASSES} classes, got {checkpoint_map}")
    return checkpoint


_SPLIT_ACCESS: dict[str, set[str]] = {
    "array_splits": set(),
    "label_splits": set(),
}


def reset_split_access_log() -> None:
    for values in _SPLIT_ACCESS.values():
        values.clear()


def record_split_access(kind: str, split: str) -> None:
    if kind not in _SPLIT_ACCESS:
        raise ValueError(f"unknown access kind {kind!r}")
    _SPLIT_ACCESS[kind].add(_canonical_split(split))


def split_access_log() -> dict[str, list[str]]:
    return {key: sorted(values) for key, values in _SPLIT_ACCESS.items()}


def _canonical_split(split: str) -> str:
    alias = {
        "train": "train",
        "training": "train",
        "valid": "valid",
        "validation": "valid",
        "test": "test",
        "testing": "test",
    }
    key = str(split).strip().lower()
    if key not in alias:
        raise ValueError(f"unknown split {split!r}")
    return alias[key]


def assert_no_test_split_access() -> None:
    log = split_access_log()
    if "test" in log["array_splits"]:
        raise RuntimeError("texture_only development training loaded testing NAM arrays")
    if "test" in log["label_splits"]:
        raise RuntimeError("texture_only development training loaded testing labels")


class LockedSplit:
    """Stand-in that fails if train accidentally touches a locked split."""

    def __init__(self, split: str, reason: str) -> None:
        self.split = split
        self._reason = reason

    def _deny(self, *_args: Any, **_kwargs: Any):
        raise RuntimeError(self._reason)

    def __len__(self):
        return self._deny()

    def __getitem__(self, index):
        return self._deny(index)

    def __iter__(self):
        return self._deny()

    def __getattr__(self, name: str):
        raise RuntimeError(self._reason)


def resolve_fusion_dataset_splits(
    kind: str,
    *,
    is_test: bool,
    confirm_final_test: bool = False,
) -> tuple[str, ...]:
    """Development texture-only training uses train+validation only; test is locked."""
    kind = str(kind).strip().lower()
    if kind != "texture_only":
        return ("train", "valid", "test")
    if is_test:
        if not confirm_final_test:
            raise ValueError(
                "texture_only locked test requires --confirm-final-test; "
                "development training never reads the test split"
            )
        return ("test",)
    if confirm_final_test:
        raise ValueError("texture_only --confirm-final-test is only valid with --test")
    return ("train", "valid")


def fusion_guide_type_from_config(config: Any) -> str:
    aux_cls = getattr(getattr(config, "diffusion", None), "aux_cls", None)
    fusion_guide = getattr(aux_cls, "fusion_guide", None) if aux_cls is not None else None
    return str(getattr(fusion_guide, "type", "none")).strip().lower()


def texture_splits_from_args(args: Any) -> tuple[str, ...]:
    return resolve_fusion_dataset_splits(
        "texture_only",
        is_test=bool(getattr(args, "test", False)),
        confirm_final_test=bool(getattr(args, "confirm_final_test", False)),
    )


def _attach_texture_metadata(model: nn.Module, checkpoint: dict[str, Any], config: dict[str, Any]) -> nn.Module:
    model.guide_kind = "texture_only"
    model.guide_class_to_idx = checkpoint.get("class_to_idx")
    model.guide_selection = checkpoint.get("selection")
    model.guide_positive_class = checkpoint.get("positive_class")
    model.guide_input_dim = int(config.get("input_dim", TEXTURE_INPUT_DIM))
    model.guide_num_classes = int(config.get("num_classes", TEXTURE_NUM_CLASSES))
    return model


def _load_texture_classifier_class() -> tuple[type[nn.Module], Any]:
    """Load TextureClassifier without requiring the full dual-branch package."""
    candidates = (
        REPO_ROOT / "mapping" / "texture_classifier.py",
    )
    for path in candidates:
        if path.is_file():
            module = _load_module("_diffusion_texture_classifier", path)
            return module.TextureClassifier, module.texture_from_config
    raise FileNotFoundError(
        "TextureClassifier missing. Expected mapping/texture_classifier.py"
    )


def _assert_texture_config(config: dict[str, Any]) -> dict[str, Any]:
    input_dim = int(config.get("input_dim", TEXTURE_INPUT_DIM))
    num_classes = int(config.get("num_classes", TEXTURE_NUM_CLASSES))
    hidden = config.get("hidden_dims")
    if hidden is None or not tuple(hidden):
        raise ValueError("texture checkpoint model_config must include non-empty hidden_dims")
    if input_dim != TEXTURE_INPUT_DIM:
        raise ValueError(f"texture guide NAM dim must be {TEXTURE_INPUT_DIM}, got {input_dim}")
    if num_classes != TEXTURE_NUM_CLASSES:
        raise ValueError(f"texture guide must have {TEXTURE_NUM_CLASSES} logits, got {num_classes}")
    return config


def assert_nam_features(nam: torch.Tensor | None, input_dim: int = TEXTURE_INPUT_DIM) -> torch.Tensor:
    if nam is None:
        raise ValueError("texture_only guide requires NAM144 in the batch")
    if not torch.is_tensor(nam):
        raise TypeError(f"NAM must be a tensor, got {type(nam)!r}")
    if nam.ndim != 2 or nam.shape[1] != input_dim:
        raise ValueError(f"NAM features must be (B,{input_dim}), got {tuple(nam.shape)}")
    if not torch.isfinite(nam).all():
        raise ValueError("NAM features contain non-finite values")
    return nam


def assert_class_logits(logits: torch.Tensor, num_classes: int = TEXTURE_NUM_CLASSES) -> torch.Tensor:
    if not torch.is_tensor(logits):
        raise TypeError(f"logits must be a tensor, got {type(logits)!r}")
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(f"guide logits must be (B,{num_classes}), got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError("guide logits contain non-finite values")
    return logits


def load_fusion_guide(kind: str, checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    """Rebuild a supported guide from config + state_dict and freeze it."""
    kind = str(kind).strip().lower()
    checkpoint = _load_tensor_checkpoint(checkpoint_path)
    state = _state_dict(checkpoint)

    if kind == "dual_branch":
        raise ValueError("dual_branch guide is not included in this release")

    if kind == "texture_only":
        model_class, texture_from_config = _load_texture_classifier_class()
        config = _assert_texture_config(
            _constructor_config(model_class, checkpoint.get("model_config", checkpoint.get("config")))
        )
        model = texture_from_config(config)
        model.load_state_dict(state, strict=True)
        model = _attach_texture_metadata(model, checkpoint, config)
        if model.guide_class_to_idx is None:
            raise KeyError("texture checkpoint must contain class_to_idx for label-space checks")
        normalize_class_to_idx(model.guide_class_to_idx)
        return _freeze(model, device)

    if kind == "region_attention":
        raise ValueError("region_attention guide is not included in this release")
    raise ValueError("fusion guide type must be 'texture_only' (or 'none')")


def replace_guide_index(
    guides: list[torch.Tensor | None],
    fusion_logits: torch.Tensor,
    backbone_logits: torch.Tensor,
) -> list[torch.Tensor | None]:
    """Apply the five-route contract: fusion at 3, backbone logits at 4."""
    if len(guides) != 5:
        raise ValueError(f"five guide slots are required, got {len(guides)}")
    guides[GUIDE_INDEX] = fusion_logits
    guides[BACKBONE_INDEX] = backbone_logits
    return guides


@torch.no_grad()
def run_fusion_guide(
    kind: str,
    model: nn.Module,
    tokens: torch.Tensor | None,
    *,
    nam: torch.Tensor | None = None,
    region: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    kind = str(kind).strip().lower()
    if kind == "texture_only":
        input_dim = int(getattr(model, "guide_input_dim", TEXTURE_INPUT_DIM))
        num_classes = int(getattr(model, "guide_num_classes", TEXTURE_NUM_CLASSES))
        nam = assert_nam_features(nam, input_dim=input_dim)
        return assert_class_logits(model(nam), num_classes=num_classes)
    raise ValueError(f"unsupported fusion guide type: {kind!r} (supported: texture_only)")


@torch.no_grad()
def compute_fusion_guides(
    guides: list[torch.Tensor | None],
    backbone: nn.Module,
    x: torch.Tensor,
    extract_stage,
    kind: str,
    model: nn.Module,
    *,
    nam: torch.Tensor | None = None,
    region: dict[str, torch.Tensor] | None = None,
) -> list[torch.Tensor | None]:
    """Compute and install the fusion route without changing the backbone route."""
    kind = str(kind).strip().lower()
    if kind == "texture_only":
        fusion_logits = run_fusion_guide(kind, model, None, nam=nam, region=region)
    else:
        tokens = extract_stage(backbone, x, GUIDE_INDEX)
        fusion_logits = run_fusion_guide(kind, model, tokens, nam=nam, region=region)
    return replace_guide_index(guides, fusion_logits, backbone(x))
