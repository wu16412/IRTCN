#!/usr/bin/env python3
"""Train TextureClassifier on NAM WLD-144 npz (training + validation)."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MAP_DIR = Path(__file__).resolve().parent
if str(MAP_DIR) not in sys.path:
    sys.path.insert(0, str(MAP_DIR))

from texture_classifier import TextureClassifier


def load_split(feat_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(feat_dir / f"{name}.npz", allow_pickle=True)
    x = np.asarray(data["lbp"], dtype=np.float32)
    y = np.asarray(data["labels"], dtype=np.int64)
    return x, y


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def balanced_acc(model: TextureClassifier, x: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[float, float]:
    model.eval()
    logits = model(torch.from_numpy(x).to(device))
    pred = logits.argmax(dim=1).cpu().numpy()
    nll = float(F.cross_entropy(logits, torch.from_numpy(y).to(device)).item())
    pos = y == 1
    neg = ~pos
    sens = float((pred[pos] == 1).mean()) if pos.any() else 0.0
    spec = float((pred[neg] == 0).mean()) if neg.any() else 0.0
    return 0.5 * (sens + spec), nll


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=4444)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    train_x, train_y = load_split(args.feature_dir, "training")
    val_x, val_y = load_split(args.feature_dir, "validation")

    model = TextureClassifier(input_dim=int(train_x.shape[1])).to(device)
    weights = class_weights(train_y).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_score, best_nll = -1.0, float("inf")
    stale = 0
    rng = np.random.RandomState(args.seed)
    n = len(train_x)

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = torch.from_numpy(train_x[idx]).to(device)
            yb = torch.from_numpy(train_y[idx]).to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb, weight=weights).backward()
            opt.step()
        score, nll = balanced_acc(model, val_x, val_y, device)
        better = score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and nll < best_nll)
        if better:
            best_score, best_nll = score, nll
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(f"epoch {epoch + 1:03d} val_bacc={score:.4f} nll={nll:.4f} best={best_score:.4f}", flush=True)
        if stale >= args.patience:
            break

    if best_state is None:
        raise SystemExit("no checkpoint")
    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "model_config": model.config,
            "class_to_idx": {"Normal": 0, "Tuberculosis": 1},
            "positive_class": "Tuberculosis",
            "selection": {
                "metric": "balanced_acc",
                "tie_break": "nll",
                "best_score": best_score,
                "best_nll": best_nll,
                "split": "validation",
            },
        },
        args.output,
    )
    print(f"wrote {args.output} bacc={best_score:.4f}")


if __name__ == "__main__":
    main()
