"""P1 benchmark: train ECC predictor on synthetic data, evaluate F1 at all horizons."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agents"))
from nemesis.models.ecc_predictor import EccPredictor

sys.path.insert(0, str(Path(__file__).parent))
from synthetic import generate_dataset


def compute_f1(preds: np.ndarray, labels: np.ndarray, horizon: int, threshold: float = 0.5) -> float:
    y_pred = (preds[:, horizon] > threshold).astype(int)
    y_true = labels[:, horizon].astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return float(2 * precision * recall / (precision + recall + 1e-9))


def best_threshold(preds: np.ndarray, labels: np.ndarray, horizon: int) -> float:
    """Find threshold in [0.05, 0.95] that maximises F1 on the given split."""
    best_t, best_f = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        f = compute_f1(preds, labels, horizon, float(t))
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    X, y = generate_dataset(n_healthy=10_000, n_failing=1_000, seed=args.seed)

    n = len(X)
    n_test = int(0.2 * n)   # 2600 held-out test samples
    n_val = int(0.1 * n)    # 1300 for threshold calibration
    n_train = n - n_test - n_val  # 9100 training samples

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=256, shuffle=True,
    )

    # Per-horizon pos_weights from actual label distribution — class ratios differ
    # substantially: 1h≈12:1, 2h≈5.4:1, 3h≈3.3:1. A uniform weight of 10 over-weights
    # 2h/3h positives (causing high FP) and under-weights 1h positives.
    pos_counts = torch.from_numpy(y_train.sum(axis=0)).float()
    neg_counts = float(n_train) - pos_counts
    pw = (neg_counts / pos_counts).to(device)  # shape (3,) ≈ [12.0, 5.4, 3.3]

    model = EccPredictor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    model.train()
    for _ in range(30):
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            # TCN output → head logits (before sigmoid) for BCEWithLogitsLoss
            logits = model.head(model.tcn(Xb.transpose(1, 2))[:, :, -1])
            criterion(logits, yb).backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(torch.from_numpy(X_val).to(device)).cpu().numpy()
        preds = model(torch.from_numpy(X_test).to(device)).cpu().numpy()

    # Tune decision threshold per horizon on the held-out val split.
    # NEMESIS tunes threshold at deployment — fixed 0.5 is wrong when pos_weights
    # shift the model's calibration away from 50%.
    thresholds = [best_threshold(val_preds, y_val, h) for h in range(3)]

    result = {
        "f1_1h": round(compute_f1(preds, y_test, 0, thresholds[0]), 4),
        "f1_2h": round(compute_f1(preds, y_test, 1, thresholds[1]), 4),
        "f1_3h": round(compute_f1(preds, y_test, 2, thresholds[2]), 4),
        "seed": args.seed,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if result["f1_2h"] < 0.90:
        print(f"HARD GATE FAILED: f1_2h={result['f1_2h']} < 0.90", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
