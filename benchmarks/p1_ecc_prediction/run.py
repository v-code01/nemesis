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


def compute_f1(preds: np.ndarray, labels: np.ndarray, horizon: int) -> float:
    y_pred = (preds[:, horizon] > 0.5).astype(int)
    y_true = labels[:, horizon].astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return float(2 * precision * recall / (precision + recall + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    X, y = generate_dataset(n_healthy=2_000, n_failing=200, seed=args.seed)

    n = len(X)
    n_train, n_val = int(0.8 * n), int(0.9 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_val:], y[n_val:]

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=256, shuffle=True,
    )

    model = EccPredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((3,), 10.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    model.train()
    for _ in range(30):
        for Xb, yb in train_dl:
            optimizer.zero_grad()
            # TCN output → head logits (before sigmoid) for BCEWithLogitsLoss
            logits = model.head(model.tcn(Xb.transpose(1, 2))[:, :, -1])
            criterion(logits, yb).backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test)).numpy()

    result = {
        "f1_1h": round(compute_f1(preds, y_test, 0), 4),
        "f1_2h": round(compute_f1(preds, y_test, 1), 4),
        "f1_3h": round(compute_f1(preds, y_test, 2), 4),
        "seed": args.seed,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
