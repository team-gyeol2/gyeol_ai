#!/usr/bin/env python3
"""
train_relay.py
──────────────
relay 분류기: 네트워크 스냅샷을 보고 최적 relay_uav (0~4) 예측

입력: (N, 51) — 10쌍 × 5 feature + LSTM 예측값 1개
출력: optimal_relay_uav (5분류)

모델: 3층 MLP (51 → 128 → 64 → 5)
학습: Adam lr=0.001, batch=32, epochs=100, early stopping patience=15

실행:
    python3 train_relay.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"
OUT_DIR.mkdir(exist_ok=True)

LR         = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE   = 15


class RelayClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(51, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        return self.net(x)


def load_split(split: str):
    Xr      = torch.tensor(np.load(DATA_DIR / f"Xr_{split}.npy"),     dtype=torch.float32)
    y_relay = torch.tensor(np.load(DATA_DIR / f"y_relay_{split}.npy"), dtype=torch.long)
    return TensorDataset(Xr, y_relay)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(X)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(X)
    return total_loss / total, correct / total


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"디바이스: {device}")

    train_ds = load_split("train")
    val_ds   = load_split("val")
    test_ds  = load_split("test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    print(f"train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}")

    # relay class weight (역빈도)
    y_relay_train = np.load(DATA_DIR / "y_relay_train.npy")
    counts = np.bincount(y_relay_train, minlength=5).astype(np.float32)
    weights = torch.tensor(counts.sum() / (5 * counts), dtype=torch.float32).to(device)
    print(f"relay class weights: {[round(w,3) for w in weights.tolist()]}")

    model     = RelayClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n{'Epoch':<6} {'TrainLoss':>10} {'ValLoss':>10} {'ValAcc':>8}")
    print("─" * 40)

    best_val_loss = float("inf")
    patience_cnt  = 0
    history       = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader,   criterion, device)

        history.append({"epoch": epoch,
                        "train_loss": round(train_loss, 4),
                        "val_loss":   round(val_loss, 4),
                        "val_acc":    round(val_acc, 4)})

        print(f"{epoch:<6} {train_loss:>10.4f} {val_loss:>10.4f} {val_acc:>8.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(model.state_dict(), OUT_DIR / "best_relay.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(OUT_DIR / "best_relay.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'─'*40}")
    print(f"[TEST]  loss:{test_loss:.4f}  acc:{test_acc*100:.2f}%")

    with open(OUT_DIR / "relay_history.json", "w") as f:
        json.dump({"history": history,
                   "test_loss": round(test_loss, 4),
                   "test_acc":  round(test_acc, 4)}, f, indent=2)

    print(f"모델 저장: models/best_relay.pt")
    print(f"히스토리: models/relay_history.json")


if __name__ == "__main__":
    main()
