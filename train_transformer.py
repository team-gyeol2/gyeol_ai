#!/usr/bin/env python3
"""
train_transformer.py
────────────────────
Transformer 기반 link_state 3분류 예측 모델

LSTM과 동일한 조건으로 비교:
  - 입력: (batch, 20, 5) — 5초 시계열, 5개 feature
  - 출력: link_state (healthy/degraded/disconnected)
  - Optimizer: Adam lr=0.001, batch=64, epochs=50, early stopping patience=10

Transformer 구조:
  - Positional Encoding + Transformer Encoder 2층
  - d_model=64, nhead=4, dim_feedforward=128, dropout=0.2
  - 마지막 타임스텝 출력 → Linear(64, 3)

실행:
    python3 train_transformer.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"
OUT_DIR.mkdir(exist_ok=True)

INPUT_SIZE   = 7
D_MODEL      = 64
NHEAD        = 4       # attention head 수 (D_MODEL의 약수)
NUM_LAYERS   = 2
DIM_FF       = 128     # feedforward 내부 차원
DROPOUT      = 0.2
LR           = 1e-3
BATCH_SIZE   = 64
MAX_EPOCHS   = 50
PATIENCE     = 10


# ── Positional Encoding ───────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """시퀀스 내 위치 정보를 sinusoidal 방식으로 추가."""
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, : x.size(1), :])


# ── Transformer 모델 ──────────────────────────────────────────────────────────
class LinkStateTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # feature 차원(5) → d_model(64) 투영
        self.input_proj = nn.Linear(INPUT_SIZE, D_MODEL)
        self.pos_enc    = PositionalEncoding(D_MODEL, dropout=DROPOUT)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.dropout = nn.Dropout(DROPOUT)
        self.head    = nn.Linear(D_MODEL, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)       # (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)      # (batch, seq_len, d_model)
        last = self.dropout(x[:, -1, :])  # 마지막 타임스텝
        return self.head(last)


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_split(split: str):
    X       = torch.tensor(np.load(DATA_DIR / f"X_{split}.npy"),      dtype=torch.float32)
    y_state = torch.tensor(np.load(DATA_DIR / f"y_state_{split}.npy"), dtype=torch.long)
    return TensorDataset(X, y_state)


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


# ── 학습 루프 ─────────────────────────────────────────────────────────────────
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

    model     = LinkStateTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
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

        train_loss, _ = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.append({"epoch": epoch,
                        "train_loss": round(train_loss, 4),
                        "val_loss":   round(val_loss, 4),
                        "val_acc":    round(val_acc, 4)})

        print(f"{epoch:<6} {train_loss:>10.4f} {val_loss:>10.4f} {val_acc:>8.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(model.state_dict(), OUT_DIR / "best_transformer.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(OUT_DIR / "best_transformer.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'─'*40}")
    print(f"[TEST]  loss:{test_loss:.4f}  acc:{test_acc*100:.2f}%")

    with open(OUT_DIR / "transformer_history.json", "w") as f:
        json.dump({"history": history,
                   "test_loss": round(test_loss, 4),
                   "test_acc":  round(test_acc, 4)}, f, indent=2)

    print(f"모델 저장: models/best_transformer.pt")
    print(f"히스토리: models/transformer_history.json")


if __name__ == "__main__":
    main()
