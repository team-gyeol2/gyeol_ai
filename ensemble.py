#!/usr/bin/env python3
"""
ensemble.py
───────────
LSTM + Transformer softmax 평균 앙상블 모델

LSTM과 Transformer 각각의 softmax 출력 확률값을 평균하여
최종 link_state (healthy/degraded/disconnected) 예측 생성.

비교 출력:
  - LSTM 단독 / Transformer 단독 / Ensemble
  - Accuracy, Precision, Recall, F1-score (macro)
  - Confusion Matrix

실행:
    python3 ensemble.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader, TensorDataset

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"

STATE_NAMES = ["healthy", "degraded", "disconnected"]


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
class LinkStateLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=7, hidden_size=64,
                               num_layers=2, dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.head    = nn.Linear(64, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class LinkStateTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(7, 64)
        self.pos_enc    = PositionalEncoding(64, dropout=0.2)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout     = nn.Dropout(0.2)
        self.head        = nn.Linear(64, 3)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.head(self.dropout(x[:, -1, :]))


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_split(split: str, device: torch.device):
    X = torch.tensor(np.load(DATA_DIR / f"X_{split}.npy"),       dtype=torch.float32).to(device)
    y = torch.tensor(np.load(DATA_DIR / f"y_state_{split}.npy"), dtype=torch.long).to(device)
    return X, y


# ── 평가 ─────────────────────────────────────────────────────────────────────
def get_preds(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            logits = model(X[i:i+256])
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def get_probs(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            logits = model(X[i:i+256])
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs)


def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc  = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score   (y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score       (y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print(f"\n{'─'*50}")
    print(f"  [{name}]")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%  (macro)")
    print(f"  Recall   : {rec*100:.2f}%  (macro)")
    print(f"  F1-score : {f1*100:.2f}%  (macro)")
    print(f"\n  Confusion Matrix (행=실제, 열=예측)")
    print(f"  {'':12s}  " + "  ".join(f"{s:>12s}" for s in STATE_NAMES))
    for i, row in enumerate(cm):
        print(f"  {STATE_NAMES[i]:12s}  " + "  ".join(f"{v:>12d}" for v in row))

    return {"accuracy": round(acc, 6), "precision": round(prec, 6),
            "recall": round(rec, 6), "f1": round(f1, 6)}


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"디바이스: {device}")

    lstm = LinkStateLSTM().to(device)
    transformer = LinkStateTransformer().to(device)
    lstm.load_state_dict(torch.load(OUT_DIR / "best_lstm.pt", map_location=device))
    transformer.load_state_dict(torch.load(OUT_DIR / "best_transformer.pt", map_location=device))

    # val set으로 최적 alpha(LSTM 비중) 탐색
    X_val, y_val = load_split("val", device)
    y_val_np     = y_val.cpu().numpy()
    prob_lstm_val  = get_probs(lstm, X_val)
    prob_trans_val = get_probs(transformer, X_val)

    best_alpha, best_f1 = 0.5, -1.0
    for alpha in np.arange(0.0, 1.01, 0.05):
        prob_mix = alpha * prob_lstm_val + (1 - alpha) * prob_trans_val
        pred_mix = prob_mix.argmax(axis=1)
        score    = f1_score(y_val_np, pred_mix, average="macro", zero_division=0)
        if score > best_f1:
            best_f1, best_alpha = score, alpha

    print(f"\n[가중 앙상블 최적 alpha 탐색]")
    print(f"  최적 alpha(LSTM 비중) : {best_alpha:.2f}  (Transformer 비중: {1-best_alpha:.2f})")
    print(f"  Val F1 (최적) : {best_f1*100:.2f}%")

    results = {}

    for split in ("val", "test"):
        X, y = load_split(split, device)
        y_np = y.cpu().numpy()

        print(f"\n{'='*50}")
        print(f"  {split.upper()} SET  (샘플 수: {len(X)})")
        print(f"{'='*50}")

        pred_lstm        = get_preds(lstm, X)
        pred_transformer = get_preds(transformer, X)

        prob_lstm  = get_probs(lstm, X)
        prob_trans = get_probs(transformer, X)

        # 단순 평균 앙상블
        pred_ensemble = ((prob_lstm + prob_trans) / 2).argmax(axis=1)
        # 가중 앙상블
        pred_weighted = (best_alpha * prob_lstm + (1 - best_alpha) * prob_trans).argmax(axis=1)

        m_lstm  = print_metrics("LSTM",             y_np, pred_lstm)
        m_trans = print_metrics("Transformer",      y_np, pred_transformer)
        m_ens   = print_metrics("Ensemble (평균)",  y_np, pred_ensemble)
        m_wens  = print_metrics(f"Ensemble (가중 α={best_alpha:.2f})", y_np, pred_weighted)

        best_single = max(m_lstm["accuracy"], m_trans["accuracy"])
        best_f1_single = max(m_lstm["f1"], m_trans["f1"])
        delta_acc = m_wens["accuracy"] - best_single
        delta_f1  = m_wens["f1"] - best_f1_single
        print(f"\n  [단독 최고 대비 가중 앙상블 개선]")
        print(f"  Accuracy : {delta_acc*100:+.2f}%p")
        print(f"  F1-score : {delta_f1*100:+.2f}%p")

        results[split] = {
            "lstm": m_lstm, "transformer": m_trans,
            "ensemble_avg": m_ens, "ensemble_weighted": m_wens,
            "best_alpha": best_alpha,
            "delta_accuracy": round(delta_acc, 6),
            "delta_f1": round(delta_f1, 6),
        }

    # 결과 저장
    out_path = OUT_DIR / "ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
