#!/usr/bin/env python3
"""
plot_predictions.py
───────────────────
LSTM / Transformer 예측 결과 시각화

① pred_timeline.png  — 시나리오별 실제 vs 예측 step plot
② model_metrics.png  — 모델 성능 지표 비교 막대 차트
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"

FEATURES    = ["rssi_dbm_est", "snr_db_est", "plr_pct_est", "throughput_mbps_est",
               "distance_m", "hop_count", "blocked_building_count"]
WINDOW_SIZE = 20
STATE_NAMES = ["healthy", "degraded", "disconnected"]
STATE_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
INPUT_SIZE  = len(FEATURES)


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
class LinkStateLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=INPUT_SIZE, hidden_size=64,
                               num_layers=2, dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.head    = nn.Linear(64, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_len=100, dropout=0.1):
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
        return self.dropout(x + self.pe[:, :x.size(1), :])


class LinkStateTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj  = nn.Linear(INPUT_SIZE, 64)
        self.pos_enc     = PositionalEncoding(64, dropout=0.2)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.dropout     = nn.Dropout(0.2)
        self.head        = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        return self.head(self.dropout(self.transformer(x)[:, -1, :]))


# ── 예측 생성 ─────────────────────────────────────────────────────────────────
def get_predictions(model, device, scenario_id, src, dst):
    with open(DATA_DIR / "test.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["scenario_id"] == scenario_id
                and int(r["src_uav"]) == src
                and int(r["dst_uav"]) == dst]
    if not rows:
        return [], [], []

    rows     = sorted(rows, key=lambda r: float(r["time_s"]))
    features = np.array([[float(r[f]) for f in FEATURES] for r in rows], dtype=np.float32)
    labels   = [int(r["link_state"]) for r in rows]
    times    = [float(r["time_s"])   for r in rows]

    feat_t = torch.tensor(features).to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(rows) - WINDOW_SIZE + 1):
            window = feat_t[i:i + WINDOW_SIZE].unsqueeze(0)
            preds.append(int(model(window).argmax(1).item()))

    offset = WINDOW_SIZE - 1
    return times[offset:], labels[offset:], preds


def find_best_pair(device):
    """test.csv에서 3가지 상태가 골고루 나타나는 (scenario, src, dst) 탐색."""
    with open(DATA_DIR / "test.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        groups[(r["scenario_id"], int(r["src_uav"]), int(r["dst_uav"]))].append(r)

    best_key, best_score = None, -1
    for key, grp in groups.items():
        if len(grp) < WINDOW_SIZE:
            continue
        labels = [int(r["link_state"]) for r in grp]
        n_states = len(set(labels))
        # 3가지 상태가 있으면 우선, 그 다음 degraded 비율 중간인 것
        score = n_states * 100 - abs(labels.count(1) / len(labels) - 0.4) * 100
        if score > best_score:
            best_score = score
            best_key   = key

    return best_key  # (scenario_id, src, dst)


# ── 차트 1: step plot 타임라인 ────────────────────────────────────────────────
def plot_timeline(device):
    models_def = {
        "LSTM":        (LinkStateLSTM(),        OUT_DIR / "best_lstm.pt"),
        "Transformer": (LinkStateTransformer(), OUT_DIR / "best_transformer.pt"),
    }

    scenario_id, src, dst = find_best_pair(device)
    print(f"  선택된 시나리오: {scenario_id}  UAV{src}↔UAV{dst}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(
        f"Prediction Timeline — {scenario_id}  (UAV{src}↔UAV{dst})",
        fontsize=12, fontweight="bold")

    for ax, (name, (model, path)) in zip(axes, models_def.items()):
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)

        times, true_labels, pred_labels = get_predictions(
            model, device, scenario_id, src, dst)
        if not times:
            ax.set_title(f"{name} — no data")
            continue

        times = np.array(times)
        true  = np.array(true_labels)
        pred  = np.array(pred_labels)

        # 배경: 실제 상태를 색상 구간으로 표시
        for i in range(len(times)):
            t0 = times[i]
            t1 = times[i + 1] if i + 1 < len(times) else times[i] + 0.25
            ax.axvspan(t0, t1, color=STATE_COLORS[true[i]], alpha=0.25, linewidth=0)

        # 오분류 구간 하이라이트
        for i in range(len(times)):
            if true[i] != pred[i]:
                t0 = times[i]
                t1 = times[i + 1] if i + 1 < len(times) else times[i] + 0.25
                ax.axvspan(t0, t1, color="black", alpha=0.15, linewidth=0)

        # 실제 상태: step 함수 (굵은 선)
        ax.step(times, true, where="post", color="#2c3e50",
                linewidth=2.0, label="Actual", zorder=3)
        # 예측 상태: 점선
        ax.step(times, pred, where="post", color="#8e44ad",
                linewidth=1.5, linestyle="--", label="Predicted", zorder=4)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(STATE_NAMES, fontsize=9)
        ax.set_ylim(-0.3, 2.3)
        ax.set_title(f"{name}", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)

        # 정확도 표시
        acc = (true == pred).mean() * 100
        ax.text(0.99, 0.95, f"Acc: {acc:.1f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    axes[-1].set_xlabel("Time (s)")

    # 범례 (배경 색상)
    legend = [mpatches.Patch(color=STATE_COLORS[i], alpha=0.4, label=STATE_NAMES[i])
              for i in range(3)]
    legend.append(mpatches.Patch(color="black", alpha=0.15, label="mismatch"))
    fig.legend(handles=legend, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = OUT_DIR / "pred_timeline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 차트 2: 모델 지표 비교 ────────────────────────────────────────────────────
def plot_metrics():
    metrics = {
        "LSTM": {
            "Accuracy":  94.52,
            "Precision": 96.00,
            "Recall":    93.96,
            "F1-score":  94.71,
        },
        "Transformer": {
            "Accuracy":  99.74,
            "Precision": 99.78,
            "Recall":    99.81,
            "F1-score":  99.80,
        },
    }

    labels = list(list(metrics.values())[0].keys())
    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, [metrics["LSTM"][l]        for l in labels], w,
           label="LSTM",        color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, [metrics["Transformer"][l] for l in labels], w,
           label="Transformer", color="#FF5722", alpha=0.85)

    for i, label in enumerate(labels):
        for name, off in [("LSTM", -w/2), ("Transformer", w/2)]:
            val = metrics[name][label]
            ax.text(i + off, val + 0.5, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("LSTM vs Transformer — Test Performance (macro)", fontsize=12)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / "model_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print("Generating pred_timeline...")
    plot_timeline(device)
    print("Generating model_metrics...")
    plot_metrics()
    print("Done")


if __name__ == "__main__":
    main()
