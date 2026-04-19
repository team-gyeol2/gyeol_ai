#!/usr/bin/env python3
"""
attention_visualization.py
──────────────────────────
Transformer Encoder Attention Weight 시각화 (보강 버전)

[A] 통계적 평균 attention — 상태별 100+ 샘플 평균으로 신뢰할 수 있는 패턴 도출
[B] 상태 전환 직전 attention — 단절 직전 모델이 어느 시점 신호를 보는지 분석

출력:
  - models/attention_avg_state.png      : 상태별 평균 attention 분포
  - models/attention_transition.png     : 상태 전환 직전 attention 비교
  - models/attention_heatmap.png        : 레이어별 × 상태별 평균 attention 히트맵

실행:
    python3 attention_visualization.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"

FEATURES     = ["rssi_dbm_est", "snr_db_est", "plr_pct_est", "throughput_mbps_est",
                "distance_m", "hop_count", "blocked_building_count"]
WINDOW_SIZE  = 20
INPUT_SIZE   = len(FEATURES)
STATE_NAMES  = ["Healthy", "Degraded", "Disconnected"]
STATE_COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]
MAX_SAMPLES  = 200   # 상태별 최대 샘플 수


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
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

    def get_attn_last_row(self, x: torch.Tensor) -> list[np.ndarray]:
        """
        각 레이어의 마지막 타임스텝(query=19) attention weight 반환.
        Returns: list of (nhead, seq_len) per layer
        """
        self.eval()
        with torch.no_grad():
            x = self.pos_enc(self.input_proj(x))
            last_rows = []
            for layer in self.transformer.layers:
                _, w = layer.self_attn(x, x, x,
                                       need_weights=True,
                                       average_attn_weights=False)
                # w: (batch, nhead, seq, seq) → 마지막 query 행
                last_rows.append(w[0, :, -1, :].cpu().numpy())  # (nhead, seq)
                x2 = layer.self_attn(x, x, x, need_weights=False)[0]
                x  = layer.norm1(x + layer.dropout1(x2))
                x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x  = layer.norm2(x + layer.dropout2(x2))
        return last_rows

    def get_full_attn(self, x: torch.Tensor) -> list[np.ndarray]:
        """각 레이어의 전체 attention 행렬 반환. Returns: list of (seq, seq) per layer"""
        self.eval()
        with torch.no_grad():
            x = self.pos_enc(self.input_proj(x))
            mats = []
            for layer in self.transformer.layers:
                _, w = layer.self_attn(x, x, x,
                                       need_weights=True,
                                       average_attn_weights=True)
                mats.append(w[0].cpu().numpy())  # (seq, seq) head 평균
                x2 = layer.self_attn(x, x, x, need_weights=False)[0]
                x  = layer.norm1(x + layer.dropout1(x2))
                x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x  = layer.norm2(x + layer.dropout2(x2))
        return mats


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_windows(device: torch.device):
    """
    Returns:
        state_windows: dict[state] → list of tensors (1, W, F)
        transition_windows: dict[(from, to)] → list of tensors
    """
    rows = (load_csv(DATA_DIR / "train.csv") +
            load_csv(DATA_DIR / "val.csv") +
            load_csv(DATA_DIR / "test.csv"))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["scenario_id"], r["src_uav"], r["dst_uav"])].append(r)

    state_windows: dict[int, list] = {0: [], 1: [], 2: []}
    trans_windows: dict[tuple, list] = {(0, 1): [], (1, 2): [], (2, 1): []}

    for (sid, src, dst), grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))
        n = len(grp_sorted)
        if n < WINDOW_SIZE:
            continue

        features = np.array([[float(r[f]) for f in FEATURES] for r in grp_sorted],
                             dtype=np.float32)
        labels   = [int(r["link_state"]) for r in grp_sorted]

        for i in range(n - WINDOW_SIZE + 1):
            cur_state = labels[i + WINDOW_SIZE - 1]
            window    = torch.tensor(features[i:i + WINDOW_SIZE]).unsqueeze(0).to(device)

            # 상태별 샘플 수집
            if len(state_windows[cur_state]) < MAX_SAMPLES:
                state_windows[cur_state].append(window)

            # 전환 직전 샘플: 현재 윈도우 끝이 전환 전 상태, 다음이 다른 상태
            if i + WINDOW_SIZE < n:
                next_state = labels[i + WINDOW_SIZE]
                key = (cur_state, next_state)
                if key in trans_windows and len(trans_windows[key]) < MAX_SAMPLES:
                    trans_windows[key].append(window)

    return state_windows, trans_windows


# ── attention 평균 계산 ───────────────────────────────────────────────────────
def avg_attn_last_row(model, windows: list, layer_idx: int) -> np.ndarray:
    """windows 리스트의 평균 last-row attention (head 평균). Returns: (seq,)"""
    accumulated = []
    for w in windows:
        last_rows = model.get_attn_last_row(w)
        accumulated.append(last_rows[layer_idx].mean(axis=0))  # head 평균 → (seq,)
    return np.stack(accumulated).mean(axis=0)  # 샘플 평균 → (seq,)


def avg_full_attn(model, windows: list, layer_idx: int) -> np.ndarray:
    """windows 리스트의 평균 전체 attention 행렬. Returns: (seq, seq)"""
    accumulated = []
    for w in windows:
        mats = model.get_full_attn(w)
        accumulated.append(mats[layer_idx])
    return np.stack(accumulated).mean(axis=0)


# ── 차트 A: 상태별 평균 attention 분포 ───────────────────────────────────────
def plot_avg_state(model, state_windows: dict) -> None:
    n_layers = len(model.transformer.layers)
    time_labels = [f"t-{WINDOW_SIZE-1-i}" if i < WINDOW_SIZE-1 else "t"
                   for i in range(WINDOW_SIZE)]

    fig, axes = plt.subplots(1, n_layers, figsize=(14, 5), sharey=False)
    fig.suptitle(
        "Average Attention from Final Timestep — Per State\n"
        f"(averaged over {MAX_SAMPLES} samples per state)",
        fontsize=12, fontweight="bold")

    for layer_idx, ax in enumerate(axes):
        for state in [0, 1, 2]:
            wins = state_windows[state]
            if not wins:
                continue
            avg = avg_attn_last_row(model, wins, layer_idx)
            ax.plot(range(WINDOW_SIZE), avg,
                    color=STATE_COLORS[state], label=f"{STATE_NAMES[state]} (n={len(wins)})",
                    linewidth=2.2, marker="o", markersize=3.5)

        ax.set_title(f"Layer {layer_idx+1}", fontsize=10)
        ax.set_xticks(range(0, WINDOW_SIZE, 2))
        ax.set_xticklabels([time_labels[i] for i in range(0, WINDOW_SIZE, 2)],
                           fontsize=7, rotation=45)
        ax.set_xlabel("Timestep", fontsize=9)
        ax.set_ylabel("Avg Attention Weight", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / "attention_avg_state.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 차트 B: 상태 전환 직전 attention 비교 ────────────────────────────────────
def plot_transition(model, state_windows: dict, trans_windows: dict) -> None:
    time_labels = [f"t-{WINDOW_SIZE-1-i}" if i < WINDOW_SIZE-1 else "t"
                   for i in range(WINDOW_SIZE)]
    layer_idx = 1   # Layer 2 — 전역 패턴 포착에 특화

    transitions = [
        ((0, 1), "Healthy→Degraded",       "#f39c12"),
        ((1, 2), "Degraded→Disconnected",  "#e74c3c"),
        ((2, 1), "Disconnected→Degraded",  "#9b59b6"),
    ]

    fig, axes = plt.subplots(1, len(transitions), figsize=(16, 5), sharey=False)
    fig.suptitle(
        "Attention Pattern RIGHT BEFORE State Transition (Layer 2)\n"
        "vs. Baseline (steady-state average)",
        fontsize=12, fontweight="bold")

    for ax, (key, label, color) in zip(axes, transitions):
        from_state, to_state = key
        wins = trans_windows.get(key, [])

        # 전환 직전 attention
        if wins:
            avg_trans = avg_attn_last_row(model, wins, layer_idx)
            ax.plot(range(WINDOW_SIZE), avg_trans,
                    color=color, linewidth=2.5, label=f"Pre-transition (n={len(wins)})",
                    marker="o", markersize=4)

        # 해당 상태의 steady-state baseline
        baseline_wins = state_windows[from_state]
        if baseline_wins:
            avg_base = avg_attn_last_row(model, baseline_wins, layer_idx)
            ax.plot(range(WINDOW_SIZE), avg_base,
                    color=STATE_COLORS[from_state], linewidth=1.5,
                    linestyle="--", label=f"Steady {STATE_NAMES[from_state]}", alpha=0.7)

        ax.set_title(label, fontsize=10, color=color, fontweight="bold")
        ax.set_xticks(range(0, WINDOW_SIZE, 2))
        ax.set_xticklabels([time_labels[i] for i in range(0, WINDOW_SIZE, 2)],
                           fontsize=7, rotation=45)
        ax.set_xlabel("Timestep (t = prediction moment)", fontsize=8)
        ax.set_ylabel("Avg Attention Weight", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        n_trans = len(wins)
        ax.text(0.97, 0.97, f"n={n_trans}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    plt.tight_layout()
    path = OUT_DIR / "attention_transition.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 차트 C: 레이어별 평균 히트맵 ─────────────────────────────────────────────
def plot_avg_heatmap(model, state_windows: dict) -> None:
    n_layers = len(model.transformer.layers)
    fig, axes = plt.subplots(3, n_layers, figsize=(n_layers * 5, 3 * 4))
    fig.suptitle("Average Attention Matrix — Per State & Layer",
                 fontsize=12, fontweight="bold")

    for row, state in enumerate([0, 1, 2]):
        wins = state_windows[state]
        for col in range(n_layers):
            ax = axes[row][col]
            if wins:
                avg = avg_full_attn(model, wins, col)
                im = ax.imshow(avg, aspect="auto", cmap="Blues", vmin=0, vmax=avg.max())
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Layer {col+1} | {STATE_NAMES[state]}\n(n={len(wins)})",
                         fontsize=9, color=STATE_COLORS[state], fontweight="bold")
            ax.set_xlabel("Key timestep", fontsize=7)
            ax.set_ylabel("Query timestep", fontsize=7)
            ax.set_xticks(range(0, WINDOW_SIZE, 4))
            ax.set_yticks(range(0, WINDOW_SIZE, 4))

    plt.tight_layout()
    path = OUT_DIR / "attention_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 차트 D: Attention Entropy 비교 ───────────────────────────────────────────
def attention_entropy(attn: np.ndarray) -> float:
    """(seq,) attention weight의 Shannon entropy. 클수록 분산, 작을수록 집중."""
    a = np.clip(attn, 1e-9, None)
    a = a / a.sum()
    return float(-np.sum(a * np.log(a)))


def plot_entropy(model, state_windows: dict) -> None:
    n_layers = len(model.transformer.layers)
    max_entropy = np.log(WINDOW_SIZE)   # 균등 분포 시 이론적 최대값

    # 레이어별, 상태별, 샘플별 entropy 수집
    # shape: [n_layers][state] → list of floats
    ent_data: list[dict[int, list]] = [{s: [] for s in [0, 1, 2]} for _ in range(n_layers)]

    for state in [0, 1, 2]:
        for w in state_windows[state]:
            last_rows = model.get_attn_last_row(w)
            for l_idx, head_attn in enumerate(last_rows):
                # head_attn: (nhead, seq) → 헤드 평균 후 entropy
                avg_attn = head_attn.mean(axis=0)
                ent_data[l_idx][state].append(attention_entropy(avg_attn))

    fig, axes = plt.subplots(1, n_layers, figsize=(10, 5), sharey=True)
    fig.suptitle(
        "Attention Entropy by State — Final Timestep Query\n"
        f"(higher = more diffuse, lower = more focused; max={max_entropy:.2f})",
        fontsize=12, fontweight="bold")

    x = np.arange(3)
    for l_idx, ax in enumerate(axes):
        means = [np.mean(ent_data[l_idx][s]) for s in [0, 1, 2]]
        stds  = [np.std(ent_data[l_idx][s])  for s in [0, 1, 2]]
        bars  = ax.bar(x, means, color=STATE_COLORS, alpha=0.85, width=0.5,
                       yerr=stds, capsize=5, error_kw={"elinewidth": 1.5})
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(max_entropy, color="gray", linestyle="--", linewidth=1,
                   label=f"Max entropy ({max_entropy:.2f})")
        ax.set_title(f"Layer {l_idx + 1}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(STATE_NAMES, fontsize=9)
        ax.set_ylabel("Shannon Entropy (nats)", fontsize=9)
        ax.set_ylim(0, max_entropy * 1.15)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        counts = [len(ent_data[l_idx][s]) for s in [0, 1, 2]]
        for xi, cnt in zip(x, counts):
            ax.text(xi, 0.05, f"n={cnt}", ha="center", va="bottom",
                    fontsize=7, color="white", fontweight="bold")

    plt.tight_layout()
    path = OUT_DIR / "attention_entropy.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    model = LinkStateTransformer()
    model.load_state_dict(torch.load(OUT_DIR / "best_transformer.pt", map_location=device))
    model.to(device)
    model.eval()

    print("샘플 수집 중...")
    state_windows, trans_windows = collect_windows(device)
    for s in [0, 1, 2]:
        print(f"  {STATE_NAMES[s]}: {len(state_windows[s])}개")
    for k, v in trans_windows.items():
        print(f"  전환 {STATE_NAMES[k[0]]}→{STATE_NAMES[k[1]]}: {len(v)}개")

    print("\n[A] 상태별 평균 attention 분포...")
    plot_avg_state(model, state_windows)

    print("[B] 상태 전환 직전 attention...")
    plot_transition(model, state_windows, trans_windows)

    print("[C] 평균 attention 히트맵...")
    plot_avg_heatmap(model, state_windows)

    print("[D] Attention Entropy 분석...")
    plot_entropy(model, state_windows)

    print("\nDone.")


if __name__ == "__main__":
    main()
