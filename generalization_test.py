#!/usr/bin/env python3
"""
generalization_test.py
──────────────────────
Leave-one-scenario-out 일반화 테스트

특정 시나리오를 학습 데이터에서 완전히 제외하고 재학습한 후,
해당 시나리오 전체 구간으로 테스트하여 미학습 시나리오에 대한 일반화 성능 측정.

테스트 시나리오 (4개 선택):
  - corridor_baseline  : 가장 긴 시나리오, 직선 이동 패턴
  - cluster_spread     : 군집 분산 패턴
  - relay_handover     : relay 전환이 빈번한 패턴
  - random_waypoint_1  : 무작위 이동 패턴

출력:
  - 콘솔 결과 테이블
  - models/generalization_test.png

실행:
    python3 generalization_test.py
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"

FEATURES    = ["rssi_dbm_est", "snr_db_est", "plr_pct_est", "throughput_mbps_est",
               "distance_m", "hop_count", "blocked_building_count"]
WINDOW_SIZE = 20
INPUT_SIZE  = len(FEATURES)
BATCH_SIZE  = 64
MAX_EPOCHS  = 50
PATIENCE    = 10
LR          = 1e-3

TEST_SCENARIOS = [
    "cluster_spread",
    "converge_diverge",
    "corridor_baseline",
    "orbit",
    "random_waypoint_1",
    "random_waypoint_2",
    "random_waypoint_3",
    "relay_competition",
    "relay_handover",
    "relay_rotation",
    "relay_stretch",
    "relay_uav4_handover",
    "relay_uav4_rotation",
    "slow_separation",
]


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


# ── 데이터 준비 ───────────────────────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_windows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """시나리오+링크 페어 단위 슬라이딩 윈도우."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["scenario_id"], r["src_uav"], r["dst_uav"])].append(r)

    X_list, y_list = [], []
    for key, grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))
        features = np.array([[float(r[f]) for f in FEATURES] for r in grp_sorted],
                             dtype=np.float32)
        labels   = np.array([int(r["link_state"]) for r in grp_sorted], dtype=np.int64)
        for i in range(len(grp_sorted) - WINDOW_SIZE + 1):
            X_list.append(features[i:i + WINDOW_SIZE])
            y_list.append(labels[i + WINDOW_SIZE - 1])

    if not X_list:
        return np.empty((0, WINDOW_SIZE, INPUT_SIZE)), np.empty((0,), dtype=np.int64)
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)


# ── 학습 ──────────────────────────────────────────────────────────────────────
def train_leave_one_out(held_out: str, device: torch.device) -> dict:
    # train + val 전체를 학습 데이터로 사용, held_out 시나리오 제외
    all_rows = load_csv(DATA_DIR / "train.csv") + load_csv(DATA_DIR / "val.csv")
    train_rows = [r for r in all_rows if r["scenario_id"] != held_out]
    test_rows  = (load_csv(DATA_DIR / "train.csv") +
                  load_csv(DATA_DIR / "val.csv") +
                  load_csv(DATA_DIR / "test.csv"))
    test_rows  = [r for r in test_rows if r["scenario_id"] == held_out]

    X_tr, y_tr = make_windows(train_rows)
    X_te, y_te = make_windows(test_rows)

    print(f"  train: {X_tr.shape}  test(held-out): {X_te.shape}")

    # class weights
    counts  = np.bincount(y_tr, minlength=3)
    weights = len(y_tr) / (3.0 * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True)

    model   = LinkStateTransformer().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    X_va_t = torch.tensor(X_tr[-len(X_tr)//5:]).to(device)  # 학습 데이터 끝 20%를 임시 val
    y_va_t = torch.tensor(y_tr[-len(y_tr)//5:]).to(device)

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), y_va_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    X_te_t = torch.tensor(X_te).to(device)
    with torch.no_grad():
        preds = model(X_te_t).argmax(1).cpu().numpy()

    acc = (preds == y_te).mean()
    f1  = f1_score(y_te, preds, average="macro", zero_division=0)
    f1_per = f1_score(y_te, preds, average=None, labels=[0, 1, 2], zero_division=0)

    return {
        "scenario": held_out,
        "acc":      acc,
        "f1_macro": f1,
        "f1_per":   f1_per.tolist(),
        "n_test":   len(y_te),
    }


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_results(results: list[dict], baseline_acc: float, baseline_f1: float) -> None:
    scenarios = [r["scenario"].replace("_", "\n") for r in results]
    accs = [r["acc"] * 100 for r in results]
    f1s  = [r["f1_macro"] * 100 for r in results]

    x = np.arange(len(scenarios))
    w = 0.35
    colors_acc = ["#2196F3"] * len(scenarios)
    colors_f1  = ["#FF5722"] * len(scenarios)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy",  color=colors_acc, alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s,  w, label="F1 (macro)", color=colors_f1,  alpha=0.85)

    # baseline 점선
    ax.axhline(baseline_acc * 100, color="#2196F3", linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"Baseline Acc ({baseline_acc*100:.1f}%)")
    ax.axhline(baseline_f1 * 100,  color="#FF5722", linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"Baseline F1 ({baseline_f1*100:.1f}%)")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Generalization Test — Leave-One-Scenario-Out (Transformer)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / "generalization_test.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # 기존 모델 baseline 성능
    baseline_model = LinkStateTransformer()
    baseline_model.load_state_dict(
        torch.load(OUT_DIR / "best_transformer.pt", map_location=device))
    baseline_model.to(device).eval()

    X_te_base = torch.tensor(np.load(DATA_DIR / "X_test.npy"), dtype=torch.float32).to(device)
    y_te_base = np.load(DATA_DIR / "y_state_test.npy")
    with torch.no_grad():
        base_preds = baseline_model(X_te_base).argmax(1).cpu().numpy()
    baseline_acc = (base_preds == y_te_base).mean()
    baseline_f1  = f1_score(y_te_base, base_preds, average="macro", zero_division=0)
    print(f"\nBaseline (original test set): Acc={baseline_acc*100:.2f}%  F1={baseline_f1*100:.2f}%")

    results = []
    for scenario in TEST_SCENARIOS:
        print(f"\n{'='*55}")
        print(f"  Held-out: {scenario}")
        print(f"{'='*55}")
        r = train_leave_one_out(scenario, device)
        results.append(r)
        print(f"  → Acc: {r['acc']*100:.2f}%  F1: {r['f1_macro']*100:.2f}%  "
              f"(healthy={r['f1_per'][0]*100:.1f}% "
              f"degraded={r['f1_per'][1]*100:.1f}% "
              f"disconnected={r['f1_per'][2]*100:.1f}%)")

    # 요약
    print(f"\n{'='*65}")
    print(f"  {'Held-out Scenario':<25} {'Acc':>8} {'F1 macro':>10} {'n_test':>8}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['scenario']:<25} {r['acc']*100:>7.2f}%  "
              f"{r['f1_macro']*100:>9.2f}%  {r['n_test']:>8}")
    avg_acc = np.mean([r["acc"] for r in results])
    avg_f1  = np.mean([r["f1_macro"] for r in results])
    print(f"  {'Average':<25} {avg_acc*100:>7.2f}%  {avg_f1*100:>9.2f}%")
    print(f"  {'Baseline (same scenarios)':<25} {baseline_acc*100:>7.2f}%  "
          f"{baseline_f1*100:>9.2f}%")
    print(f"  {'Δ (generalization gap)':<25} "
          f"{(avg_acc-baseline_acc)*100:>+7.2f}%p  "
          f"{(avg_f1-baseline_f1)*100:>+9.2f}%p")
    print(f"{'='*65}")

    plot_results(results, baseline_acc, baseline_f1)
    print("Done.")


if __name__ == "__main__":
    main()
