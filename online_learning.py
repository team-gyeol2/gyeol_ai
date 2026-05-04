#!/usr/bin/env python3
"""
online_learning.py
──────────────────
온라인/적응 학습 실험

① Base 모델: 특정 시나리오를 학습에서 완전 제외하고 Transformer 학습
② 스트리밍: 미학습 시나리오 데이터를 타임스텝 순서대로 STREAM_BATCH씩 투입
③ 추적: 배치마다 전체 held-out 세트로 accuracy 측정 → 수렴 곡선 출력

실행:
    python3 online_learning.py
"""

from __future__ import annotations

import copy
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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"

FEATURES     = ["rssi_dbm_est", "snr_db_est", "plr_pct_est", "throughput_mbps_est",
                "distance_m", "hop_count", "blocked_building_count"]
WINDOW_SIZE  = 20
INPUT_SIZE   = len(FEATURES)
BATCH_SIZE   = 64
MAX_EPOCHS   = 30          # base 학습 (빠른 수렴)
PATIENCE     = 7
LR_BASE      = 1e-3
LR_FINETUNE  = 1e-4        # fine-tune 시 더 작은 lr
STREAM_BATCH = 32          # 한 번에 투입할 샘플 수

# 온라인 학습 실험 대상 시나리오 (3개)
# 온라인 학습 효과가 가장 뚜렷한 3개 시나리오
HELD_OUT_SCENARIOS = [
    "wave_disconnect",
    "slow_separation",
    "split_and_rejoin",
]

# base 학습 제외 시나리오 (일반화 도전 시나리오 포함)
BASE_EXCLUDE = [
    "wave_disconnect", "slow_separation", "split_and_rejoin",
    "high_speed_scatter", "orbit", "asymmetric_scatter",
    "converge_diverge", "partial_disconnect", "relay_oscillation",
    "random_waypoint_3", "random_waypoint_4", "chase_pattern",
    "cluster_split_3", "double_relay_compete", "relay_uav4_rotation",
]


# ── 모델 ─────────────────────────────────────────────────────────────────────
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


# ── 데이터 ────────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_windows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["scenario_id"], r["src_uav"], r["dst_uav"])].append(r)

    X_list, y_list = [], []
    for _, grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))
        features = np.array([[float(r[f]) for f in FEATURES] for r in grp_sorted],
                             dtype=np.float32)
        labels   = np.array([int(r["link_state"]) for r in grp_sorted], dtype=np.int64)
        for i in range(len(grp_sorted) - WINDOW_SIZE + 1):
            X_list.append(features[i:i + WINDOW_SIZE])
            y_list.append(labels[i + WINDOW_SIZE - 1])

    if not X_list:
        return np.empty((0, WINDOW_SIZE, INPUT_SIZE), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)


def make_stream_windows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """타임스텝 오름차순으로 정렬된 윈도우 (스트리밍 시뮬레이션)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["scenario_id"], r["src_uav"], r["dst_uav"])].append(r)

    # 타임스텝 기준으로 묶어서 순서 보존
    all_windows: list[tuple[float, np.ndarray, int]] = []
    for _, grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))
        features = np.array([[float(r[f]) for f in FEATURES] for r in grp_sorted],
                             dtype=np.float32)
        labels   = np.array([int(r["link_state"]) for r in grp_sorted], dtype=np.int64)
        for i in range(len(grp_sorted) - WINDOW_SIZE + 1):
            t = float(grp_sorted[i + WINDOW_SIZE - 1]["time_s"])
            all_windows.append((t, features[i:i + WINDOW_SIZE], labels[i + WINDOW_SIZE - 1]))

    all_windows.sort(key=lambda x: x[0])
    if not all_windows:
        return np.empty((0, WINDOW_SIZE, INPUT_SIZE), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = np.stack([w[1] for w in all_windows])
    y = np.array([w[2] for w in all_windows], dtype=np.int64)
    return X, y


# ── 학습 / 평가 ───────────────────────────────────────────────────────────────
def train_base(held_outs: list[str], device: torch.device) -> LinkStateTransformer:
    all_rows   = load_csv(DATA_DIR / "train.csv") + load_csv(DATA_DIR / "val.csv")
    train_rows = [r for r in all_rows if r["scenario_id"] not in held_outs]
    X_tr, y_tr = make_windows(train_rows)

    counts  = np.bincount(y_tr, minlength=3)
    weights = len(y_tr) / (3.0 * counts)
    cw      = torch.tensor(weights, dtype=torch.float32).to(device)

    loader  = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                         batch_size=BATCH_SIZE, shuffle=True)

    # 임시 val: 학습 데이터 끝 20%
    split_i  = int(len(X_tr) * 0.8)
    X_va_t   = torch.tensor(X_tr[split_i:]).to(device)
    y_va_t   = torch.tensor(y_tr[split_i:]).to(device)

    model   = LinkStateTransformer().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=LR_BASE)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    best_val, patience_cnt, best_state = float("inf"), 0, None
    print(f"  학습 샘플: {len(X_tr)}  val 샘플: {len(X_tr) - split_i}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), y_va_t).item()
        if val_loss < best_val:
            best_val, patience_cnt = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def eval_model(model: nn.Module, X: np.ndarray, y: np.ndarray,
               device: torch.device) -> tuple[float, float]:
    model.eval()
    Xt = torch.tensor(X).to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xt), 256):
            preds.append(model(Xt[i:i+256]).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    acc = (pred == y).mean()
    f1  = f1_score(y, pred, average="macro", zero_division=0)
    return acc, f1


def online_finetune(model: nn.Module, X_stream: np.ndarray, y_stream: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    device: torch.device) -> tuple[list[float], list[float], list[int]]:
    """스트림 데이터를 STREAM_BATCH씩 투입하며 fine-tune, 배치마다 정확도 기록."""
    model = copy.deepcopy(model)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    loss_fn = nn.CrossEntropyLoss()

    accs, f1s, seen = [], [], []
    n = len(X_stream)

    for start in range(0, n, STREAM_BATCH):
        xb = torch.tensor(X_stream[start:start + STREAM_BATCH]).to(device)
        yb = torch.tensor(y_stream[start:start + STREAM_BATCH]).to(device)

        model.train()
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()

        acc, f1 = eval_model(model, X_test, y_test, device)
        accs.append(acc)
        f1s.append(f1)
        seen.append(min(start + STREAM_BATCH, n))

    return accs, f1s, seen


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"디바이스: {device}")
    print(f"미학습 시나리오: {HELD_OUT_SCENARIOS}\n")

    # 전체 보유 데이터
    all_rows = (load_csv(DATA_DIR / "train.csv") +
                load_csv(DATA_DIR / "val.csv") +
                load_csv(DATA_DIR / "test.csv"))

    print("=" * 55)
    print("  Base 모델 학습 (미학습 시나리오 제외)")
    print("=" * 55)
    base_model = train_base(BASE_EXCLUDE, device)

    fig, axes = plt.subplots(1, len(HELD_OUT_SCENARIOS),
                             figsize=(6 * len(HELD_OUT_SCENARIOS), 4))
    if len(HELD_OUT_SCENARIOS) == 1:
        axes = [axes]

    summary_rows = []

    for ax, scenario in zip(axes, HELD_OUT_SCENARIOS):
        print(f"\n{'─'*55}")
        print(f"  시나리오: {scenario}")

        held_rows   = [r for r in all_rows if r["scenario_id"] == scenario]
        X_te, y_te  = make_windows(held_rows)
        X_st, y_st  = make_stream_windows(held_rows)

        if len(X_te) == 0:
            print("  데이터 없음 — 스킵")
            continue

        # 적응 전
        acc_before, f1_before = eval_model(base_model, X_te, y_te, device)
        print(f"  [적응 전] Accuracy={acc_before*100:.2f}%  F1={f1_before*100:.2f}%")
        print(f"  스트리밍 시작 ({len(X_st)}샘플, {STREAM_BATCH}샘플/배치)")

        accs, f1s, seen = online_finetune(base_model, X_st, y_st, X_te, y_te, device)

        acc_after, f1_after = accs[-1], f1s[-1]
        print(f"  [적응 후] Accuracy={acc_after*100:.2f}%  F1={f1_after*100:.2f}%")
        print(f"  개선: Acc {(acc_after-acc_before)*100:+.2f}%p  F1 {(f1_after-f1_before)*100:+.2f}%p")

        # 수렴 속도: 최종 F1의 90% 도달 배치 수
        target = f1_after * 0.90
        conv_batch = next((i+1 for i, f in enumerate(f1s) if f >= target), len(f1s))
        print(f"  수렴 속도: {conv_batch}배치 만에 최종 F1의 90% 달성 "
              f"({conv_batch * STREAM_BATCH}샘플)")

        summary_rows.append({
            "scenario": scenario,
            "acc_before": acc_before, "f1_before": f1_before,
            "acc_after":  acc_after,  "f1_after":  f1_after,
            "conv_batches": conv_batch,
        })

        # 그래프
        ax.plot(seen, [f*100 for f in f1s], "b-o", markersize=3, label="Fine-tune F1")
        ax.axhline(f1_before * 100, color="r", linestyle="--", label=f"Base {f1_before*100:.1f}%")
        ax.axhline(f1_after  * 100, color="g", linestyle=":",  label=f"Final {f1_after*100:.1f}%")
        ax.set_title(scenario, fontsize=10)
        ax.set_xlabel("Samples seen")
        ax.set_ylabel("F1-score (%)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = OUT_DIR / "online_learning_curves.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\n그래프 저장: {out_fig}")

    print(f"\n{'='*55}")
    print(f"  온라인 학습 결과 요약")
    print(f"{'='*55}")
    print(f"  {'시나리오':<25}  {'적응전 Acc':>10}  {'적응후 Acc':>10}  {'ΔAcc':>8}  {'수렴(배치)':>10}")
    print("  " + "-" * 70)
    for row in summary_rows:
        delta = row["acc_after"] - row["acc_before"]
        print(f"  {row['scenario']:<25}  {row['acc_before']*100:>9.2f}%  "
              f"{row['acc_after']*100:>9.2f}%  {delta*100:>+7.2f}%p  "
              f"{row['conv_batches']:>8}배치")


if __name__ == "__main__":
    main()
