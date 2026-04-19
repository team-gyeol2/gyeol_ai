#!/usr/bin/env python3
"""
early_warning.py
────────────────
Early Warning 시스템 — 리드타임별 통신 단절 예측

리드타임 정의:
  현재 5초 시계열 윈도우를 기반으로 N초 후의 link_state를 예측
  lead_time = 1s (4 step), 3s (12 step), 5s (20 step)

출력:
  - 리드타임별 F1-score 비교표
  - 시나리오별 경고 발생 시점 시각화 (models/early_warning_curves.png)
  - 리드타임별 모델 저장 (models/best_ew_{N}s.pt)

실행:
    python3 early_warning.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"

FEATURES    = ["rssi_dbm_est", "snr_db_est", "plr_pct_est", "throughput_mbps_est",
               "distance_m", "hop_count", "blocked_building_count"]
WINDOW_SIZE = 20        # 5초
TIME_STEP   = 0.25      # s
LEAD_TIMES  = [1, 3, 5] # 초

INPUT_SIZE  = len(FEATURES)
STATE_NAMES = ["healthy", "degraded", "disconnected"]


# ── 모델 ──────────────────────────────────────────────────────────────────────
class EarlyWarningLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=INPUT_SIZE, hidden_size=64,
                               num_layers=2, dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.head    = nn.Linear(64, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :]))


# ── 슬라이딩 윈도우 (리드타임 포함) ──────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_ew_windows(rows: list[dict], lead_steps: int):
    """
    lead_steps 이후의 link_state를 레이블로 사용하는 슬라이딩 윈도우.
    Returns: X (N, WINDOW_SIZE, INPUT_SIZE), y (N,), meta (N, [scenario_id, time_s])
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["scenario_id"], r["src_uav"], r["dst_uav"])
        groups[key].append(r)

    X_list, y_list, meta_list = [], [], []
    for key, grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))
        n = len(grp_sorted)
        if n < WINDOW_SIZE + lead_steps:
            continue

        features = np.array([[float(r[f]) for f in FEATURES] for r in grp_sorted],
                             dtype=np.float32)
        labels   = np.array([int(r["link_state"]) for r in grp_sorted], dtype=np.int64)

        for i in range(n - WINDOW_SIZE - lead_steps + 1):
            X_list.append(features[i : i + WINDOW_SIZE])
            y_list.append(labels[i + WINDOW_SIZE - 1 + lead_steps])
            # 예측 기준 시각 (윈도우 마지막 시점)
            meta_list.append((grp_sorted[0]["scenario_id"],
                              float(grp_sorted[i + WINDOW_SIZE - 1]["time_s"])))

    if not X_list:
        return (np.empty((0, WINDOW_SIZE, INPUT_SIZE)),
                np.empty((0,), dtype=np.int64), [])
    return (np.stack(X_list).astype(np.float32),
            np.array(y_list, dtype=np.int64),
            meta_list)


# ── 학습 ──────────────────────────────────────────────────────────────────────
def train_model(lead_s: int, device: torch.device) -> dict:
    lead_steps = int(lead_s / TIME_STEP)
    print(f"\n{'='*50}")
    print(f"  Lead time: {lead_s}s  ({lead_steps} steps)")
    print(f"{'='*50}")

    splits = {}
    for split in ("train", "val", "test"):
        rows = load_csv(DATA_DIR / f"{split}.csv")
        X, y, meta = make_ew_windows(rows, lead_steps)
        splits[split] = (X, y, meta)
        print(f"  {split:<6} {X.shape}  disconnected: {(y==2).sum()}/{len(y)}")

    X_tr, y_tr, _ = splits["train"]
    X_va, y_va, _ = splits["val"]
    X_te, y_te, meta_te = splits["test"]

    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                           batch_size=64, shuffle=True)

    model = EarlyWarningLSTM().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_cnt  = 0
    PATIENCE = 10

    X_va_t = torch.tensor(X_va).to(device)
    y_va_t = torch.tensor(y_va).to(device)
    X_te_t = torch.tensor(X_te).to(device)

    for epoch in range(1, 51):
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
            torch.save(model.state_dict(), OUT_DIR / f"best_ew_{lead_s}s.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # 최적 모델로 평가
    model.load_state_dict(torch.load(OUT_DIR / f"best_ew_{lead_s}s.pt",
                                     map_location=device))
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).argmax(1).cpu().numpy()

    acc = (preds == y_te).mean()
    f1  = f1_score(y_te, preds, average="macro", zero_division=0)
    f1_per = f1_score(y_te, preds, average=None, labels=[0,1,2], zero_division=0)

    print(f"\n  [TEST] Accuracy: {acc*100:.2f}%  |  F1 (macro): {f1*100:.2f}%")
    print(f"         F1 per class: healthy={f1_per[0]*100:.1f}%  "
          f"degraded={f1_per[1]*100:.1f}%  disconnected={f1_per[2]*100:.1f}%")

    return {
        "lead_s": lead_s,
        "acc": acc,
        "f1_macro": f1,
        "f1_per": f1_per.tolist(),
        "preds": preds,
        "y_true": y_te,
        "meta": meta_te,
        "model": model,
    }


# ── 시각화 ────────────────────────────────────────────────────────────────────
def _best_pair(meta, preds, y_true):
    """disconnected 레이블이 가장 많은 (scenario_id, src, dst) 조합 선택."""
    from collections import defaultdict, Counter
    # meta: [(scenario_id, time_s), ...]  — src/dst 정보 없음
    # 시나리오별로 그룹 → disconnected 비율 높은 시나리오 선택 후
    # 해당 시나리오의 고유 시간 구간을 그대로 사용
    sid_disc = defaultdict(int)
    sid_total = defaultdict(int)
    for i, (sid, _) in enumerate(meta):
        sid_total[sid] += 1
        if y_true[i] == 2:
            sid_disc[sid] += 1
    if not sid_total:
        return None, []
    # disconnected 비율 높은 시나리오 우선
    chosen = max(sid_total, key=lambda s: sid_disc[s] / max(sid_total[s], 1))
    # 해당 시나리오의 고유 시간값으로 중복 제거 (쌍 여러 개 겹침 방지)
    seen_times = set()
    mask = []
    for i, (sid, t) in enumerate(meta):
        if sid == chosen and t not in seen_times:
            seen_times.add(t)
            mask.append(i)
    mask.sort(key=lambda i: meta[i][1])
    return chosen, mask


def plot_warning_timeline(results: list[dict]) -> None:
    from matplotlib.patches import Patch
    color_map = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
    label_map = {0: "healthy", 1: "degraded", 2: "disconnected"}

    fig, axes = plt.subplots(len(results), 1, figsize=(14, 3.5 * len(results)), sharex=False)
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        lead_s = res["lead_s"]
        meta   = res["meta"]
        preds  = res["preds"]
        y_true = res["y_true"]

        chosen_sid, mask = _best_pair(meta, preds, y_true)
        if not mask:
            ax.set_title(f"Lead {lead_s}s — no data")
            continue

        times = [meta[i][1] for i in mask]
        p_arr = [preds[i]   for i in mask]
        t_arr = [y_true[i]  for i in mask]

        for t, p, gt in zip(times, p_arr, t_arr):
            ax.bar(t, 1, width=TIME_STEP * 0.85, color=color_map[p], alpha=0.8)
            if gt == 2 and p != 2:
                ax.bar(t, 1, width=TIME_STEP * 0.85, color="none",
                       edgecolor="black", linewidth=1.5)

        for t, gt in zip(times, t_arr):
            if gt == 2:
                ax.axvline(t + lead_s, color="#c0392b", alpha=0.3, linewidth=1)

        legend = [Patch(color=color_map[k], label=f"Pred: {label_map[k]}")
                  for k in [0, 1, 2]]
        legend.append(Patch(facecolor="none", edgecolor="black",
                            linewidth=1.5, label="Miss (true disconnected)"))
        ax.legend(handles=legend, loc="upper right", fontsize=8)
        ax.set_title(f"Lead time {lead_s}s — {chosen_sid}", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        ax.set_xlim(min(times) - 0.5, max(times) + 0.5)

    plt.tight_layout()
    path = OUT_DIR / "early_warning_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_f1_comparison(results: list[dict]) -> None:
    leads  = [r["lead_s"] for r in results]
    f1_mac = [r["f1_macro"] * 100 for r in results]
    f1_dis = [r["f1_per"][2] * 100 for r in results]

    x = np.arange(len(leads))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, f1_mac, w, label="F1 macro",        color="#3498db")
    ax.bar(x + w/2, f1_dis, w, label="F1 disconnected", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}s" for l in leads])
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("F1-score (%)")
    ax.set_title("F1-score by Lead Time")
    ax.set_ylim(0, 105)
    ax.legend()
    for i, (m, d) in enumerate(zip(f1_mac, f1_dis)):
        ax.text(i - w/2, m + 0.5, f"{m:.1f}", ha="center", fontsize=8)
        ax.text(i + w/2, d + 0.5, f"{d:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / "ew_f1_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    results = []
    for lead_s in LEAD_TIMES:
        res = train_model(lead_s, device)
        results.append(res)

    # 요약 테이블
    print(f"\n{'='*55}")
    print(f"  리드타임별 성능 요약 (TEST)")
    print(f"{'='*55}")
    print(f"  {'리드타임':<10} {'Accuracy':>10} {'F1 macro':>10} {'F1 disc':>10}")
    print(f"  {'-'*44}")
    for r in results:
        print(f"  {str(r['lead_s'])+'s':<10} "
              f"{r['acc']*100:>9.2f}%  "
              f"{r['f1_macro']*100:>9.2f}%  "
              f"{r['f1_per'][2]*100:>9.2f}%")

    # 시각화
    plot_f1_comparison(results)
    plot_warning_timeline(results)


if __name__ == "__main__":
    main()
