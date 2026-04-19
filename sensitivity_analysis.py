#!/usr/bin/env python3
"""
sensitivity_analysis.py
───────────────────────
relay 선택 가중합의 각 feature 가중치에 대한 민감도 분석

방법:
  - 기준 가중치(baseline)에서 각 feature 가중치를 [0.5, 0.7, 1.0, 1.3, 1.5, 2.0] 배 변동
  - 나머지 가중치는 고정, 변동 후 정규화 없음 (상대적 비율 유지)
  - test set relay accuracy 변화량 측정

출력:
  - 콘솔 민감도 테이블 (feature별 max 변화량)
  - models/sensitivity_analysis.png

실행:
    python3 sensitivity_analysis.py
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

FEATURES    = ["rssi_dbm_est", "snr_db_est", "plr_pct_est", "throughput_mbps_est",
               "distance_m", "hop_count", "blocked_building_count"]
WINDOW_SIZE = 20
HYSTERESIS_THRESH = 2

BASELINE_WEIGHTS = {
    "rssi_dbm_est":        0.20,
    "snr_db_est":          0.15,
    "plr_pct_est":         0.20,
    "throughput_mbps_est": 0.15,
    "distance_m":          0.10,
    "hop_count":           0.10,
    "blocked_building_count": 0.10,
}
INVERT = {"plr_pct_est", "distance_m", "hop_count", "blocked_building_count"}

MULTIPLIERS = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
class LinkStateTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj  = nn.Linear(7, 64)
        self.pos_enc     = _PositionalEncoding(64, dropout=0.2)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.dropout     = nn.Dropout(0.2)
        self.head        = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        return self.head(self.dropout(self.transformer(x)[:, -1, :]))


class _PositionalEncoding(nn.Module):
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


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def relay_accuracy(groups, raw_lookup, snapshots_raw, model, device,
                   weights: dict) -> float:
    """주어진 weights로 relay 선택 후 test accuracy 계산."""
    invert = INVERT

    def weighted_relay(snapshot: dict, current_relay: int) -> int:
        feat_sum = defaultdict(lambda: defaultdict(float))
        feat_cnt = defaultdict(int)
        for (src, dst), r in snapshot.items():
            for f in weights:
                v = float(r.get(f, 0))
                feat_sum[src][f] += v
                feat_sum[dst][f] += v
            feat_cnt[src] += 1
            feat_cnt[dst]  += 1
        if not feat_cnt:
            return current_relay
        uav_avg = {uid: {f: feat_sum[uid][f] / feat_cnt[uid] for f in weights}
                   for uid in feat_cnt}
        f_min = {f: min(uav_avg[u][f] for u in uav_avg) for f in weights}
        f_max = {f: max(uav_avg[u][f] for u in uav_avg) for f in weights}

        def norm(v, f):
            rng = f_max[f] - f_min[f]
            n = (v - f_min[f]) / rng if rng > 1e-8 else 0.5
            return (1.0 - n) if f in invert else n

        scores = {uid: sum(weights[f] * norm(uav_avg[uid][f], f) for f in weights)
                  for uid in uav_avg}
        return max(scores, key=scores.get)

    model.eval()
    correct = total = 0

    for (sid, src, dst), grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))
        n = len(grp_sorted)
        if n < WINDOW_SIZE:
            continue

        features    = np.array([[float(r[f]) for f in FEATURES] for r in grp_sorted],
                                dtype=np.float32)
        feat_tensor = torch.tensor(features).to(device)

        bad_streak = 0
        last_relay = None

        for i in range(n - WINDOW_SIZE + 1):
            target_row    = grp_sorted[i + WINDOW_SIZE - 1]
            t_s           = target_row["time_s"]
            true_relay    = int(target_row["optimal_relay_uav"])
            raw_row       = raw_lookup.get((sid, t_s, src, dst))
            current_relay = int(raw_row["optimal_relay_uav"]) if raw_row else 2
            if last_relay is None:
                last_relay = current_relay

            snap_key = (sid, t_s)
            snap     = snapshots_raw.get(snap_key, {})

            window = feat_tensor[i:i + WINDOW_SIZE].unsqueeze(0)
            with torch.no_grad():
                pred_state = int(model(window).argmax(1).item())

            bad_streak = bad_streak + 1 if pred_state == 2 else 0

            if bad_streak >= HYSTERESIS_THRESH:
                chosen_relay = weighted_relay(snap, last_relay)
                last_relay   = chosen_relay
            else:
                last_relay   = current_relay
                chosen_relay = current_relay

            correct += (chosen_relay == true_relay)
            total   += 1

    return correct / total if total > 0 else 0.0


# ── 민감도 분석 ───────────────────────────────────────────────────────────────
def run_sensitivity(model, device, groups, raw_lookup, snapshots_raw) -> dict:
    baseline_acc = relay_accuracy(groups, raw_lookup, snapshots_raw,
                                  model, device, BASELINE_WEIGHTS)
    print(f"  Baseline relay accuracy: {baseline_acc*100:.3f}%")

    results = {}  # feature → {multiplier → acc}

    for feat in BASELINE_WEIGHTS:
        accs = {}
        for mult in MULTIPLIERS:
            w = dict(BASELINE_WEIGHTS)
            w[feat] = BASELINE_WEIGHTS[feat] * mult
            acc = relay_accuracy(groups, raw_lookup, snapshots_raw,
                                 model, device, w)
            accs[mult] = acc
        results[feat] = accs
        delta_vals = [abs(accs[m] - baseline_acc) * 100 for m in MULTIPLIERS]
        print(f"  {feat:<28} max Δ = {max(delta_vals):.4f}%p")

    return baseline_acc, results


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_sensitivity(baseline_acc: float, results: dict) -> None:
    features = list(results.keys())
    n = len(features)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle("Relay Weight Sensitivity Analysis (Transformer)", fontsize=13,
                 fontweight="bold")
    axes = axes.flatten()

    for ax, feat in zip(axes, features):
        accs   = results[feat]
        mults  = sorted(accs.keys())
        vals   = [accs[m] * 100 for m in mults]
        deltas = [v - baseline_acc * 100 for v in vals]

        colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in deltas]
        ax.bar([str(m) for m in mults], deltas, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(feat, fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Weight multiplier", fontsize=7)
        ax.set_ylabel("Δ relay acc (%p)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    # 마지막 빈 subplot 제거
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    path = OUT_DIR / "sensitivity_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    rows_scaled = load_csv(DATA_DIR / "test.csv")
    all_raw     = load_csv(DATA_DIR / "link_metrics.csv")

    raw_lookup: dict[tuple, dict] = {}
    for r in all_raw:
        raw_lookup[(r["scenario_id"], r["time_s"], r["src_uav"], r["dst_uav"])] = r

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows_scaled:
        groups[(r["scenario_id"], r["src_uav"], r["dst_uav"])].append(r)

    snapshots_raw: dict[tuple, dict] = defaultdict(dict)
    for r in all_raw:
        snapshots_raw[(r["scenario_id"], r["time_s"])][
            (int(r["src_uav"]), int(r["dst_uav"]))] = r

    model = LinkStateTransformer()
    model.load_state_dict(torch.load(OUT_DIR / "best_transformer.pt", map_location=device))
    model.to(device)

    print("\n[Sensitivity Analysis — Transformer Relay Weights]")
    baseline_acc, results = run_sensitivity(model, device, groups, raw_lookup, snapshots_raw)

    # 요약 테이블
    print(f"\n  {'Feature':<28} {'min acc':>10} {'max acc':>10} {'max |Δ|':>10}")
    print("  " + "-" * 62)
    for feat, accs in results.items():
        vals = list(accs.values())
        print(f"  {feat:<28} {min(vals)*100:>9.3f}%  {max(vals)*100:>9.3f}%  "
              f"{max(abs(v - baseline_acc)*100 for v in vals):>9.4f}%p")
    print(f"\n  Baseline: {baseline_acc*100:.3f}%")

    plot_sensitivity(baseline_acc, results)
    print("Done.")


if __name__ == "__main__":
    main()
