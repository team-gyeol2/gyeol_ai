#!/usr/bin/env python3
"""
compare_systems.py
──────────────────
4개 시스템 성능 비교 (동일 test set 기준)

System 1 — No Control  : 예측/제어 없음, relay 고정
System 2 — Rule-based  : 임계값 기반 감지, ML 없음
System 3 — ML Reactive : LSTM/Transformer + hysteresis (pipeline.py)
System 4 — ML Proactive: LSTM/Transformer + EW + hysteresis (pipeline_proactive.py)

출력:
  - 콘솔 비교 테이블
  - models/system_comparison.png
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
from position_correction import correct_positions, load_positions, _connected_components

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"

FEATURES    = ["rssi_dbm_est", "snr_db_est", "plr_pct_est", "throughput_mbps_est",
               "distance_m", "hop_count", "blocked_building_count"]
WINDOW_SIZE = 20
HYSTERESIS_THRESH = 2

# Rule-based 임계값 (link_metrics.csv 원본 단위 기준)
RSSI_BAD_THRESH = -75.0   # dBm
PLR_BAD_THRESH  = 20.0    # %
DIST_BAD_THRESH = 89.0    # m

WEIGHTS = {
    "rssi_dbm_est": 0.20, "snr_db_est": 0.15, "plr_pct_est": 0.20,
    "throughput_mbps_est": 0.15, "distance_m": 0.10,
    "hop_count": 0.10, "blocked_building_count": 0.10,
}
INVERT = {"plr_pct_est", "distance_m", "hop_count", "blocked_building_count"}


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
        self.input_proj  = nn.Linear(7, 64)
        self.pos_enc     = PositionalEncoding(64, dropout=0.2)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.dropout     = nn.Dropout(0.2)
        self.head        = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        return self.head(self.dropout(self.transformer(x)[:, -1, :]))


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def weighted_relay(snapshot: dict, current_relay: int) -> int:
    feat_sum = defaultdict(lambda: defaultdict(float))
    feat_cnt = defaultdict(int)
    for (src, dst), r in snapshot.items():
        for f in WEIGHTS:
            v = float(r.get(f, 0))
            feat_sum[src][f] += v
            feat_sum[dst][f] += v
        feat_cnt[src] += 1
        feat_cnt[dst]  += 1
    if not feat_cnt:
        return current_relay
    uav_avg = {uid: {f: feat_sum[uid][f] / feat_cnt[uid] for f in WEIGHTS}
               for uid in feat_cnt}
    f_min = {f: min(uav_avg[u][f] for u in uav_avg) for f in WEIGHTS}
    f_max = {f: max(uav_avg[u][f] for u in uav_avg) for f in WEIGHTS}

    def norm(v, f):
        rng = f_max[f] - f_min[f]
        n = (v - f_min[f]) / rng if rng > 1e-8 else 0.5
        return (1.0 - n) if f in INVERT else n

    scores = {uid: sum(WEIGHTS[f] * norm(uav_avg[uid][f], f) for f in WEIGHTS)
              for uid in uav_avg}
    return max(scores, key=scores.get)


# ── System 1: No Control ──────────────────────────────────────────────────────
def run_no_control(groups, raw_lookup, snapshots_raw, positions_db) -> dict:
    """예측/제어 없음. relay를 절대 바꾸지 않고, 재배치도 없음."""
    relay_correct = pipeline_correct = total = 0

    for (sid, src, dst), grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))
        n = len(grp_sorted)
        if n < WINDOW_SIZE:
            continue

        for i in range(n - WINDOW_SIZE + 1):
            target_row = grp_sorted[i + WINDOW_SIZE - 1]
            t_s        = target_row["time_s"]
            raw_row    = raw_lookup.get((sid, t_s, src, dst))
            current_relay = int(raw_row["optimal_relay_uav"]) if raw_row else 2

            # 제어 없음: 그냥 현재 relay 유지
            chosen_relay = current_relay
            true_relay   = int(target_row["optimal_relay_uav"])
            true_state   = int(target_row["link_state"])

            relay_correct    += (chosen_relay == true_relay)
            # pipeline_correct: 상태 예측 없으므로 healthy(0)로 가정
            pipeline_correct += (true_state == 0 and chosen_relay == true_relay)
            total += 1

    return {
        "relay_acc":    relay_correct / total,
        "pipeline_acc": pipeline_correct / total,
        "reposition_triggered": 0,
        "reposition_success":   0,
        "total": total,
    }


# ── System 2: Rule-based Threshold ───────────────────────────────────────────
def run_rule_based(groups, raw_lookup, snapshots_raw, positions_db) -> dict:
    """
    ML 없이 원본 지표 임계값으로 링크 상태 감지.
    RSSI < -75 dBm OR PLR > 20% OR distance > 89m → bad 판정
    """
    relay_correct = pipeline_correct = total = 0
    reposition_triggered = reposition_success = 0

    for (sid, src, dst), grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))
        n = len(grp_sorted)
        if n < WINDOW_SIZE:
            continue

        bad_streak = 0
        last_relay = None

        for i in range(n - WINDOW_SIZE + 1):
            target_row    = grp_sorted[i + WINDOW_SIZE - 1]
            t_s           = target_row["time_s"]
            raw_row       = raw_lookup.get((sid, t_s, src, dst))
            current_relay = int(raw_row["optimal_relay_uav"]) if raw_row else 2
            if last_relay is None:
                last_relay = current_relay

            true_relay = int(target_row["optimal_relay_uav"])
            true_state = int(target_row["link_state"])

            # 원본 지표로 bad 감지
            is_bad = False
            if raw_row:
                rssi = float(raw_row.get("rssi_dbm", raw_row.get("rssi_dbm_est", 0)))
                plr  = float(raw_row.get("plr_pct",  raw_row.get("plr_pct_est",  0)))
                dist = float(raw_row.get("distance_m", 0))
                is_bad = (rssi < RSSI_BAD_THRESH or plr > PLR_BAD_THRESH
                          or dist > DIST_BAD_THRESH)

            bad_streak = bad_streak + 1 if is_bad else 0

            snap_key = (sid, t_s)
            snap     = snapshots_raw.get(snap_key, {})

            if bad_streak >= HYSTERESIS_THRESH:
                chosen_relay = weighted_relay(snap, last_relay)
                last_relay   = chosen_relay

                if snap_key in positions_db:
                    positions = positions_db[snap_key]
                    comps = _connected_components(positions)
                    if len(comps) > 1:
                        reposition_triggered += 1
                        result = correct_positions(positions)
                        if result["success"]:
                            reposition_success += 1
            else:
                last_relay   = current_relay
                chosen_relay = current_relay

            relay_correct    += (chosen_relay == true_relay)
            # rule-based는 상태 예측 없음 → bad 감지 = disconnected(2) 가정
            pred_state = 2 if bad_streak >= HYSTERESIS_THRESH else 0
            pipeline_correct += (pred_state == true_state and chosen_relay == true_relay)
            total += 1

    return {
        "relay_acc":    relay_correct / total,
        "pipeline_acc": pipeline_correct / total,
        "reposition_triggered": reposition_triggered,
        "reposition_success":   reposition_success,
        "total": total,
    }


# ── System 3 & 4: ML 기반 ─────────────────────────────────────────────────────
def run_ml_system(groups, raw_lookup, snapshots_raw, positions_db,
                  main_model, ew_model, device, proactive=False) -> dict:
    main_model.eval()
    if ew_model:
        ew_model.eval()

    relay_correct = pipeline_correct = total = 0
    reposition_triggered = reposition_success = 0
    proactive_relay_cnt = 0

    for (sid, src, dst), grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))
        n = len(grp_sorted)
        min_len = WINDOW_SIZE + 1 if proactive else WINDOW_SIZE
        if n < min_len:
            continue

        features    = np.array([[float(r[f]) for f in FEATURES] for r in grp_sorted],
                                dtype=np.float32)
        feat_tensor = torch.tensor(features, dtype=torch.float32).to(device)

        bad_streak = 0
        last_relay = None
        win_range  = n - WINDOW_SIZE if proactive else n - WINDOW_SIZE + 1

        for i in range(win_range):
            window     = feat_tensor[i:i + WINDOW_SIZE].unsqueeze(0)
            target_row = grp_sorted[i + WINDOW_SIZE - 1]
            t_s        = target_row["time_s"]

            with torch.no_grad():
                pred_state = int(main_model(window).argmax(1).item())
                ew_state   = int(ew_model(window).argmax(1).item()) if ew_model else None

            true_state    = int(target_row["link_state"])
            raw_row       = raw_lookup.get((sid, t_s, src, dst))
            current_relay = int(raw_row["optimal_relay_uav"]) if raw_row else 2
            if last_relay is None:
                last_relay = current_relay

            true_relay = int(target_row["optimal_relay_uav"])
            snap_key   = (sid, t_s)
            snap       = snapshots_raw.get(snap_key, {})

            # bad 조건 판단
            if proactive:
                is_bad = (pred_state == 2) or (pred_state == 1 and ew_state == 2)
            else:
                is_bad = (pred_state == 2)

            bad_streak = bad_streak + 1 if is_bad else 0

            if bad_streak >= HYSTERESIS_THRESH:
                if proactive and pred_state == 1 and ew_state == 2:
                    proactive_relay_cnt += 1
                chosen_relay = weighted_relay(snap, last_relay)
                last_relay   = chosen_relay

                if snap_key in positions_db:
                    positions = positions_db[snap_key]
                    comps = _connected_components(positions)
                    if len(comps) > 1:
                        reposition_triggered += 1
                        result = correct_positions(positions)
                        if result["success"]:
                            reposition_success += 1
            else:
                last_relay   = current_relay
                chosen_relay = current_relay

            relay_correct    += (chosen_relay == true_relay)
            pipeline_correct += (pred_state == true_state and chosen_relay == true_relay)
            total += 1

    return {
        "relay_acc":    relay_correct / total,
        "pipeline_acc": pipeline_correct / total,
        "reposition_triggered": reposition_triggered,
        "reposition_success":   reposition_success,
        "proactive_relay": proactive_relay_cnt,
        "total": total,
    }


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_comparison(results: dict) -> None:
    systems = list(results.keys())
    relay_accs    = [results[s]["relay_acc"] * 100    for s in systems]
    pipeline_accs = [results[s]["pipeline_acc"] * 100 for s in systems]

    x = np.arange(len(systems))
    w = 0.35
    colors_relay    = ["#78909C", "#4CAF50", "#2196F3", "#1565C0", "#FF5722", "#BF360C"]
    colors_pipeline = ["#B0BEC5", "#A5D6A7", "#90CAF9", "#82B1FF", "#FFAB91", "#FF8A65"]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars1 = ax.bar(x - w/2, relay_accs,    w, label="Relay Accuracy",    color=colors_relay)
    bars2 = ax.bar(x + w/2, pipeline_accs, w, label="Pipeline Accuracy", color=colors_pipeline,
                   edgecolor=[c for c in colors_relay], linewidth=1.2)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_title("System Comparison — Test Set", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / "system_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    # 데이터 로드
    rows_scaled = load_csv(DATA_DIR / "test.csv")
    all_raw     = load_csv(DATA_DIR / "link_metrics.csv")
    positions_db = load_positions()

    raw_lookup: dict[tuple, dict] = {}
    for r in all_raw:
        raw_lookup[(r["scenario_id"], r["time_s"], r["src_uav"], r["dst_uav"])] = r

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows_scaled:
        groups[(r["scenario_id"], r["src_uav"], r["dst_uav"])].append(r)

    snapshots_raw: dict[tuple, dict] = defaultdict(dict)
    for r in all_raw:
        snapshots_raw[(r["scenario_id"], r["time_s"])][(int(r["src_uav"]), int(r["dst_uav"]))] = r

    # 모델 로드
    lstm = LinkStateLSTM()
    lstm.load_state_dict(torch.load(OUT_DIR / "best_lstm.pt", map_location=device))
    lstm.to(device)

    transformer = LinkStateTransformer()
    transformer.load_state_dict(torch.load(OUT_DIR / "best_transformer.pt", map_location=device))
    transformer.to(device)

    ew_1s = LinkStateLSTM()
    ew_1s.load_state_dict(torch.load(OUT_DIR / "best_ew_1s.pt", map_location=device))
    ew_1s.to(device)

    print("\n" + "="*65)
    print("  System Comparison — TEST")
    print("="*65)

    results = {}

    # System 1
    print("\n[1] No Control ...")
    results["No Control"] = run_no_control(groups, raw_lookup, snapshots_raw, positions_db)

    # System 2
    print("[2] Rule-based ...")
    results["Rule-based"] = run_rule_based(groups, raw_lookup, snapshots_raw, positions_db)

    # System 3 — LSTM Reactive
    print("[3] ML Reactive (LSTM) ...")
    results["ML Reactive\n(LSTM)"] = run_ml_system(
        groups, raw_lookup, snapshots_raw, positions_db,
        lstm, None, device, proactive=False)

    # System 4 — Transformer Reactive
    print("[4] ML Reactive (Transformer) ...")
    results["ML Reactive\n(Transformer)"] = run_ml_system(
        groups, raw_lookup, snapshots_raw, positions_db,
        transformer, None, device, proactive=False)

    # System 5 — LSTM Proactive
    print("[5] ML Proactive (LSTM) ...")
    results["ML Proactive\n(LSTM)"] = run_ml_system(
        groups, raw_lookup, snapshots_raw, positions_db,
        lstm, ew_1s, device, proactive=True)

    # System 6 — Transformer Proactive
    print("[6] ML Proactive (Transformer) ...")
    results["ML Proactive\n(Transformer)"] = run_ml_system(
        groups, raw_lookup, snapshots_raw, positions_db,
        transformer, ew_1s, device, proactive=True)

    # ── 결과 테이블 출력 ──────────────────────────────────────────────────────
    header = f"  {'System':<22} {'relay Acc':>10} {'pipe Acc':>10} {'reposition':>12}"
    print("\n" + "="*65)
    print(header)
    print("-"*65)

    for name, r in results.items():
        label = name.replace("\n", " ")
        repos = (f"{r['reposition_success']}/{r['reposition_triggered']}"
                 f" ({r['reposition_success']/r['reposition_triggered']*100:.0f}%)"
                 if r["reposition_triggered"] > 0 else "-")
        print(f"  {label:<22} {r['relay_acc']*100:>9.2f}%  "
              f"{r['pipeline_acc']*100:>9.2f}%  {repos:>12}")

    print("="*65)

    plot_comparison(results)


if __name__ == "__main__":
    main()
