#!/usr/bin/env python3
"""
kpi_analysis.py
───────────────
6개 시스템 KPI 비교 분석

KPI 정의:
  ① 통신 유지율   : relay == optimal AND link_state == healthy 인 스텝 비율
  ② 단절 횟수     : 실제 disconnected 에피소드 수 (ground truth)
  ③ 평균 복원 시간 : disconnected 에피소드 평균 지속 시간 (s)
  ④ 임무 성공률   : relay 정확도 (시스템이 최적 relay 선택한 비율)
  ⑤ 에너지 효율   : UAV 재배치 총 이동 거리 (m) — 낮을수록 효율적

출력:
  - 콘솔 KPI 테이블
  - models/kpi_comparison.png

실행:
    python3 kpi_analysis.py
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
from pipeline_proactive import proactive_correct_positions

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"

FEATURES        = ["rssi_dbm_est", "snr_db_est", "plr_pct_est", "throughput_mbps_est",
                   "distance_m", "hop_count", "blocked_building_count"]
WINDOW_SIZE     = 20
TIME_STEP       = 0.25   # s
HYSTERESIS_THRESH = 2
RSSI_BAD_THRESH = -75.0
PLR_BAD_THRESH  = 20.0
DIST_BAD_THRESH = 89.0

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


def count_episodes(states: list[int], target: int) -> tuple[int, float]:
    """target 상태의 에피소드 수와 평균 지속 길이 반환."""
    episodes, cur_len, in_ep = 0, 0, False
    total_len = 0
    for s in states:
        if s == target:
            in_ep = True
            cur_len += 1
        else:
            if in_ep:
                episodes += 1
                total_len += cur_len
                cur_len = 0
            in_ep = False
    if in_ep:
        episodes += 1
        total_len += cur_len
    avg_len = total_len / episodes if episodes > 0 else 0.0
    return episodes, avg_len


# ── KPI 수집 공통 로직 ────────────────────────────────────────────────────────
def _run_system(groups, raw_lookup, snapshots_raw, positions_db,
                main_model, ew_model, device,
                mode: str) -> dict:
    """
    mode: 'no_control' | 'rule_based' | 'reactive' | 'proactive'
    """
    if main_model:
        main_model.eval()
    if ew_model:
        ew_model.eval()

    relay_correct = 0
    comm_maintain = 0   # relay==optimal AND link_state==healthy
    total         = 0
    total_move_m  = 0.0
    all_true_states: list[int] = []
    proactive_trigger = 0   # pred=degraded AND ew=disconnected AND bad_streak>=THRESH
    disconnection_avoided = 0  # 위 조건 발동 후 1s 뒤 true_state != disconnected

    for (sid, src, dst), grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))
        n = len(grp_sorted)

        if mode == "proactive":
            if n < WINDOW_SIZE + 1:
                continue
            win_range = n - WINDOW_SIZE
        else:
            if n < WINDOW_SIZE:
                continue
            win_range = n - WINDOW_SIZE + 1

        features    = (np.array([[float(r[f]) for f in FEATURES] for r in grp_sorted],
                                  dtype=np.float32)
                       if main_model else None)
        feat_tensor = torch.tensor(features).to(device) if features is not None else None

        bad_streak = 0
        last_relay = None

        for i in range(win_range):
            target_row = grp_sorted[i + WINDOW_SIZE - 1]
            t_s        = target_row["time_s"]
            true_state = int(target_row["link_state"])
            true_relay = int(target_row["optimal_relay_uav"])

            raw_row       = raw_lookup.get((sid, t_s, src, dst))
            current_relay = int(raw_row["optimal_relay_uav"]) if raw_row else 2
            if last_relay is None:
                last_relay = current_relay

            snap_key = (sid, t_s)
            snap     = snapshots_raw.get(snap_key, {})

            # ── 시스템별 bad 감지 ──────────────────────────────────────────
            if mode == "no_control":
                is_bad = False
                pred_state = 0

            elif mode == "rule_based":
                is_bad = False
                if raw_row:
                    rssi = float(raw_row.get("rssi_dbm_est", 0))
                    plr  = float(raw_row.get("plr_pct_est",  0))
                    dist = float(raw_row.get("distance_m",   0))
                    is_bad = (rssi < RSSI_BAD_THRESH or plr > PLR_BAD_THRESH
                              or dist > DIST_BAD_THRESH)
                pred_state = 2 if is_bad else 0

            else:  # reactive / proactive
                window = feat_tensor[i:i + WINDOW_SIZE].unsqueeze(0)
                with torch.no_grad():
                    pred_state = int(main_model(window).argmax(1).item())
                    ew_state   = int(ew_model(window).argmax(1).item()) if ew_model else None

                if mode == "proactive":
                    is_bad = (pred_state == 2) or (pred_state == 1 and ew_state == 2)
                else:
                    is_bad = (pred_state == 2)

            bad_streak = bad_streak + 1 if is_bad else 0

            # ── 단절 회피율 집계 (proactive 전용) ─────────────────────────
            if (mode == "proactive" and bad_streak >= HYSTERESIS_THRESH
                    and pred_state == 1 and ew_state == 2):
                proactive_trigger += 1
                future_idx = i + WINDOW_SIZE - 1 + 4   # 1초 후 (4 steps)
                if future_idx < n:
                    future_state = int(grp_sorted[future_idx]["link_state"])
                    if future_state != 2:
                        disconnection_avoided += 1

            # ── relay 선택 ─────────────────────────────────────────────────
            if mode == "no_control":
                chosen_relay = current_relay
                last_relay   = current_relay
            elif bad_streak >= HYSTERESIS_THRESH:
                chosen_relay = weighted_relay(snap, last_relay)
                last_relay   = chosen_relay

                # 재배치 거리 집계
                if snap_key in positions_db:
                    positions = positions_db[snap_key]
                    comps = _connected_components(positions)
                    if len(comps) > 1:
                        if mode == "proactive":
                            result = proactive_correct_positions(positions)
                        else:
                            result = correct_positions(positions)
                        total_move_m += sum(result["moves"].values())
            else:
                last_relay   = current_relay
                chosen_relay = current_relay

            all_true_states.append(true_state)
            relay_correct += (chosen_relay == true_relay)
            comm_maintain += (chosen_relay == true_relay and true_state == 0)
            total += 1

    # ── KPI 계산 ──────────────────────────────────────────────────────────────
    disc_episodes, avg_disc_len = count_episodes(all_true_states, target=2)

    avoidance_rate = (disconnection_avoided / proactive_trigger
                      if proactive_trigger > 0 else None)

    return {
        "통신_유지율":    comm_maintain / total if total > 0 else 0.0,
        "단절_횟수":      disc_episodes,
        "평균_복원_시간": avg_disc_len * TIME_STEP,   # steps → 초
        "임무_성공률":    relay_correct / total if total > 0 else 0.0,
        "에너지_이동거리": total_move_m,
        "단절_회피율":    avoidance_rate,
        "proactive_trigger": proactive_trigger,
        "disconnection_avoided": disconnection_avoided,
        "total": total,
    }


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_kpi(results: dict) -> None:
    systems = list(results.keys())
    n_sys   = len(systems)
    colors  = ["#78909C", "#4CAF50", "#2196F3", "#1565C0", "#FF5722", "#BF360C"]

    kpis = [
        ("Comm. Maintenance (%)", "통신_유지율",    True,  100),
        ("Mission Success (%)",   "임무_성공률",    True,  100),
        ("Disconnections",        "단절_횟수",      False, None),
        ("Avg Recovery (s)",      "평균_복원_시간", False, None),
        ("Movement Dist. (m)",    "에너지_이동거리",False, None),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle("System KPI Comparison — Test Set", fontsize=13, fontweight="bold")

    for ax, (title, key, higher_better, scale) in zip(axes, kpis):
        vals = [results[s][key] * (scale if scale else 1) for s in systems]
        bars = ax.bar(range(n_sys), vals, color=colors[:n_sys], alpha=0.85)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks(range(n_sys))
        ax.set_xticklabels([s.replace("\n", "\n") for s in systems],
                           fontsize=7, rotation=30, ha="right")
        ax.set_ylim(0, max(vals) * 1.18 + 1e-6)
        ax.grid(True, axis="y", alpha=0.3)

        arrow = "↑ 높을수록 Good" if higher_better else "↓ 낮을수록 Good"
        ax.set_xlabel(arrow, fontsize=7, color="#555555")

    plt.tight_layout()
    path = OUT_DIR / "kpi_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    rows_scaled  = load_csv(DATA_DIR / "test.csv")
    all_raw      = load_csv(DATA_DIR / "link_metrics.csv")
    positions_db = load_positions()

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

    # ground truth 에피소드 통계 (시스템과 무관하게 한 번만 계산)
    gt_states = [int(r["link_state"]) for r in rows_scaled]
    gt_disc_episodes, gt_avg_disc_len = count_episodes(gt_states, target=2)
    gt_avg_recovery_s = gt_avg_disc_len * TIME_STEP
    print(f"[Ground Truth] 단절 에피소드: {gt_disc_episodes}회  "
          f"평균 지속: {gt_avg_recovery_s:.2f}s")

    args = (groups, raw_lookup, snapshots_raw, positions_db)

    system_configs = [
        ("No Control",              None,        None,  device, "no_control"),
        ("Rule-based",              None,        None,  device, "rule_based"),
        ("ML Reactive\n(LSTM)",     lstm,        None,  device, "reactive"),
        ("ML Reactive\n(Transf.)",  transformer, None,  device, "reactive"),
        ("ML Proactive\n(LSTM)",    lstm,        ew_1s, device, "proactive"),
        ("ML Proactive\n(Transf.)", transformer, ew_1s, device, "proactive"),
    ]

    results = {}
    for name, main_m, ew_m, dev, mode in system_configs:
        label = name.replace("\n", " ")
        print(f"[{label}] ...")
        r = _run_system(*args, main_m, ew_m, dev, mode)
        # 단절 횟수 / 복원 시간은 ground truth 값으로 통일
        r["단절_횟수"]      = gt_disc_episodes
        r["평균_복원_시간"] = gt_avg_recovery_s
        results[name] = r

    # ── 콘솔 테이블 출력 ──────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"  {'System':<22} {'통신유지율':>10} {'임무성공률':>10} "
          f"{'단절횟수':>8} {'복원시간(s)':>11} {'이동거리(m)':>11} {'단절회피율':>11}")
    print("-" * 95)
    for name, r in results.items():
        label = name.replace("\n", " ")
        avoidance = (f"{r['단절_회피율']*100:>9.1f}%"
                     if r["단절_회피율"] is not None else f"{'N/A':>10}")
        print(f"  {label:<22} "
              f"{r['통신_유지율']*100:>9.2f}%  "
              f"{r['임무_성공률']*100:>9.2f}%  "
              f"{r['단절_횟수']:>8d}  "
              f"{r['평균_복원_시간']:>10.2f}s  "
              f"{r['에너지_이동거리']:>10.1f}m  "
              f"{avoidance}")
    print("=" * 95)

    # ── 단절 회피율 상세 출력 ─────────────────────────────────────────────────
    print("\n[Proactive 단절 회피율 상세]")
    for name, r in results.items():
        if r["단절_회피율"] is not None:
            label = name.replace("\n", " ")
            print(f"  {label}: 사전전환 {r['proactive_trigger']}회 → "
                  f"회피 성공 {r['disconnection_avoided']}회 "
                  f"({r['단절_회피율']*100:.1f}%)")

    plot_kpi(results)
    print("Done.")


if __name__ == "__main__":
    main()
