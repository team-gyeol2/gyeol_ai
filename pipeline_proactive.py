#!/usr/bin/env python3
"""
pipeline_proactive.py
─────────────────────
Proactive 파이프라인 — 사전 대응 기반 토폴로지 제어 + 드론 재배치

[Reactive vs Proactive 비교]

Reactive (pipeline.py):
  ① LSTM → 현재 link_state 예측
  ② disconnected → relay 전환
  ③ 고립 UAV → 위치 보정 (centroid 방향 반복 이동)

Proactive (본 파일):
  ① LSTM → 현재 link_state 예측
  ② EW-1s → 1초 후 link_state 예측
  ③ 토폴로지 제어 (고도화):
     - 현재=healthy  → relay 유지
     - 현재=degraded AND 미래=disconnected → 미리 relay 전환 (proactive)
     - 현재=degraded AND 미래=degraded    → relay 유지 (단, 경고 플래그)
     - 현재=disconnected → relay 전환 (reactive)
  ④ 드론 재배치 (고도화):
     - 현재=disconnected OR (현재=degraded AND 미래=disconnected)
       → 통신 범위(COMM_RANGE_M)까지 최소 이동 거리 계산 후 1회 이동
     - 반복 이동 없음 → 더 빠르고 에너지 효율적

실행:
    python3 pipeline_proactive.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from position_correction import load_positions, _connected_components, _dist, _link_state_from_dist

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"

FEATURES    = ["rssi_dbm_est", "snr_db_est", "plr_pct_est", "throughput_mbps_est",
               "distance_m", "hop_count", "blocked_building_count"]
WINDOW_SIZE = 20
STATE_NAMES = ["healthy", "degraded", "disconnected"]
COMM_RANGE_M = 89.0   # disconnected 경계 거리 (m) — position_correction과 동일 모델 기준
HYSTERESIS_THRESH = 2   # 연속 N스텝 이상 bad state 예측 시에만 relay 전환


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


# ── relay 선택 (가중합 score) ─────────────────────────────────────────────────
WEIGHTS = {
    "rssi_dbm_est": 0.20, "snr_db_est": 0.15, "plr_pct_est": 0.20,
    "throughput_mbps_est": 0.15, "distance_m": 0.10,
    "hop_count": 0.10, "blocked_building_count": 0.10,
}
INVERT = {"plr_pct_est", "distance_m", "hop_count", "blocked_building_count"}


def score_based_relay(snapshot: dict, current_relay: int) -> int:
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

    def norm(val, f):
        rng = f_max[f] - f_min[f]
        n = (val - f_min[f]) / rng if rng > 1e-8 else 0.5
        return (1.0 - n) if f in INVERT else n

    scores = {uid: sum(WEIGHTS[f] * norm(uav_avg[uid][f], f) for f in WEIGHTS)
              for uid in uav_avg}
    return max(scores, key=scores.get)


# ── 최소 이동 거리 기반 위치 보정 ─────────────────────────────────────────────
def proactive_correct_positions(positions: dict[int, tuple]) -> dict:
    """
    각 고립 UAV 쌍에 대해 통신 범위(COMM_RANGE_M) 안으로 들어오는
    최소 이동 거리를 한 번에 계산하여 이동.
    반복 없음 → 에너지 효율적.
    """
    pos   = {uid: list(p) for uid, p in positions.items()}
    moves = defaultdict(float)

    comps = _connected_components({uid: tuple(p) for uid, p in pos.items()})
    if len(comps) == 1:
        return {"success": True, "steps": 0,
                "final_positions": positions, "moves": {}}

    main_comp = max(comps, key=len)

    for comp in comps:
        if comp == main_comp:
            continue
        for uid in comp:
            # 주 클러스터에서 가장 가까운 UAV 탐색
            nearest = min(main_comp,
                          key=lambda m: _dist(tuple(pos[uid]), tuple(pos[m])))
            d = _dist(tuple(pos[uid]), tuple(pos[nearest]))

            if d <= COMM_RANGE_M:
                continue

            # 최소 이동: 통신 범위 안으로 딱 맞게
            move_dist = d - COMM_RANGE_M + 1.0   # 1m 여유
            dx = pos[nearest][0] - pos[uid][0]
            dy = pos[nearest][1] - pos[uid][1]
            ratio = move_dist / d
            pos[uid][0] += dx * ratio
            pos[uid][1] += dy * ratio
            moves[uid]  += move_dist

    final_pos = {uid: tuple(p) for uid, p in pos.items()}
    comps_after = _connected_components(final_pos)
    return {
        "success": len(comps_after) == 1,
        "steps":   1,
        "final_positions": final_pos,
        "moves": dict(moves),
    }


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_split(split: str) -> list[dict]:
    with open(DATA_DIR / f"{split}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_link_metrics() -> list[dict]:
    with open(DATA_DIR / "link_metrics.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── 평가 ─────────────────────────────────────────────────────────────────────
def run_proactive(split: str, main_model: nn.Module, ew_model: nn.Module,
                  device: torch.device, model_name: str) -> dict:
    print(f"\n{'='*52}")
    print(f"  [Proactive / {model_name}] {split.upper()}")
    print(f"{'='*52}")

    rows_scaled = load_split(split)
    all_raw     = load_link_metrics()
    positions_db = load_positions()

    raw_lookup = {}
    for r in all_raw:
        key = (r["scenario_id"], r["time_s"], r["src_uav"], r["dst_uav"])
        raw_lookup[key] = r

    groups = defaultdict(list)
    for r in rows_scaled:
        groups[(r["scenario_id"], r["src_uav"], r["dst_uav"])].append(r)

    snapshots_raw = defaultdict(dict)
    for r in all_raw:
        key  = (r["scenario_id"], r["time_s"])
        pair = (int(r["src_uav"]), int(r["dst_uav"]))
        snapshots_raw[key][pair] = r

    main_model.eval()
    ew_model.eval()


    y_true_state, y_pred_state = [], []
    relay_correct = pipeline_correct = total = 0
    proactive_relay = proactive_reposition = 0
    correction_triggered = correction_success = 0

    for (sid, src, dst), grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))
        n = len(grp_sorted)
        if n < WINDOW_SIZE + 1:
            continue

        features    = np.array([[float(r[f]) for f in FEATURES] for r in grp_sorted],
                                dtype=np.float32)
        feat_tensor = torch.tensor(features).to(device)

        bad_streak = 0    # hysteresis: 연속 bad state 카운터
        last_relay = None

        for i in range(n - WINDOW_SIZE):
            window     = feat_tensor[i:i + WINDOW_SIZE].unsqueeze(0)
            target_row = grp_sorted[i + WINDOW_SIZE - 1]
            t_s        = target_row["time_s"]

            with torch.no_grad():
                pred_state = int(main_model(window).argmax(1).item())
                ew_state   = int(ew_model(window).argmax(1).item())  # 1초 후 예측

            true_state = int(target_row["link_state"])

            snap_key      = (sid, t_s)
            snap          = snapshots_raw.get(snap_key, {})
            raw_row       = raw_lookup.get((sid, t_s, src, dst))
            current_relay = int(raw_row["optimal_relay_uav"]) if raw_row else 2
            if last_relay is None:
                last_relay = current_relay

            # hysteresis: disconnected 또는 proactive 조건 연속 판정
            is_bad = (pred_state == 2) or (pred_state == 1 and ew_state == 2)
            if is_bad:
                bad_streak += 1
            else:
                bad_streak = 0

            # ── Proactive 토폴로지 제어 ──────────────────────────────────────
            if bad_streak >= HYSTERESIS_THRESH:
                if pred_state == 2:
                    chosen_relay = score_based_relay(snap, last_relay)
                elif pred_state == 1 and ew_state == 2:
                    chosen_relay = score_based_relay(snap, last_relay)
                    proactive_relay += 1
                else:
                    chosen_relay = last_relay
                last_relay = chosen_relay
            else:
                # healthy/degraded 구간: 현재 optimal relay 추적
                last_relay   = current_relay
                chosen_relay = current_relay

            true_relay = int(target_row["optimal_relay_uav"])

            # ── Proactive 드론 재배치 ────────────────────────────────────────
            trigger_reposition = bad_streak >= HYSTERESIS_THRESH and is_bad
            if trigger_reposition and snap_key in positions_db:
                positions = positions_db[snap_key]
                comps = _connected_components(positions)
                if len(comps) > 1:
                    correction_triggered += 1
                    if pred_state == 1 and ew_state == 2:
                        result = proactive_correct_positions(positions)
                        proactive_reposition += 1
                    else:
                        from position_correction import correct_positions
                        result = correct_positions(positions)
                    if result["success"]:
                        correction_success += 1

            y_true_state.append(true_state)
            y_pred_state.append(pred_state)
            relay_correct    += (chosen_relay == true_relay)
            pipeline_correct += (pred_state == true_state and chosen_relay == true_relay)
            total += 1

    # ── 지표 출력 ─────────────────────────────────────────────────────────────
    acc  = sum(t == p for t, p in zip(y_true_state, y_pred_state)) / total
    prec = precision_score(y_true_state, y_pred_state, average="macro", zero_division=0)
    rec  = recall_score   (y_true_state, y_pred_state, average="macro", zero_division=0)
    f1   = f1_score       (y_true_state, y_pred_state, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true_state, y_pred_state, labels=[0, 1, 2])

    print(f"\n[link_state]  samples: {total}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%  (macro)")
    print(f"  Recall   : {rec*100:.2f}%  (macro)")
    print(f"  F1-score : {f1*100:.2f}%  (macro)")
    print(f"\n  Confusion Matrix (row=true, col=pred)")
    print(f"  {'':14s}" + "  ".join(f"{s:>14s}" for s in STATE_NAMES))
    for i, row in enumerate(cm):
        print(f"  {STATE_NAMES[i]:14s}" + "  ".join(f"{v:>14d}" for v in row))

    print(f"\n[Pipeline]")
    print(f"  relay Accuracy     : {relay_correct/total*100:.2f}%  ({relay_correct}/{total})")
    print(f"  pipeline Accuracy  : {pipeline_correct/total*100:.2f}%  ({pipeline_correct}/{total})")
    print(f"  proactive relay    : {proactive_relay} times")
    print(f"  proactive reposition: {proactive_reposition} times")
    if correction_triggered > 0:
        print(f"  reposition success : {correction_success}/{correction_triggered}"
              f"  ({correction_success/correction_triggered*100:.1f}%)")

    return {
        "acc": acc, "f1": f1,
        "relay_acc": relay_correct / total,
        "pipeline_acc": pipeline_correct / total,
        "proactive_relay": proactive_relay,
        "proactive_reposition": proactive_reposition,
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    model_defs = {
        "LSTM":        (LinkStateLSTM(),        OUT_DIR / "best_lstm.pt"),
        "Transformer": (LinkStateTransformer(), OUT_DIR / "best_transformer.pt"),
    }
    ew_model = LinkStateLSTM()
    ew_model.load_state_dict(torch.load(OUT_DIR / "best_ew_1s.pt", map_location=device))
    ew_model.to(device)
    ew_model.eval()

    print("\n[Reactive vs Proactive 비교 — TEST]")
    print(f"{'Model':<14} {'Mode':<12} {'link Acc':>10} {'F1':>8} {'relay Acc':>10} {'pipe Acc':>10}")
    print("-" * 68)

    for name, (model, path) in model_defs.items():
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        res = run_proactive("test", model, ew_model, device, name)
        print(f"  {name:<12} proactive    "
              f"{res['acc']*100:>9.2f}%  "
              f"{res['f1']*100:>7.2f}%  "
              f"{res['relay_acc']*100:>9.2f}%  "
              f"{res['pipeline_acc']*100:>9.2f}%")


if __name__ == "__main__":
    main()
