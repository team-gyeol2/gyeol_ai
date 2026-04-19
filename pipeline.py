#!/usr/bin/env python3
"""
pipeline.py
───────────
통신 장애 예측 + 토폴로지 재구성 + 드론 재배치 파이프라인

① LSTM / Transformer: 링크 시계열 → link_state 예측
② 규칙 기반 토폴로지 재구성:
   - healthy / degraded → 현재 relay 유지
   - disconnected       → 평균 RSSI 최고 UAV로 relay 전환 (ms 단위)
③ 물리적 드론 재배치:
   - relay 전환 후에도 고립 UAV 존재 시 position_correction 호출

평가 지표:
   - link_state: Accuracy, Precision, Recall, F1-score (macro), Confusion Matrix
   - relay 선택 Accuracy
   - 파이프라인 전체 Accuracy
   - 위치 보정 발동 횟수 및 성공률

실행:
    python3 pipeline.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from position_correction import correct_positions, load_positions, _connected_components

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"

FEATURES    = ["rssi_dbm_est", "snr_db_est", "plr_pct_est", "throughput_mbps_est",
               "distance_m", "hop_count", "blocked_building_count"]
PAIR_ORDER  = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
WINDOW_SIZE = 20
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
        x    = self.input_proj(x)
        x    = self.pos_enc(x)
        x    = self.transformer(x)
        last = self.dropout(x[:, -1, :])
        return self.head(last)


# ── 규칙 기반 relay 선택 (가중합 score 기반) ─────────────────────────────────
# Score_i = w1·R + w2·S + w3·P + w4·T + w5·D + w6·H + w7·B
# R,S,T: 클수록 좋음 (정방향 정규화)
# P,D,H,B: 작을수록 좋음 (역정규화)
WEIGHTS = {
    "rssi_dbm_est":        0.20,
    "snr_db_est":          0.15,
    "plr_pct_est":         0.20,
    "throughput_mbps_est": 0.15,
    "distance_m":          0.10,
    "hop_count":           0.10,
    "blocked_building_count": 0.10,
}
INVERT = {"plr_pct_est", "distance_m", "hop_count", "blocked_building_count"}


def rule_based_relay(snapshot: dict, current_relay: int) -> int:
    # UAV별 feature 평균 집계
    feat_sum: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    feat_cnt: dict[int, int] = defaultdict(int)
    for (src, dst), r in snapshot.items():
        for f in WEIGHTS:
            val = float(r[f]) if f in r else 0.0
            feat_sum[src][f] += val
            feat_sum[dst][f] += val
        feat_cnt[src] += 1
        feat_cnt[dst]  += 1

    if not feat_cnt:
        return current_relay

    uav_avg: dict[int, dict[str, float]] = {
        uid: {f: feat_sum[uid][f] / feat_cnt[uid] for f in WEIGHTS}
        for uid in feat_cnt
    }

    # feature별 min-max (UAV 간)
    f_min = {f: min(uav_avg[u][f] for u in uav_avg) for f in WEIGHTS}
    f_max = {f: max(uav_avg[u][f] for u in uav_avg) for f in WEIGHTS}

    def normalize(val: float, f: str) -> float:
        rng = f_max[f] - f_min[f]
        n = (val - f_min[f]) / rng if rng > 1e-8 else 0.5
        return (1.0 - n) if f in INVERT else n

    scores = {
        uid: sum(WEIGHTS[f] * normalize(uav_avg[uid][f], f) for f in WEIGHTS)
        for uid in uav_avg
    }
    return max(scores, key=scores.get)


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_raw(split: str) -> list[dict]:
    with open(DATA_DIR / f"{split}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_link_metrics() -> list[dict]:
    with open(DATA_DIR / "link_metrics.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── 평가 ─────────────────────────────────────────────────────────────────────
def run_pipeline(split: str, model: nn.Module, device: torch.device,
                 model_name: str) -> None:
    print(f"\n{'='*50}")
    print(f"  [{model_name}] {split.upper()}")
    print(f"{'='*50}")

    rows_scaled = load_raw(split)
    all_raw     = load_link_metrics()

    raw_lookup: dict[tuple, dict] = {}
    for r in all_raw:
        key = (r["scenario_id"], r["time_s"], r["src_uav"], r["dst_uav"])
        raw_lookup[key] = r

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows_scaled:
        key = (r["scenario_id"], r["src_uav"], r["dst_uav"])
        groups[key].append(r)

    snapshots_raw: dict[tuple, dict] = defaultdict(dict)
    for r in all_raw:
        key  = (r["scenario_id"], r["time_s"])
        pair = (int(r["src_uav"]), int(r["dst_uav"]))
        snapshots_raw[key][pair] = r

    positions_db = load_positions()

    model.eval()
    y_true_state, y_pred_state = [], []
    relay_correct = pipeline_correct = total = 0
    correction_triggered = correction_success = 0

    for (sid, src, dst), grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))
        n = len(grp_sorted)
        if n < WINDOW_SIZE:
            continue

        features    = np.array([[float(r[f]) for f in FEATURES] for r in grp_sorted],
                                dtype=np.float32)
        feat_tensor = torch.tensor(features, dtype=torch.float32).to(device)

        for i in range(n - WINDOW_SIZE + 1):
            window     = feat_tensor[i : i + WINDOW_SIZE].unsqueeze(0)
            target_row = grp_sorted[i + WINDOW_SIZE - 1]
            t_s        = target_row["time_s"]

            with torch.no_grad():
                pred_state = int(model(window).argmax(1).item())
            true_state = int(target_row["link_state"])

            snap_key      = (sid, t_s)
            snap          = snapshots_raw.get(snap_key, {})
            raw_row       = raw_lookup.get((sid, t_s, src, dst))
            current_relay = int(raw_row["optimal_relay_uav"]) if raw_row else 2

            chosen_relay = rule_based_relay(snap, current_relay) \
                           if pred_state == 2 else current_relay
            true_relay   = int(target_row["optimal_relay_uav"])

            # ③ 물리적 드론 재배치 — relay 전환 후에도 고립 UAV 존재 시
            pos_key = (sid, t_s)
            if pred_state == 2 and pos_key in positions_db:
                positions = positions_db[pos_key]
                comps = _connected_components(positions)
                if len(comps) > 1:
                    correction_triggered += 1
                    result = correct_positions(positions)
                    if result["success"]:
                        correction_success += 1

            y_true_state.append(true_state)
            y_pred_state.append(pred_state)
            relay_correct    += (chosen_relay == true_relay)
            pipeline_correct += (pred_state == true_state and chosen_relay == true_relay)
            total += 1

    # ── link_state 지표 ───────────────────────────────────────────────────────
    acc   = sum(t == p for t, p in zip(y_true_state, y_pred_state)) / total
    prec  = precision_score(y_true_state, y_pred_state, average="macro", zero_division=0)
    rec   = recall_score   (y_true_state, y_pred_state, average="macro", zero_division=0)
    f1    = f1_score       (y_true_state, y_pred_state, average="macro", zero_division=0)
    cm    = confusion_matrix(y_true_state, y_pred_state, labels=[0, 1, 2])

    print(f"\n[link_state 분류]  샘플 수: {total}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%  (macro)")
    print(f"  Recall   : {rec*100:.2f}%  (macro)")
    print(f"  F1-score : {f1*100:.2f}%  (macro)")
    print(f"\n  Confusion Matrix (행=실제, 열=예측)")
    print(f"  {'':12s}  " + "  ".join(f"{s:>12s}" for s in STATE_NAMES))
    for i, row in enumerate(cm):
        print(f"  {STATE_NAMES[i]:12s}  " + "  ".join(f"{v:>12d}" for v in row))

    print(f"\n[파이프라인]")
    print(f"  relay 선택 Accuracy : {relay_correct/total*100:.2f}%  ({relay_correct}/{total})")
    print(f"  전체 파이프라인 Acc  : {pipeline_correct/total*100:.2f}%  ({pipeline_correct}/{total})")
    if correction_triggered > 0:
        print(f"\n[드론 재배치]")
        print(f"  발동 횟수 : {correction_triggered}")
        print(f"  복원 성공 : {correction_success}  ({correction_success/correction_triggered*100:.1f}%)")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    models = {
        "LSTM":        (LinkStateLSTM(),        OUT_DIR / "best_lstm.pt"),
        "Transformer": (LinkStateTransformer(), OUT_DIR / "best_transformer.pt"),
    }

    for name, (model, path) in models.items():
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        for split in ("train", "val", "test"):
            run_pipeline(split, model, device, name)


if __name__ == "__main__":
    main()
