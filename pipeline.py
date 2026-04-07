#!/usr/bin/env python3
"""
pipeline.py
───────────
통신 장애 예측 + 토폴로지 재구성 파이프라인

① LSTM / Transformer: 링크 시계열 → link_state 예측 (healthy/degraded/disconnected)
② 규칙 기반 relay 선택:
   - healthy / degraded → 현재 relay 유지
   - disconnected       → 평균 RSSI 최고 UAV로 relay 전환

평가 지표:
   - link_state: Accuracy, Precision, Recall, F1-score (macro), Confusion Matrix
   - relay 선택 Accuracy
   - 파이프라인 전체 Accuracy

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

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"

FEATURES    = ["rssi_dbm_est", "plr_pct_est", "distance_m",
               "hop_count", "blocked_building_count"]
PAIR_ORDER  = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
WINDOW_SIZE = 20
STATE_NAMES = ["healthy", "degraded", "disconnected"]


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
class LinkStateLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(input_size=5, hidden_size=64,
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
        self.input_proj = nn.Linear(5, 64)
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


# ── 규칙 기반 relay 선택 ──────────────────────────────────────────────────────
def rule_based_relay(snapshot: dict, current_relay: int) -> int:
    rssi_sum   = defaultdict(float)
    rssi_count = defaultdict(int)
    for (src, dst), r in snapshot.items():
        rssi_val = float(r["rssi_dbm_est"])
        rssi_sum[src]   += rssi_val; rssi_count[src] += 1
        rssi_sum[dst]   += rssi_val; rssi_count[dst] += 1
    avg_rssi = {uid: rssi_sum[uid] / rssi_count[uid]
                for uid in rssi_sum if rssi_count[uid] > 0}
    if not avg_rssi:
        return current_relay
    return max(avg_rssi, key=avg_rssi.get)


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

    model.eval()
    y_true_state, y_pred_state = [], []
    relay_correct = pipeline_correct = total = 0

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
