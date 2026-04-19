#!/usr/bin/env python3
"""
sliding_window.py
─────────────────
두 가지 데이터를 생성합니다:

[A] LSTM용 — 1쌍 시계열 슬라이딩 윈도우
    입력: (N, 20, 5)  — 5초 과거 시계열, 5개 feature
    타겟: y_state (0=healthy, 1=degraded, 2=disconnected)
    파일: X_train.npy, y_state_train.npy

[B] relay 분류기용 — 네트워크 스냅샷
    입력: (M, 51)  — 10쌍 × 5 feature + LSTM 예측값(placeholder=0) 1개
    타겟: y_relay (0~4, optimal_relay_uav)
    파일: Xr_train.npy, y_relay_train.npy

실행:
    python3 sliding_window.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"

FEATURES    = ["rssi_dbm_est", "snr_db_est", "plr_pct_est", "throughput_mbps_est",
               "distance_m", "hop_count", "blocked_building_count"]
WINDOW_SIZE = 20   # 20 × 0.25s = 5초
PAIR_ORDER  = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── [A] LSTM용 슬라이딩 윈도우 ────────────────────────────────────────────────
def make_lstm_windows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    시나리오 + UAV 페어 단위로 독립적인 슬라이딩 윈도우 생성.
    Returns:
        X:       (N, WINDOW_SIZE, 5)
        y_state: (N,)
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["scenario_id"], r["src_uav"], r["dst_uav"])
        groups[key].append(r)

    X_list, y_list = [], []
    for key, grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))
        features = np.array([[float(r[f]) for f in FEATURES] for r in grp_sorted],
                             dtype=np.float32)
        labels   = np.array([int(r["link_state"]) for r in grp_sorted], dtype=np.int64)

        n = len(grp_sorted)
        if n < WINDOW_SIZE:
            continue
        for i in range(n - WINDOW_SIZE + 1):
            X_list.append(features[i : i + WINDOW_SIZE])
            y_list.append(labels[i + WINDOW_SIZE - 1])

    if not X_list:
        return np.empty((0, WINDOW_SIZE, len(FEATURES))), np.empty((0,))
    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.int64)


# ── [B] relay 분류기용 스냅샷 ─────────────────────────────────────────────────
def make_relay_snapshots(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    타임스텝별 네트워크 전체 스냅샷 생성.
    10쌍 × 5 feature = 50 + LSTM 예측값 placeholder 1 = 51개 feature.
    (LSTM 예측값은 추론 시 채워짐. 학습 시에는 실제 link_state 최악값 사용)

    Returns:
        Xr: (M, 51)
        y_relay: (M,)
    """
    # (scenario_id, time_s) → {(src,dst): row}
    snapshots: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        key  = (r["scenario_id"], float(r["time_s"]))
        pair = (int(r["src_uav"]), int(r["dst_uav"]))
        snapshots[key][pair] = r

    Xr_list, yr_list = [], []
    for (sid, t), snap in snapshots.items():
        if len(snap) < len(PAIR_ORDER):
            continue

        # 10쌍 feature 이어붙이기 (50개)
        pair_feats = []
        for pair in PAIR_ORDER:
            r = snap[pair]
            pair_feats.extend([float(r[f]) for f in FEATURES])

        # LSTM 예측값 placeholder: 실제 네트워크 최악 link_state 사용
        states = [int(snap[p]["link_state"]) for p in PAIR_ORDER]
        lstm_pred = float(max(states))  # 0=healthy, 1=degraded, 2=disconnected

        Xr_list.append(pair_feats + [lstm_pred])

        # 타겟: optimal_relay_uav (모든 쌍에서 동일)
        yr_list.append(int(next(iter(snap.values()))["optimal_relay_uav"]))

    if not Xr_list:
        return np.empty((0, 51)), np.empty((0,))
    return np.array(Xr_list, dtype=np.float32), np.array(yr_list, dtype=np.int64)


def label_dist(y: np.ndarray) -> str:
    total = len(y)
    names = {0: "healthy", 1: "degraded", 2: "disconnected"}
    return "  ".join(f"{names.get(k,k)}:{int((y==k).sum())}({int((y==k).sum())/total*100:.1f}%)"
                     for k in range(3))

def relay_dist(y: np.ndarray) -> str:
    total = len(y)
    return "  ".join(f"UAV{k}:{int((y==k).sum())}({int((y==k).sum())/total*100:.1f}%)"
                     for k in range(5))


def main() -> None:
    print("=== [A] LSTM용 슬라이딩 윈도우 ===")
    for split in ("train", "val", "test"):
        rows = load_csv(DATA_DIR / f"{split}.csv")
        X, y_state = make_lstm_windows(rows)
        np.save(DATA_DIR / f"X_{split}.npy",      X)
        np.save(DATA_DIR / f"y_state_{split}.npy", y_state)
        print(f"  {split:<6} X:{X.shape}  {label_dist(y_state)}")

    print()
    print("=== [B] relay 분류기용 스냅샷 ===")
    for split in ("train", "val", "test"):
        rows = load_csv(DATA_DIR / f"{split}.csv")
        Xr, y_relay = make_relay_snapshots(rows)
        np.save(DATA_DIR / f"Xr_{split}.npy",      Xr)
        np.save(DATA_DIR / f"y_relay_{split}.npy",  y_relay)
        print(f"  {split:<6} Xr:{Xr.shape}  {relay_dist(y_relay)}")

    print("\n저장 완료")


if __name__ == "__main__":
    main()
