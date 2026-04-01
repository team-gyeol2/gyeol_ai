#!/usr/bin/env python3
"""
sliding_window.py
─────────────────
train/val/test.csv를 읽어 슬라이딩 윈도우를 적용,
LSTM 입력 형태의 numpy 배열로 저장합니다.

윈도우 크기: 20 (0.25s × 20 = 5초 과거 참조)
레이블: 윈도우 마지막 타임스텝의 link_state
시나리오 경계, UAV 페어 경계를 넘는 윈도우는 생성하지 않음

출력:
    datasets/uav_2d_initial/X_train.npy  shape: (N, 20, 5)
    datasets/uav_2d_initial/y_train.npy  shape: (N,)
    (val, test 동일)

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

FEATURES     = ["rssi_dbm_est", "plr_pct_est", "distance_m",
                 "hop_count", "blocked_building_count"]
WINDOW_SIZE  = 20   # 20 × 0.25s = 5초


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_windows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    시나리오 + UAV 페어 단위로 독립적인 슬라이딩 윈도우 생성.
    Returns X: (N, WINDOW_SIZE, n_features), y: (N,)
    """
    # (scenario_id, src_uav, dst_uav) 별로 그룹화
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["scenario_id"], r["src_uav"], r["dst_uav"])
        groups[key].append(r)

    X_list, y_list = [], []

    for key, grp in groups.items():
        # 시간 순 정렬
        grp_sorted = sorted(grp, key=lambda r: float(r["time_s"]))

        features = np.array([[float(r[f]) for f in FEATURES] for r in grp_sorted],
                             dtype=np.float32)
        labels   = np.array([int(r["link_state"]) for r in grp_sorted], dtype=np.int64)

        n = len(grp_sorted)
        if n < WINDOW_SIZE:
            continue  # 윈도우보다 짧은 시퀀스는 건너뜀

        for i in range(n - WINDOW_SIZE + 1):
            X_list.append(features[i : i + WINDOW_SIZE])   # (20, 5)
            y_list.append(labels[i + WINDOW_SIZE - 1])     # 마지막 스텝 레이블

    if not X_list:
        return np.empty((0, WINDOW_SIZE, len(FEATURES))), np.empty((0,))

    X = np.stack(X_list).astype(np.float32)   # (N, 20, 5)
    y = np.array(y_list, dtype=np.int64)       # (N,)
    return X, y


def label_dist(y: np.ndarray) -> str:
    total = len(y)
    parts = []
    for cls, name in [(0, "healthy"), (1, "degraded"), (2, "disconnected")]:
        cnt = int((y == cls).sum())
        parts.append(f"{name}:{cnt}({cnt/total*100:.1f}%)")
    return "  ".join(parts)


def main() -> None:
    for split in ("train", "val", "test"):
        rows = load_csv(DATA_DIR / f"{split}.csv")
        X, y = make_windows(rows)

        np.save(DATA_DIR / f"X_{split}.npy", X)
        np.save(DATA_DIR / f"y_{split}.npy", y)

        print(f"{split:<6} X:{X.shape}  y:{y.shape}")
        print(f"       {label_dist(y)}")

    print("\n저장 완료:")
    for split in ("train", "val", "test"):
        for prefix in ("X", "y"):
            p = DATA_DIR / f"{prefix}_{split}.npy"
            print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")

    print(f"\n입력 shape 설명: (샘플수, 윈도우크기={WINDOW_SIZE}, feature수={len(FEATURES)})")
    print(f"  윈도우 1개 = {WINDOW_SIZE} × 0.25s = {WINDOW_SIZE * 0.25:.1f}초 과거 참조")


if __name__ == "__main__":
    main()
