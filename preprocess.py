#!/usr/bin/env python3
"""
preprocess.py
─────────────
1. 5개 feature 추출 (rssi_dbm_est, plr_pct_est, distance_m, hop_count, blocked_building_count)
2. 시나리오별 시간 기준 70/15/15 분할 → train / val / test
3. train 기준 StandardScaler fit → val/test에 transform만 적용 (data leakage 방지)
4. 결과 저장: datasets/uav_2d_initial/train.csv, val.csv, test.csv
             scaler 파라미터: datasets/uav_2d_initial/scaler_params.json

실행:
    python3 preprocess.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"

FEATURES = [
    "rssi_dbm_est",
    "plr_pct_est",
    "distance_m",
    "hop_count",
    "blocked_building_count",
]
TARGET      = "link_state"
LABEL_MAP   = {"healthy": 0, "degraded": 1, "disconnected": 2}
RELAY_TARGET = "optimal_relay_uav"  # 0~4 정수, 그대로 사용

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = 나머지 0.15

# relay 전환 시나리오는 val/test에 전환 구간이 더 많이 들어가도록 별도 비율 적용
RELAY_SCENARIOS = {"relay_handover", "relay_rotation", "relay_competition"}
RELAY_TRAIN_RATIO = 0.50
RELAY_VAL_RATIO   = 0.25
# relay test = 나머지 0.25


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_scenario(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """시간 기준으로 한 시나리오의 rows를 train/val/test로 분할.
    relay 전환 시나리오는 50/25/25 비율 적용."""
    sid = rows[0]["scenario_id"]
    is_relay = sid in RELAY_SCENARIOS
    tr = RELAY_TRAIN_RATIO if is_relay else TRAIN_RATIO
    va = RELAY_VAL_RATIO   if is_relay else VAL_RATIO

    times = sorted(set(float(r["time_s"]) for r in rows))
    n = len(times)
    train_end_idx = int(n * tr)
    val_end_idx   = int(n * (tr + va))

    train_times = set(times[:train_end_idx])
    val_times   = set(times[train_end_idx:val_end_idx])
    test_times  = set(times[val_end_idx:])

    train = [r for r in rows if float(r["time_s"]) in train_times]
    val   = [r for r in rows if float(r["time_s"]) in val_times]
    test  = [r for r in rows if float(r["time_s"]) in test_times]
    return train, val, test


def compute_scaler_params(rows: list[dict]) -> dict:
    """train 데이터에서 각 feature의 mean, std 계산."""
    params = {}
    for f in FEATURES:
        vals = [float(r[f]) for r in rows]
        mean = sum(vals) / len(vals)
        std  = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        std  = std if std > 1e-8 else 1.0  # 0 나눔 방지
        params[f] = {"mean": mean, "std": std}
    return params


def apply_scaling(rows: list[dict], params: dict) -> list[dict]:
    """StandardScaler 적용 (transform only)."""
    scaled = []
    for r in rows:
        new_r = {
            "scenario_id": r["scenario_id"],
            "time_s":      r["time_s"],
            "src_uav":     r["src_uav"],
            "dst_uav":     r["dst_uav"],
        }
        for f in FEATURES:
            val = float(r[f])
            new_r[f] = round((val - params[f]["mean"]) / params[f]["std"], 6)
        new_r[TARGET]       = LABEL_MAP[r[TARGET]]
        new_r[RELAY_TARGET] = int(r[RELAY_TARGET])
        scaled.append(new_r)
    return scaled


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("데이터 로딩 중…")
    all_rows = load_csv(DATA_DIR / "link_metrics.csv")
    print(f"  총 {len(all_rows)}행 로드 완료")

    # 시나리오별로 그룹화
    sc_rows: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        sc_rows[r["scenario_id"]].append(r)

    # 시나리오별 분할
    print("\n시나리오별 시간 기준 분할 (70/15/15)…")
    all_train, all_val, all_test = [], [], []
    for sid in sorted(sc_rows):
        tr, va, te = split_scenario(sc_rows[sid])
        all_train.extend(tr)
        all_val.extend(va)
        all_test.extend(te)
        print(f"  {sid:<24} train={len(tr):>5}  val={len(va):>5}  test={len(te):>5}")

    print(f"\n  전체 train={len(all_train)}  val={len(all_val)}  test={len(all_test)}")

    # train 기준 scaler 파라미터 계산
    print("\ntrain 기준 StandardScaler 파라미터 계산 중…")
    scaler_params = compute_scaler_params(all_train)
    for f, p in scaler_params.items():
        print(f"  {f:<26} mean={p['mean']:>9.4f}  std={p['std']:>8.4f}")

    # scaler 파라미터 저장 (val/test 추론 시 재사용)
    scaler_path = DATA_DIR / "scaler_params.json"
    with scaler_path.open("w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"\n  scaler 파라미터 저장: {scaler_path.name}")

    # 스케일링 적용
    print("\nStandard Scaling 적용 중…")
    train_scaled = apply_scaling(all_train, scaler_params)
    val_scaled   = apply_scaling(all_val,   scaler_params)
    test_scaled  = apply_scaling(all_test,  scaler_params)

    # 저장
    save_csv(train_scaled, DATA_DIR / "train.csv")
    save_csv(val_scaled,   DATA_DIR / "val.csv")
    save_csv(test_scaled,  DATA_DIR / "test.csv")

    print("\n저장 완료:")
    print(f"  train.csv  — {len(train_scaled)}행")
    print(f"  val.csv    — {len(val_scaled)}행")
    print(f"  test.csv   — {len(test_scaled)}행")

    # 레이블 분포 확인
    print("\n레이블 분포 (0=healthy, 1=degraded, 2=disconnected):")
    for split_name, rows in [("train", train_scaled), ("val", val_scaled), ("test", test_scaled)]:
        from collections import Counter
        cnt = Counter(r[TARGET] for r in rows)
        total = len(rows)
        dist = "  ".join(f"{k}:{v}({v/total*100:.0f}%)" for k, v in sorted(cnt.items()))
        print(f"  {split_name:<6} {dist}")


if __name__ == "__main__":
    main()
