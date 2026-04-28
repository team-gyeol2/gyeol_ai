#!/usr/bin/env python3
"""
optimize_relay_weights.py
─────────────────────────
Relay Score 가중합 함수의 7개 가중치를 Bayesian Optimization(Optuna TPE)으로 최적화.

목적함수: val 세트 기준 relay 선택 Accuracy 최대화
최적화된 가중치를 pipeline.py의 WEIGHTS와 비교하여 개선 효과 측정.

실행:
    python3 optimize_relay_weights.py
"""

from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"

FEATURES = ["rssi_dbm_est", "snr_db_est", "plr_pct_est",
            "throughput_mbps_est", "distance_m", "hop_count",
            "blocked_building_count"]
INVERT   = {"plr_pct_est", "distance_m", "hop_count", "blocked_building_count"}
PAIR_ORDER = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
WINDOW_SIZE    = 20
HYSTERESIS_THRESH = 2

# 현재(기존) 가중치
CURRENT_WEIGHTS = {
    "rssi_dbm_est":           0.20,
    "snr_db_est":             0.15,
    "plr_pct_est":            0.20,
    "throughput_mbps_est":    0.15,
    "distance_m":             0.10,
    "hop_count":              0.10,
    "blocked_building_count": 0.10,
}


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_snapshots(rows: list[dict]):
    """시나리오·타임스텝별로 링크 딕셔너리 구성."""
    snaps: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
    for r in rows:
        key = (r["scenario_id"], r["time_s"])
        pair = (int(r["src_uav"]), int(r["dst_uav"]))
        snaps[key][pair] = r
    return snaps


# ── Relay Score 계산 ──────────────────────────────────────────────────────────
def normalize(val: float, f: str, f_min: float, f_max: float) -> float:
    rng = f_max - f_min
    n = (val - f_min) / rng if rng > 1e-8 else 0.5
    return (1.0 - n) if f in INVERT else n


def select_relay(snapshot: dict, weights: dict[str, float], current_relay: int) -> int:
    feat_sum: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    feat_cnt: dict[int, int] = defaultdict(int)
    for (src, dst), r in snapshot.items():
        for f in FEATURES:
            val = float(r.get(f, 0.0))
            feat_sum[src][f] += val
            feat_sum[dst][f] += val
        feat_cnt[src] += 1
        feat_cnt[dst]  += 1

    if not feat_cnt:
        return current_relay

    uav_avg = {
        uid: {f: feat_sum[uid][f] / feat_cnt[uid] for f in FEATURES}
        for uid in feat_cnt
    }
    f_min = {f: min(uav_avg[u][f] for u in uav_avg) for f in FEATURES}
    f_max = {f: max(uav_avg[u][f] for u in uav_avg) for f in FEATURES}

    scores = {
        uid: sum(weights[f] * normalize(uav_avg[uid][f], f, f_min[f], f_max[f])
                 for f in FEATURES)
        for uid in uav_avg
    }
    return max(scores, key=scores.get)


# ── 평가 함수 ─────────────────────────────────────────────────────────────────
def evaluate(rows: list[dict], weights: dict[str, float]) -> float:
    snaps = build_snapshots(rows)
    keys  = sorted(snaps.keys())

    relay_correct = relay_total = 0
    bad_streak: dict[str, int] = defaultdict(int)
    last_relay:  dict[str, int] = {}

    for key in keys:
        scen = key[0]
        snap = snaps[key]
        if not snap:
            continue

        # 링크 상태 판단 (실제 레이블 사용 — relay 선택만 평가)
        any_disconnected = any(r["link_state"] == "disconnected" for r in snap.values())
        if any_disconnected:
            bad_streak[scen] += 1
        else:
            bad_streak[scen] = 0

        current_relay = last_relay.get(scen, 2)

        if bad_streak[scen] >= HYSTERESIS_THRESH:
            chosen = select_relay(snap, weights, current_relay)
            last_relay[scen] = chosen
        else:
            chosen = current_relay

        # 정답 relay
        true_relays = [int(r["optimal_relay_uav"]) for r in snap.values() if r.get("optimal_relay_uav")]
        if not true_relays:
            continue
        true_relay = max(set(true_relays), key=true_relays.count)

        relay_correct += (chosen == true_relay)
        relay_total   += 1

    return relay_correct / relay_total if relay_total > 0 else 0.0


# ── Optuna 목적함수 ───────────────────────────────────────────────────────────
def objective(trial: optuna.Trial, val_rows: list[dict]) -> float:
    # 7개 가중치를 0~1 범위에서 샘플링 후 합이 1이 되도록 정규화
    raw = {f: trial.suggest_float(f, 0.0, 1.0) for f in FEATURES}
    total = sum(raw.values())
    if total < 1e-8:
        return 0.0
    weights = {f: raw[f] / total for f in FEATURES}
    return evaluate(val_rows, weights)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("데이터 로딩 중...")
    val_rows  = load_csv(DATA_DIR / "val.csv")
    test_rows = load_csv(DATA_DIR / "test.csv")

    # 기존 가중치 성능 측정
    val_before  = evaluate(val_rows,  CURRENT_WEIGHTS)
    test_before = evaluate(test_rows, CURRENT_WEIGHTS)
    print(f"\n[기존 가중치]")
    for f, w in CURRENT_WEIGHTS.items():
        print(f"  {f:<28}: {w:.2f}")
    print(f"  Val  Relay Accuracy : {val_before*100:.2f}%")
    print(f"  Test Relay Accuracy : {test_before*100:.2f}%")

    # Bayesian Optimization
    print("\nBayesian Optimization 실행 중 (n_trials=200)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(lambda t: objective(t, val_rows), n_trials=200, show_progress_bar=False)

    # 최적 가중치 정규화
    best_raw = study.best_params
    total    = sum(best_raw.values())
    best_weights = {f: best_raw[f] / total for f in FEATURES}

    # 최적 가중치 성능 측정
    val_after  = evaluate(val_rows,  best_weights)
    test_after = evaluate(test_rows, best_weights)

    print(f"\n[최적화된 가중치]")
    for f in FEATURES:
        print(f"  {f:<28}: {best_weights[f]:.4f}")
    print(f"  Val  Relay Accuracy : {val_after*100:.2f}%  (기존 {val_before*100:.2f}%,  Δ{(val_after-val_before)*100:+.2f}%p)")
    print(f"  Test Relay Accuracy : {test_after*100:.2f}%  (기존 {test_before*100:.2f}%,  Δ{(test_after-test_before)*100:+.2f}%p)")

    print("\n[가중치 변화 비교]")
    print(f"  {'피처':<28}  {'기존':>6}  {'최적화':>8}  {'변화':>8}")
    print("  " + "-"*56)
    for f in FEATURES:
        diff = best_weights[f] - CURRENT_WEIGHTS[f]
        print(f"  {f:<28}  {CURRENT_WEIGHTS[f]:>6.4f}  {best_weights[f]:>8.4f}  {diff:>+8.4f}")


if __name__ == "__main__":
    main()
