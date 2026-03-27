#!/usr/bin/env python3
"""
compare_ns3_python.py
─────────────────────
Python 데이터셋(link_metrics.csv)과 NS-3 시뮬레이션 결과(ns3_link_metrics.csv)를
relay_stretch 시나리오 기준으로 비교합니다.

실행 방법:
    cd /Users/inyoung/Desktop/캡디1/ns3
    python3 compare_ns3_python.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import csv

ROOT      = Path(__file__).resolve().parent
PY_CSV    = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial" / "link_metrics.csv"
NS3_CSV   = ROOT / "ns-3.47" / "ns3_link_metrics.csv"


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(row: dict, time_col: str = "time_s") -> tuple:
    return (float(row[time_col]), int(row["src_uav"]), int(row["dst_uav"]))


def main() -> None:
    # ── 파일 존재 확인 ──────────────────────────────────────────────────────
    if not PY_CSV.exists():
        sys.exit(f"[오류] Python 데이터셋이 없습니다: {PY_CSV}")
    if not NS3_CSV.exists():
        sys.exit(f"[오류] NS-3 출력 파일이 없습니다: {NS3_CSV}\n"
                 "       먼저 NS-3 시뮬레이션을 실행하세요:\n"
                 "         cd ns-3.47 && ./ns3 run uav-scenario-validation")

    # ── 데이터 로드 ────────────────────────────────────────────────────────
    py_rows  = [r for r in load_csv(PY_CSV)  if r.get("scenario_id") == "relay_stretch"]
    ns3_rows = load_csv(NS3_CSV)

    print(f"Python 행수  (relay_stretch): {len(py_rows)}")
    print(f"NS-3   행수               : {len(ns3_rows)}\n")

    if not py_rows or not ns3_rows:
        sys.exit("[오류] 데이터가 비어 있습니다.")

    # ── 키(time_s, src, dst)로 인덱싱 ─────────────────────────────────────
    py_index  = {key(r): r for r in py_rows}
    ns3_index = {key(r): r for r in ns3_rows}

    common_keys = sorted(set(py_index) & set(ns3_index))
    print(f"공통 (time_s, src, dst) 쌍: {len(common_keys)}\n")

    if not common_keys:
        sys.exit("[오류] 두 파일 사이에 공통 키(time_s, src, dst)가 없습니다.\n"
                 "       time_s 값이 일치하는지 확인하세요.")

    # ── 비교 통계 ──────────────────────────────────────────────────────────
    rssi_diffs: list[float] = []
    plr_diffs:  list[float] = []
    state_agree = 0
    state_total = 0
    mismatches:  list[dict] = []

    for k in common_keys:
        py  = py_index[k]
        ns3 = ns3_index[k]

        py_rssi  = float(py["rssi_dbm_est"])
        ns3_rssi = float(ns3["rssi_dbm_ns3"])
        rssi_diffs.append(ns3_rssi - py_rssi)

        py_plr  = float(py["plr_pct_est"])
        ns3_plr_raw = ns3["plr_pct_ns3"]
        if ns3_plr_raw and float(ns3_plr_raw) >= 0.0:
            ns3_plr = float(ns3_plr_raw)
            plr_diffs.append(ns3_plr - py_plr)
        else:
            ns3_plr = float("nan")

        py_state  = py["link_state"]
        ns3_state = ns3["link_state_ns3"]

        state_total += 1
        if py_state == ns3_state:
            state_agree += 1
        else:
            mismatches.append({
                "time_s":     k[0],
                "src_uav":    k[1],
                "dst_uav":    k[2],
                "dist_m":     round(float(ns3["distance_m"]), 1),
                "rssi_py":    round(py_rssi, 1),
                "rssi_ns3":   round(ns3_rssi, 1),
                "plr_py":     round(py_plr, 1),
                "plr_ns3":    round(ns3_plr, 1) if ns3_plr == ns3_plr else -1.0,
                "state_py":   py_state,
                "state_ns3":  ns3_state,
            })

    def _mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else float("nan")

    def _abs_mean(lst: list[float]) -> float:
        return sum(abs(x) for x in lst) / len(lst) if lst else float("nan")

    # ── 결과 출력 ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  RSSI 비교 (NS-3 − Python)")
    print("=" * 60)
    print(f"  평균 차이 : {_mean(rssi_diffs):+.2f} dBm")
    print(f"  MAE      : {_abs_mean(rssi_diffs):.2f} dBm")
    print()

    if plr_diffs:
        print("=" * 60)
        print("  PLR 비교 (NS-3 − Python)  [데이터 있는 행만]")
        print("=" * 60)
        print(f"  평균 차이 : {_mean(plr_diffs):+.2f} %")
        print(f"  MAE      : {_abs_mean(plr_diffs):.2f} %")
        print()

    agree_pct = state_agree / state_total * 100 if state_total else 0.0
    print("=" * 60)
    print("  link_state 일치율")
    print("=" * 60)
    print(f"  일치 : {state_agree} / {state_total}  ({agree_pct:.1f}%)")
    print()

    # 상태별 분포
    py_counts  = {"healthy": 0, "degraded": 0, "disconnected": 0}
    ns3_counts = {"healthy": 0, "degraded": 0, "disconnected": 0}
    for k in common_keys:
        ps = py_index[k]["link_state"]
        ns = ns3_index[k]["link_state_ns3"]
        py_counts[ps]  = py_counts.get(ps,  0) + 1
        ns3_counts[ns] = ns3_counts.get(ns, 0) + 1

    print(f"  {'상태':<14} {'Python':>8} {'NS-3':>8}")
    print(f"  {'-'*32}")
    for s in ("healthy", "degraded", "disconnected"):
        print(f"  {s:<14} {py_counts.get(s,0):>8} {ns3_counts.get(s,0):>8}")
    print()

    # 불일치 목록 (최대 20개)
    if mismatches:
        print("=" * 60)
        print(f"  link_state 불일치 목록 (총 {len(mismatches)}건, 최대 20건 표시)")
        print("=" * 60)
        hdr = f"  {'t':>5} {'src':>3} {'dst':>3} {'dist':>7} {'rssi_py':>8} {'rssi_ns3':>9} {'plr_py':>7} {'plr_ns3':>8} {'py→ns3'}"
        print(hdr)
        print(f"  {'-'*65}")
        for m in mismatches[:20]:
            print(f"  {m['time_s']:>5.1f} {m['src_uav']:>3} {m['dst_uav']:>3} "
                  f"{m['dist_m']:>7.1f} "
                  f"{m['rssi_py']:>8.1f} {m['rssi_ns3']:>9.1f} "
                  f"{m['plr_py']:>7.1f} {m['plr_ns3']:>8.1f} "
                  f"  {m['state_py']} → {m['state_ns3']}")
    else:
        print("  불일치 없음 — 두 모델이 완전히 일치합니다!")

    print()
    print("[완료] 비교 결과 출력 완료")


if __name__ == "__main__":
    main()
