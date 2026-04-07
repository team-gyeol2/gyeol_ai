#!/usr/bin/env python3
"""
position_correction.py
──────────────────────
물리적 위치 보정 알고리즘

파이프라인 3단계:
  ① LSTM: link_state 예측
  ② relay 전환 (disconnected 시)
  ③ [본 모듈] relay 전환 후에도 disconnected가 남으면
     드론을 물리적으로 이동시켜 전체 연결성 복원

알고리즘:
  - 연결 그래프에서 고립된 UAV 탐지
  - 고립 UAV → 클러스터 중심 방향으로 STEP_M씩 이동
  - 이동 후 link_state 재계산 → 반복 (최대 MAX_STEPS)
  - 목표: 모든 UAV 쌍이 disconnected 아닌 상태 달성

RSSI 모델 (generate_uav_2d_dataset.py와 동일):
  rssi = -46.0 - 20*log10(dist) - atten
  healthy:  rssi >= -78 AND plr <= 5
  degraded: rssi >= -85 AND plr <= 20
  disconnected: otherwise

실행:
    python3 position_correction.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"

# ── 파라미터 ──────────────────────────────────────────────────────────────────
STEP_M     = 5.0   # 한 번 이동 거리 (m)
MAX_STEPS  = 50    # 최대 반복 횟수
NUM_UAVS   = 5

HEALTHY_RSSI   = -78.0
DEGRADED_RSSI  = -85.0
HEALTHY_PLR    = 5.0
DEGRADED_PLR   = 20.0


# ── LSTM 모델 (파이프라인용) ──────────────────────────────────────────────────
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


# ── 물리 모델 (링크 상태 계산) ────────────────────────────────────────────────
def _dist(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def _link_state_from_dist(dist: float) -> str:
    """건물 없다고 가정한 단순 거리 기반 링크 상태."""
    rssi = -46.0 - 20.0 * math.log10(max(dist, 1.0))
    plr  = min(0.8 + max(0.0, dist - 15.0) * 0.24, 95.0)
    if rssi >= HEALTHY_RSSI  and plr <= HEALTHY_PLR:
        return "healthy"
    if rssi >= DEGRADED_RSSI and plr <= DEGRADED_PLR:
        return "degraded"
    return "disconnected"


def _all_link_states(positions: dict[int, tuple]) -> dict[tuple, str]:
    """모든 UAV 쌍의 link_state 계산."""
    states = {}
    uav_ids = sorted(positions)
    for i, a in enumerate(uav_ids):
        for b in uav_ids[i+1:]:
            d = _dist(positions[a], positions[b])
            states[(a, b)] = _link_state_from_dist(d)
    return states


def _connected_components(positions: dict[int, tuple]) -> list[set[int]]:
    """연결된 UAV 클러스터 반환 (disconnected 제외)."""
    adj = defaultdict(set)
    for (a, b), state in _all_link_states(positions).items():
        if state != "disconnected":
            adj[a].add(b)
            adj[b].add(a)

    visited = set()
    components = []
    for uid in positions:
        if uid not in visited:
            comp = set()
            stack = [uid]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.add(node)
                stack.extend(adj[node] - visited)
            components.append(comp)
    return components


def _centroid(positions: dict[int, tuple], uav_ids: set[int]) -> tuple:
    xs = [positions[u][0] for u in uav_ids]
    ys = [positions[u][1] for u in uav_ids]
    return (sum(xs)/len(xs), sum(ys)/len(ys))


# ── 위치 보정 알고리즘 ────────────────────────────────────────────────────────
def correct_positions(positions: dict[int, tuple]) -> dict:
    """
    모든 UAV가 연결될 때까지 고립 UAV를 주 클러스터 방향으로 이동.

    반환:
        {
            "success": bool,
            "steps": int,
            "final_positions": dict,
            "moves": dict[uav_id -> total_distance_m],
            "final_states": dict,
        }
    """
    pos   = {uid: list(p) for uid, p in positions.items()}
    moves = defaultdict(float)

    for step in range(MAX_STEPS):
        comps = _connected_components({uid: tuple(p) for uid, p in pos.items()})

        if len(comps) == 1:
            # 모든 UAV 연결됨
            final_pos    = {uid: tuple(p) for uid, p in pos.items()}
            final_states = _all_link_states(final_pos)
            return {"success": True, "steps": step,
                    "final_positions": final_pos,
                    "moves": dict(moves),
                    "final_states": final_states}

        # 가장 큰 클러스터 = 주 클러스터
        main_comp = max(comps, key=len)
        target    = _centroid({uid: tuple(p) for uid, p in pos.items()}, main_comp)

        # 고립 UAV들을 주 클러스터 방향으로 STEP_M씩 이동
        for comp in comps:
            if comp == main_comp:
                continue
            # 고립 클러스터 자체 중심
            iso_center = _centroid({uid: tuple(p) for uid, p in pos.items()}, comp)
            dx = target[0] - iso_center[0]
            dy = target[1] - iso_center[1]
            d  = math.sqrt(dx**2 + dy**2)
            if d < 0.01:
                continue
            step_x = (dx / d) * STEP_M
            step_y = (dy / d) * STEP_M

            for uid in comp:
                pos[uid][0] += step_x
                pos[uid][1] += step_y
                moves[uid]  += STEP_M

    final_pos    = {uid: tuple(p) for uid, p in pos.items()}
    final_states = _all_link_states(final_pos)
    return {"success": False, "steps": MAX_STEPS,
            "final_positions": final_pos,
            "moves": dict(moves),
            "final_states": final_states}


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_positions() -> dict[tuple, dict[int, tuple]]:
    """(scenario_id, time_s) → {uav_id: (x, y)}"""
    result = defaultdict(dict)
    with open(DATA_DIR / "uav_positions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["scenario_id"], r["time_s"])
            result[key][int(r["uav_id"])] = (float(r["x_m"]), float(r["y_m"]))
    return dict(result)


def load_link_metrics() -> dict[tuple, str]:
    """(scenario_id, time_s, src, dst) → link_state"""
    result = {}
    with open(DATA_DIR / "link_metrics.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["scenario_id"], r["time_s"], int(r["src_uav"]), int(r["dst_uav"]))
            result[key] = r["link_state"]
    return result


# ── 스냅샷 평가 ───────────────────────────────────────────────────────────────
def evaluate_snapshot(scenario_id: str, time_s: str,
                      positions: dict[int, tuple],
                      link_states: dict[tuple, str]) -> None:
    """특정 타임스텝에서 위치 보정 전후 비교 출력."""

    disc_pairs = [(a, b) for (a, b), s in link_states.items() if s == "disconnected"]

    print(f"\n{'─'*55}")
    print(f"시나리오: {scenario_id}  |  t = {float(time_s):.2f}s")
    print(f"{'─'*55}")

    if not disc_pairs:
        print("  모든 링크 정상 — 위치 보정 불필요")
        return

    # 보정 전 연결 컴포넌트
    comps_before = _connected_components(positions)
    print(f"  [보정 전] disconnected 쌍: {len(disc_pairs)}개  |  "
          f"컴포넌트 수: {len(comps_before)}")
    for a, b in disc_pairs:
        d = _dist(positions[a], positions[b])
        print(f"    UAV{a}↔UAV{b}  거리={d:.1f}m")

    if len(comps_before) == 1:
        print("  이미 전체 연결됨 (degraded만 존재) — 보정 스킵")
        return

    # 위치 보정 실행
    result = correct_positions(positions)

    comps_after = _connected_components(result["final_positions"])
    disc_after  = [(a,b) for (a,b), s in result["final_states"].items()
                   if s == "disconnected"]

    print(f"\n  [보정 후] 이동 횟수: {result['steps']}회  "
          f"({'성공' if result['success'] else '실패'})")
    for uid, moved in sorted(result["moves"].items()):
        if moved > 0:
            ox, oy = positions[uid]
            nx, ny = result["final_positions"][uid]
            print(f"    UAV{uid}  {moved:.1f}m 이동  "
                  f"({ox:.1f},{oy:.1f}) → ({nx:.1f},{ny:.1f})")

    print(f"  disconnected 쌍: {len(disc_after)}개  |  "
          f"컴포넌트 수: {len(comps_after)}")
    if result["success"]:
        print("  ✓ 전체 연결 복원 완료")
    else:
        print(f"  ✗ {MAX_STEPS}회 후에도 일부 미연결 — 추가 이동 필요")


# ── 전체 실행 ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  물리적 위치 보정 알고리즘")
    print(f"  이동 단위: {STEP_M}m/step  |  최대: {MAX_STEPS}회")
    print("=" * 55)

    positions_db  = load_positions()
    link_state_db = load_link_metrics()

    # (scenario_id, time_s) → disconnected 쌍 수 집계
    snap_disc: dict[tuple, int] = defaultdict(int)
    for (sid, t, src, dst), state in link_state_db.items():
        if state == "disconnected":
            snap_disc[(sid, t)] += 1

    # 각 시나리오에서 disconnected가 가장 많은 타임스텝 1개씩 선택
    best_snaps: dict[str, tuple] = {}
    for (sid, t), cnt in snap_disc.items():
        if sid not in best_snaps or cnt > snap_disc[best_snaps[sid]]:
            best_snaps[sid] = (sid, t)

    # 통계 수집용
    total_events = corrected = failed = 0

    for sid, (_, t) in sorted(best_snaps.items()):
        pos_key = (sid, t)
        if pos_key not in positions_db:
            continue
        positions = positions_db[pos_key]

        # 해당 스냅샷 링크 상태
        link_states = {
            (src, dst): state
            for (s, ts, src, dst), state in link_state_db.items()
            if s == sid and ts == t
        }

        comps = _connected_components(positions)
        if len(comps) <= 1:
            continue  # 이미 연결됨

        total_events += 1
        result = correct_positions(positions)
        if result["success"]:
            corrected += 1
        else:
            failed += 1

        evaluate_snapshot(sid, t, positions, link_states)

    print(f"\n{'='*55}")
    print(f"  위치 보정 결과 요약")
    print(f"{'='*55}")
    print(f"  보정 대상 이벤트: {total_events}건")
    print(f"  복원 성공       : {corrected}건  ({100*corrected/max(total_events,1):.1f}%)")
    print(f"  복원 실패       : {failed}건")
    print(f"  이동 단위       : {STEP_M}m/step  |  최대 {MAX_STEPS}회")


if __name__ == "__main__":
    main()
