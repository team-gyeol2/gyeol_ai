#!/usr/bin/env python3
"""
multihop_relay.py
─────────────────
멀티홉 Relay: 그래프 기반 k-hop 경로 탐색

현재 구조(단일 relay 선택)를 BFS 기반 k-hop 경로 탐색으로 확장.

비교:
  - Direct : 직접 링크 (hop=1)
  - Single : 단일 중계 UAV (hop=2)
  - Multi  : k-hop BFS 최단 경로 (hop≤4)

측정 지표:
  - 연결 성공률: 각 방식으로 경로를 찾을 수 있는 타임스텝 비율
  - 평균 hop 수
  - 단일 relay 실패 → 멀티홉 복구 케이스 수

실행:
    python3 multihop_relay.py
"""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"

NUM_UAVS  = 5
MAX_HOPS  = 4   # 최대 허용 hop 수


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_snapshots() -> dict[tuple, dict[tuple, str]]:
    """(scenario_id, time_s) → {(src, dst): link_state}"""
    snaps: dict[tuple, dict[tuple, str]] = defaultdict(dict)
    with open(DATA_DIR / "link_metrics.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key  = (r["scenario_id"], r["time_s"])
            pair = (int(r["src_uav"]), int(r["dst_uav"]))
            snaps[key][pair] = r["link_state"]
    return dict(snaps)


# ── 그래프 구성 ───────────────────────────────────────────────────────────────
def build_graph(link_states: dict[tuple, str]) -> dict[int, set[int]]:
    """non-disconnected 링크로 인접 그래프 구성 (양방향)."""
    adj: dict[int, set[int]] = defaultdict(set)
    for (a, b), state in link_states.items():
        if state != "disconnected":
            adj[a].add(b)
            adj[b].add(a)
    return dict(adj)


# ── 경로 탐색 ─────────────────────────────────────────────────────────────────
def bfs_path(adj: dict[int, set[int]], src: int, dst: int,
             max_hops: int) -> list[int] | None:
    """BFS로 src→dst 최단 경로 탐색. 없으면 None."""
    if dst in adj.get(src, set()):
        return [src, dst]
    queue  = deque([[src]])
    visited = {src}
    while queue:
        path = queue.popleft()
        if len(path) > max_hops:
            break
        for nb in adj.get(path[-1], set()):
            if nb in visited:
                continue
            new_path = path + [nb]
            if nb == dst:
                return new_path
            visited.add(nb)
            queue.append(new_path)
    return None


def single_relay_path(adj: dict[int, set[int]],
                      src: int, dst: int) -> list[int] | None:
    """단일 중계: src와 dst 모두에 연결된 UAV 탐색 (hop=2)."""
    src_nb = adj.get(src, set())
    dst_nb = adj.get(dst, set())
    relays = (src_nb & dst_nb) - {src, dst}
    if relays:
        relay = next(iter(relays))
        return [src, relay, dst]
    return None


# ── 평가 ─────────────────────────────────────────────────────────────────────
def evaluate(snapshots: dict) -> None:
    # 시나리오별 집계
    results: dict[str, dict] = defaultdict(lambda: {
        "total": 0,
        "direct_ok": 0, "single_ok": 0, "multi_ok": 0,
        "single_fail_multi_ok": 0,
        "hop_counts": [],
    })

    for (sid, t_s), link_states in snapshots.items():
        adj = build_graph(link_states)

        # 모든 UAV 쌍 평가
        for src in range(NUM_UAVS):
            for dst in range(src + 1, NUM_UAVS):
                r = results[sid]
                r["total"] += 1

                direct_ok = link_states.get((src, dst), "disconnected") != "disconnected"
                if direct_ok:
                    r["direct_ok"] += 1
                    r["single_ok"] += 1
                    r["multi_ok"]  += 1
                    r["hop_counts"].append(1)
                    continue

                # single relay
                s_path = single_relay_path(adj, src, dst)
                if s_path:
                    r["single_ok"] += 1
                    r["multi_ok"]  += 1
                    r["hop_counts"].append(len(s_path) - 1)
                    continue

                # multi-hop BFS
                m_path = bfs_path(adj, src, dst, MAX_HOPS)
                if m_path:
                    r["multi_ok"] += 1
                    r["single_fail_multi_ok"] += 1
                    r["hop_counts"].append(len(m_path) - 1)

    # 출력
    print("=" * 70)
    print(f"  멀티홉 Relay 평가  (max_hops={MAX_HOPS})")
    print("=" * 70)
    print(f"  {'시나리오':<28}  {'Direct':>8}  {'Single':>8}  {'Multi':>8}  {'단일실패→멀티복구':>16}")
    print("  " + "-" * 70)

    total_all = direct_all = single_all = multi_all = recovery_all = 0

    for sid in sorted(results):
        r   = results[sid]
        tot = r["total"]
        if tot == 0:
            continue
        d_r = r["direct_ok"] / tot * 100
        s_r = r["single_ok"] / tot * 100
        m_r = r["multi_ok"]  / tot * 100
        rec = r["single_fail_multi_ok"]
        print(f"  {sid:<28}  {d_r:>7.1f}%  {s_r:>7.1f}%  {m_r:>7.1f}%  {rec:>16d}건")
        total_all    += tot
        direct_all   += r["direct_ok"]
        single_all   += r["single_ok"]
        multi_all    += r["multi_ok"]
        recovery_all += r["single_fail_multi_ok"]

    print("  " + "-" * 70)
    if total_all:
        print(f"  {'전체 평균':<28}  "
              f"{direct_all/total_all*100:>7.1f}%  "
              f"{single_all/total_all*100:>7.1f}%  "
              f"{multi_all/total_all*100:>7.1f}%  "
              f"{recovery_all:>16d}건")

    # 전체 hop 분포
    all_hops = []
    for r in results.values():
        all_hops.extend(r["hop_counts"])
    if all_hops:
        hop_arr = np.array(all_hops)
        print(f"\n  [Hop 분포 — 연결 성공 케이스]")
        for h in range(1, MAX_HOPS + 1):
            cnt = (hop_arr == h).sum()
            print(f"    {h}-hop : {cnt:>6d}건  ({cnt/len(hop_arr)*100:.1f}%)")
        print(f"    평균   : {hop_arr.mean():.2f} hop")

    print(f"\n  [결론]")
    if total_all:
        gain = (multi_all - single_all) / total_all * 100
        print(f"    멀티홉 도입으로 연결 성공률 +{gain:.2f}%p 향상")
        print(f"    단일 relay 실패 케이스 중 {recovery_all}건 멀티홉으로 복구")


def main():
    print("데이터 로딩 중...")
    snapshots = load_snapshots()
    print(f"스냅샷 수: {len(snapshots)}")
    evaluate(snapshots)


if __name__ == "__main__":
    main()
