#!/usr/bin/env python3
"""
multihop_system_test.py
───────────────────────
연결 성공률 지표를 '우리 시스템' 기준으로 재측정.

기존 multihop_relay.py 는 라우팅(BFS)만으로 연결 가능성을 봤다:
  Direct(1-hop) → Single Relay(2-hop) → Multi-hop BFS(≤4-hop)
이건 '경로가 존재하나'라서 PPO(릴레이 선택)와는 무관하다.

여기서는 그 위에 **DQN 위치보정 레이어**를 추가한다:
  Multi-hop 으로도 안 되는 단절 쌍(서로 다른 컴포넌트)에 대해,
  고립 UAV를 DQN(domain-randomized 모델)으로 물리 이동시켜 재연결 → 그래프에
  새 엣지 추가 → 다시 BFS 로 연결 가능 쌍을 센다.

  = "라우팅만으론 81.7% 가 천장, 우리 시스템은 드론을 움직여 더 복구한다"

실행:
    python3 multihop_system_test.py
"""
from __future__ import annotations
from collections import deque

import numpy as np
import torch

import multihop_relay as MH
import rl_position_correction as RC

NUM = 5
DQN_MODEL = RC.OUT_DIR / "rl_correction_dqn_domainrand.pt"   # 우리가 고친 DQN


def components(adj, nodes):
    seen, comps = set(), []
    for n in nodes:
        if n in seen:
            continue
        c, st = set(), [n]
        while st:
            x = st.pop()
            if x in seen:
                continue
            seen.add(x); c.add(x)
            st += [y for y in adj.get(x, ()) if y not in seen]
        comps.append(c)
    return comps


def run_dqn(policy, obstacles, iso_pos, main_pos):
    """고립 UAV를 DQN으로 이동시켜 최종 위치 반환."""
    sc = {"iso_pos": iso_pos, "main_pos": main_pos,
          "rssi": max(RC.compute_rssi(iso_pos[0], iso_pos[1], m[0], m[1], obstacles)
                      for m in main_pos)}
    env = RC.UavCorrectionEnv(obstacles, [sc])
    state = env.reset_specific(sc)
    for _ in range(RC.MAX_STEPS):
        with torch.no_grad():
            a = policy(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
        state, _, done = env.step(a)
        if done:
            break
    return (env.iso_x, env.iso_y)


def main():
    print("데이터 로딩...")
    obstacles, positions, links = RC._load_data()
    snaps = MH.load_snapshots()

    policy = RC.DQN()
    policy.load_state_dict(torch.load(DQN_MODEL, map_location="cpu"))
    policy.eval()
    print(f"DQN 모델: {DQN_MODEL.name}\n스냅샷: {len(snaps)}\n")

    tot = direct = single = multi = dqn = 0
    snaps_with_dqn = pairs_recovered = 0

    for key, ls in snaps.items():
        if key not in positions:
            continue
        pos = positions[key]
        adj = MH.build_graph(ls)
        comps = components(adj, list(range(NUM)))
        main = max(comps, key=len)

        # ── DQN 보정: 비주류 컴포넌트 UAV를 이동시켜 새 엣지 추가 ──
        aug = {k: set(v) for k, v in adj.items()}
        if len(comps) > 1:
            main_uids = [u for u in main if u in pos]
            main_pos = [pos[u] for u in main_uids]
            if main_pos:
                applied = False
                for comp in comps:
                    if comp == main:
                        continue
                    for u in comp:
                        if u not in pos:
                            continue
                        nx, ny = run_dqn(policy, obstacles, pos[u], main_pos)
                        for m in main_uids:
                            if RC.is_connected(nx, ny, pos[m][0], pos[m][1], obstacles):
                                aug.setdefault(u, set()).add(m)
                                aug.setdefault(m, set()).add(u)
                                applied = True
                if applied:
                    snaps_with_dqn += 1

        # ── 쌍별 연결 가능성 누적 ──
        for i in range(NUM):
            for j in range(i + 1, NUM):
                tot += 1
                if ls.get((i, j), "disconnected") != "disconnected":
                    direct += 1; single += 1; multi += 1; dqn += 1
                elif MH.single_relay_path(adj, i, j):
                    single += 1; multi += 1; dqn += 1
                elif MH.bfs_path(adj, i, j, MH.MAX_HOPS):
                    multi += 1; dqn += 1
                elif MH.bfs_path(aug, i, j, MH.MAX_HOPS):
                    dqn += 1; pairs_recovered += 1

    print("=" * 60)
    print("  연결 방식별 성공률 — 우리 시스템 기준 재측정")
    print("=" * 60)
    print(f"  {'연결 방식':<28}{'성능 수치':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Direct (1-hop)':<28}{direct/tot*100:>11.1f}%")
    print(f"  {'Single Relay (2-hop)':<28}{single/tot*100:>11.1f}%")
    print(f"  {'Multi-hop BFS (≤4-hop)':<28}{multi/tot*100:>11.1f}%")
    print(f"  {'+ DQN 위치보정':<28}{dqn/tot*100:>11.1f}%")
    print(f"  {'-'*44}")
    print(f"\n  Multi-hop 천장: {multi/tot*100:.1f}%  →  +DQN: {dqn/tot*100:.1f}%  "
          f"(+{(dqn-multi)/tot*100:.1f}%p)")
    print(f"  DQN 적용 스냅샷: {snaps_with_dqn}   복구된 단절 쌍: {pairs_recovered}")
    print(f"\n  ※ PPO/PPO v2 는 '릴레이 선택'이라 이 경로존재 지표를 바꾸지 않음(무관).")
    print(f"  ※ DQN 레이어는 이 데이터셋 시나리오 기준(DQN in-distribution).")


if __name__ == "__main__":
    main()
