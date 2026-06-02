#!/usr/bin/env python3
"""
ppo_v2_accuracy.py — PPO v2를 트랜스포머처럼 '%' 지표로 표현.

두 가지 발표용 백분율:
  ① 최적 중계기 선택 정확도(%) : 매 스텝, 그 시점 오라클(최선 릴레이)이 만들 수
     있는 연결쌍과 동일하게 달성한 비율. = "릴레이를 옳게 골랐나" 정확도.
  ② 연결 유지율(%) : (실제 유지한 연결쌍) / (오라클이 유지 가능한 연결쌍).
     = "살릴 수 있는 연결을 얼마나 살렸나".

오라클 = 현재 위치·배터리에서 relay_pairs 를 최대화하는 릴레이.
"""
from __future__ import annotations
import random
import numpy as np
from sb3_contrib import RecurrentPPO

from rl_relay_agent_v2 import (
    RelayEnvV2, relay_pairs, heuristic_v2, rulebased_v2,
    N_UAVS, MAX_STEPS,
)


def oracle_pairs(pos, battery):
    return max(len(relay_pairs(pos, r, battery)) for r in range(N_UAVS))


def eval_pct(kind, model, env, n_ep=500):
    sel_hit = sel_tot = 0          # 선택 정확도
    kept = possible = 0            # 연결 유지율
    for _ in range(n_ep):
        obs, _ = env.reset()
        relay = random.randint(0, N_UAVS - 1)
        lstm, ep_start = None, True
        for _ in range(MAX_STEPS):
            orc = oracle_pairs(env.pos, env.battery)
            if kind == "ppo":
                a, lstm = model.predict(obs, state=lstm,
                                        episode_start=np.array([ep_start]),
                                        deterministic=True)
                a = int(a); ep_start = False
            elif kind == "heuristic":
                a = heuristic_v2(env.pos, env.battery, relay)
            else:
                a = rulebased_v2(env.pos, env.battery, relay)
            achieved = len(relay_pairs(env.pos, a, env.battery))
            if orc > 0:                      # 오라클이 살릴 게 있을 때만 채점
                sel_tot += 1
                sel_hit += int(achieved >= orc)
                possible += orc
                kept += achieved
            relay = a
            obs, _, _, trunc, _ = env.step(a)
            if trunc:
                break
    return {
        "select_acc": 100.0 * sel_hit / sel_tot if sel_tot else 0.0,
        "retention":  100.0 * kept / possible if possible else 0.0,
    }


def main():
    model = RecurrentPPO.load("models/ppo_relay_agent_v2.zip")
    env = RelayEnvV2()
    print(f"{'='*60}")
    print(f"  PPO v2 — 백분율 지표 (병목 토폴로지, 500 에피소드)")
    print(f"{'='*60}")
    print(f"  {'전략':<22}{'최적릴레이 선택정확도':>16}{'연결 유지율':>12}")
    print(f"  {'-'*52}")
    for name, kind in [("PPO v2 (RL)", "ppo"),
                       ("Heuristic", "heuristic"),
                       ("Rule-based", "rulebased")]:
        random.seed(0); np.random.seed(0)
        r = eval_pct(kind, model, env)
        print(f"  {name:<22}{r['select_acc']:>15.1f}%{r['retention']:>11.1f}%")
    print(f"  {'-'*52}")


if __name__ == "__main__":
    main()
