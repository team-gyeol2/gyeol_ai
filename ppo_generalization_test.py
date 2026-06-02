#!/usr/bin/env python3
"""
ppo_generalization_test.py
──────────────────────────
PPO 릴레이 선택 에이전트(models/ppo_relay_agent.zip)의 분포이동(OOD) 일반화 측정.

PPO는 매 에피소드 _random_positions()로 무작위 토폴로지를 생성해 학습하므로
'학습 분포' 자체가 랜덤이다(사실상 domain randomization). 따라서 일반화는
'학습 때 본 적 없는 분포'에서 측정해야 한다. 토폴로지 산포(eff factor)와
이동속도(SPEED)를 바꾼 OOD 환경에서 PPO vs Heuristic vs Rule-based 비교.

  in-dist  : eff=3.0×COMM_RANGE, speed=50  (학습과 동일)
  OOD-tight: eff=1.5×  (촘촘 — 대부분 연결)
  OOD-wide : eff=5.0×  (성김 — 대부분 단절)
  OOD-fast : speed=150 (3배 빠른 이동)

실행:
    python3 ppo_generalization_test.py
"""

from __future__ import annotations

import random
import numpy as np
from stable_baselines3 import PPO

import rl_relay_agent as R
from rl_relay_agent import (
    RelayEnv, COMM_RANGE, N_UAVS, N_LINKS, AREA_SIZE, MAX_STEPS,
    is_connected, heuristic_relay, rulebased_relay, ppo_strategy,
)


class RelayEnvVariant(RelayEnv):
    """산포(eff_factor)와 이동속도(speed)를 바꿀 수 있는 변형 환경."""
    def __init__(self, eff_factor=3.0, speed=50.0):
        super().__init__()
        self.eff_factor = eff_factor
        self.speed = speed

    def _random_positions(self):
        eff = COMM_RANGE * self.eff_factor
        margin = min(eff, AREA_SIZE)
        max_offset = max(AREA_SIZE - margin, 0.0)
        for _ in range(200):
            offset = np.random.uniform(0, max_offset, 2) if max_offset > 0 else np.zeros(2)
            pos = offset + np.random.uniform(0, margin, (N_UAVS, 2))
            pos = np.clip(pos, 0, AREA_SIZE)
            connected = sum(
                1 for i in range(N_UAVS) for j in range(i + 1, N_UAVS)
                if is_connected(pos[i][0], pos[i][1], pos[j][0], pos[j][1])
            )
            if 1 <= connected <= N_LINKS - 1:
                return pos
        center = np.array([AREA_SIZE / 2, AREA_SIZE / 2])
        return np.clip(center + np.random.uniform(-COMM_RANGE, COMM_RANGE, (N_UAVS, 2)),
                       0, AREA_SIZE)

    def _move_uavs(self):
        noise = np.random.uniform(-self.speed, self.speed, (N_UAVS, 2))
        self.pos = np.clip(self.pos + noise, 0, AREA_SIZE)


def evaluate_env(strategy_fn, env, n_episodes=500):
    total_pairs = total_switches = 0.0
    for _ in range(n_episodes):
        env.reset()
        relay = random.randint(0, N_UAVS - 1)
        ep_pairs, ep_switches = 0.0, 0
        for _ in range(MAX_STEPS):
            action = strategy_fn(env.pos, relay)
            if action != relay:
                ep_switches += 1
            relay = action
            _, _, _, truncated, info = env.step(action)
            ep_pairs += info["connected_pairs"]
            if truncated:
                break
        total_pairs += ep_pairs / MAX_STEPS
        total_switches += ep_switches
    return {"pairs": total_pairs / n_episodes,
            "switches": total_switches / n_episodes}


def main():
    model = PPO.load("models/ppo_relay_agent.zip")
    max_pairs = N_UAVS * (N_UAVS - 1) // 2

    strategies = {
        "PPO":       ppo_strategy(model),
        "Heuristic": heuristic_relay,
        "Rule-based": rulebased_relay,
    }
    envs = {
        "in-dist (eff3.0,v50)":  RelayEnvVariant(3.0, 50.0),
        "OOD-tight (eff1.5)":    RelayEnvVariant(1.5, 50.0),
        "OOD-wide (eff5.0)":     RelayEnvVariant(5.0, 50.0),
        "OOD-fast (v150)":       RelayEnvVariant(3.0, 150.0),
    }

    print(f"PPO 릴레이 에이전트 일반화 (연결쌍 평균 / 최대 {max_pairs})\n")
    print(f"  {'환경':<24}{'PPO':>10}{'Heuristic':>12}{'Rule-based':>12}{'PPO우위':>10}")
    print(f"  {'-'*68}")
    for ename, env in envs.items():
        row = {}
        for sname, fn in strategies.items():
            random.seed(0); np.random.seed(0)
            row[sname] = evaluate_env(fn, env, n_episodes=500)["pairs"]
        best_other = max(row["Heuristic"], row["Rule-based"])
        edge = (row["PPO"] - best_other) / max_pairs * 100
        print(f"  {ename:<24}{row['PPO']:>10.3f}{row['Heuristic']:>12.3f}"
              f"{row['Rule-based']:>12.3f}{edge:>+9.1f}%p")
    print(f"\n  (PPO우위 = PPO − max(Heuristic,Rule-based), 연결률 %p)")
    print(f"  음수면 PPO가 비-RL 기준선보다 못함")


if __name__ == "__main__":
    main()
