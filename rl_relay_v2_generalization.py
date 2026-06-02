#!/usr/bin/env python3
"""
rl_relay_v2_generalization.py
─────────────────────────────
PPO v2 (RecurrentPPO, models/ppo_relay_agent_v2.zip)의 분포이동(OOD) 일반화 측정.

v2는 매 에피소드 무작위 병목 토폴로지로 학습(사실상 domain randomization).
따라서 학습 때 본 적 없는 '파라미터 분포'에서 우위가 유지되는지 본다:

  in-dist   : 표준 병목 (간격 1.1~1.44R, 브릿지3, 속도20, 방전0.06)
  OOD-wide  : 간격 더 넓음 (1.44~1.8R) — 브릿지 어려움
  OOD-fast  : 이동속도 2배 (40) — 구조가 빨리 변함
  OOD-drain : 배터리 방전 빠름 (0.10) — 회전 더 중요
  OOD-2brdg : 브릿지 후보 2개 — 회전 여지 적음

각 조건에서 PPO v2 vs Heuristic vs Rule-based 종합보상 비교.

실행:
    /opt/anaconda3/envs/capstone/bin/python rl_relay_v2_generalization.py
"""

from __future__ import annotations

import math
import random
import numpy as np

import rl_relay_agent_v2 as V2
from rl_relay_agent_v2 import (
    RelayEnvV2, relay_pairs, is_connected, eval_strategy,
    R_EFF, N_UAVS, MAX_STEPS, AREA_SIZE,
)


class RelayEnvV2Variant(RelayEnvV2):
    """병목 파라미터를 바꿀 수 있는 변형 환경 (OOD 평가용)."""
    def __init__(self, speed=20.0, drain=0.06, half=(0.55, 0.72), n_bridges=3):
        super().__init__()
        self.speed = speed
        self.drain = drain
        self.half_rng = half
        self.n_bridges = n_bridges

    def _move(self):
        self.pos = np.clip(
            self.pos + np.random.uniform(-self.speed, self.speed, (N_UAVS, 2)),
            0, AREA_SIZE)

    def _gen(self) -> np.ndarray:
        R = R_EFF
        bridges = [1, 3, 4][:self.n_bridges]
        nonbridge = [1, 3, 4][self.n_bridges:]
        for _ in range(400):
            mx = np.random.uniform(R * 2, AREA_SIZE - R * 2)
            my = np.random.uniform(R * 2, AREA_SIZE - R * 2)
            ang = np.random.uniform(0, 2 * math.pi)
            half = np.random.uniform(self.half_rng[0] * R, self.half_rng[1] * R)
            ux, uy = math.cos(ang), math.sin(ang)
            px, py = -uy, ux   # 수직 방향(비-브릿지 배치용)

            pos = np.zeros((N_UAVS, 2))
            pos[0] = [mx - half * ux, my - half * uy]
            pos[2] = [mx + half * ux, my + half * uy]
            for k in bridges:
                pos[k] = [mx + np.random.uniform(-R*0.18, R*0.18),
                          my + np.random.uniform(-R*0.18, R*0.18)]
            for k in nonbridge:   # 다리 못 놓게 멀리(수직)
                pos[k] = [mx + px * R * 1.5, my + py * R * 1.5]
            pos = np.clip(pos, 0, AREA_SIZE)

            if is_connected(pos[0][0], pos[0][1], pos[2][0], pos[2][1]):
                continue
            n_ok = sum(
                is_connected(pos[0][0], pos[0][1], pos[r][0], pos[r][1]) and
                is_connected(pos[r][0], pos[r][1], pos[2][0], pos[2][1])
                for r in bridges)
            if n_ok >= min(2, self.n_bridges):
                return pos
        return np.clip(np.random.uniform(0, AREA_SIZE, (N_UAVS, 2)), 0, AREA_SIZE)

    def reset(self, *, seed=None, options=None):
        import gymnasium as gym
        gym.Env.reset(self, seed=seed)
        self.pos = self._gen()
        self.relay = random.randint(0, N_UAVS - 1)
        self.battery = np.ones(N_UAVS, dtype=np.float32)
        self.step_count = 0
        return self._observe(), {}

    def step(self, action: int):
        prev = self.relay
        self.relay = int(action)
        marginal = len(relay_pairs(self.pos, self.relay, self.battery))
        switched = self.relay != prev
        for u in range(N_UAVS):
            self.battery[u] -= self.drain if u == self.relay else V2.DRAIN_IDLE
        self.battery = np.clip(self.battery, 0.0, 1.0)
        reward = (V2.W_CONN * marginal
                  - (V2.W_SWITCH if switched else 0.0)
                  - V2.W_BATT * (1.0 - self.battery[self.relay]))
        self._move()
        self.step_count += 1
        truncated = self.step_count >= MAX_STEPS
        return self._observe(), float(reward), False, truncated, {
            "marginal": marginal, "switched": switched,
            "min_batt": float(self.battery.min())}


def main():
    from sb3_contrib import RecurrentPPO
    model = RecurrentPPO.load("models/ppo_relay_agent_v2.zip")

    envs = {
        "in-dist":        RelayEnvV2Variant(20, 0.06, (0.55, 0.72), 3),
        "OOD-wide(간격↑)":  RelayEnvV2Variant(20, 0.06, (0.80, 1.00), 3),
        "OOD-fast(속도2배)": RelayEnvV2Variant(40, 0.06, (0.55, 0.72), 3),
        "OOD-drain(방전↑)": RelayEnvV2Variant(20, 0.10, (0.55, 0.72), 3),
        "OOD-2bridge":     RelayEnvV2Variant(20, 0.06, (0.55, 0.72), 2),
    }

    print(f"{'='*78}")
    print(f"  PPO v2 일반화 (분포이동) — 종합보상 기준, n=300")
    print(f"{'='*78}")
    print(f"  {'환경':<18}{'PPO v2':>10}{'Heuristic':>11}{'Rule':>9}"
          f"{'PPO우위':>10}{'PPO연결쌍':>11}{'PPO배터리':>11}")
    print(f"  {'-'*74}")
    for ename, env in envs.items():
        rows = {}
        detail = {}
        for name, kind in [("ppo", "ppo"), ("heuristic", "heuristic"),
                           ("rule", "rulebased")]:
            random.seed(0); np.random.seed(0)
            r = eval_strategy(kind, model, env, n_ep=300)
            rows[name] = r["reward"]
            if name == "ppo":
                detail = r
        best_other = max(rows["heuristic"], rows["rule"])
        edge = rows["ppo"] - best_other
        print(f"  {ename:<18}{rows['ppo']:>10.3f}{rows['heuristic']:>11.3f}"
              f"{rows['rule']:>9.3f}{edge:>+10.3f}"
              f"{detail['marginal_pairs']:>11.3f}{detail['min_battery']:>11.3f}")
    print(f"  {'-'*74}")
    print("  PPO우위 = PPO − max(Heuristic,Rule).  양수면 OOD에서도 RL 우위 유지.")


if __name__ == "__main__":
    main()
