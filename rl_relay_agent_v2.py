#!/usr/bin/env python3
"""
rl_relay_agent_v2.py — 개선된 PPO 릴레이 선택 에이전트
======================================================
기존 PPO(rl_relay_agent.py)는 단순 규칙(중앙 고정)을 못 이겼다. 원인은 과제
구조상 '릴레이를 누가 고르냐'가 결과를 거의 안 바꿔 RL이 짜낼 여지가 없던 것.

v2는 RL이 구조적으로 이길 수 있게 과제를 재설계한다 (이동 X, 선택만 유지):

  ③ 병목 토폴로지 학습:
     두 클러스터를 COMM_RANGE 보다 약간 멀게 분리하고, 중간 대역 UAV만 다리를
     놓을 수 있게 배치 → 릴레이 선택이 연결성을 좌우.

  ② 다목적 보상:
     marginal_conn (릴레이 *덕분에만* 연결된 쌍)  ── 신호 선명화
     − switch_penalty                              ── 핸드오버 비용
     − battery_penalty (릴레이는 빨리 닳음)         ── 한 노드 혹사 방지
     배터리 0.1 이하 릴레이는 중계 불가 → '회전'을 강제. 그리디 max-RSSI
     휴리스틱은 한 노드만 계속 골라 방전 → RL은 미리 회전시켜 균형 유지.

  ④ Recurrent(LSTM) PPO:
     시계열 토폴로지 변화를 추적, 곧 끊길/방전될 릴레이 대신 오래 버틸 릴레이를
     선제 선택. gamma 상향(미래 가중).

State(45): RSSI×10 + SNR×10 + PLR×10 + relay_onehot×5 + battery×5 + best_bridge_hint×5
Action: Discrete(5)

실행:
    python3 rl_relay_agent_v2.py            # 학습 + 비교
    python3 rl_relay_agent_v2.py --eval-only
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 물리 모델 재사용
from rl_relay_agent import (
    N_UAVS, COMM_RANGE, AREA_SIZE, N_LINKS, MAX_STEPS,
    compute_rssi, compute_snr, compute_plr, is_connected, link_pairs,
)

STATE_DIM = N_LINKS * 3 + N_UAVS * 3   # 30 + 15 = 45

# 다목적 보상 가중치
W_CONN    = 1.0     # marginal 연결쌍 보너스
W_SWITCH  = 0.3     # 릴레이 전환 패널티
W_BATT    = 0.6     # 저배터리 릴레이 선택 패널티
DRAIN_RELAY = 0.06  # 릴레이 UAV 스텝당 배터리 소모 (~16스텝이면 방전 → 회전 필요)
DRAIN_IDLE  = 0.004 # 비릴레이 UAV 소모
BATT_DEAD   = 0.1   # 이 이하면 중계 불가
SPEED       = 20.0  # 완만한 드리프트 (병목 구조 유지하되 시간변화 부여)


def _effective_range() -> float:
    """RSSI 임계로 결정되는 실효 연결거리(m). COMM_RANGE(800)보다 훨씬 짧음(~177)."""
    lo, hi = 1.0, COMM_RANGE
    for _ in range(40):
        mid = (lo + hi) / 2
        if is_connected(0.0, 0.0, mid, 0.0):
            lo = mid
        else:
            hi = mid
    return lo


R_EFF = _effective_range()

OUT_DIR    = Path(__file__).parent / "models"
MODEL_PATH = OUT_DIR / "ppo_relay_agent_v2.zip"
RESULTS    = OUT_DIR / "ppo_relay_v2_results.json"


# ── 연결성 유틸 ───────────────────────────────────────────────────────────────
def direct_pairs(pos) -> set:
    s = set()
    for i, j in link_pairs():
        if is_connected(pos[i][0], pos[i][1], pos[j][0], pos[j][1]):
            s.add((i, j))
    return s


def relay_pairs(pos, relay, battery) -> set:
    """relay 경유로 추가 연결되는 쌍 (배터리 살아있을 때만)."""
    if battery[relay] <= BATT_DEAD:
        return set()
    s = set()
    rx, ry = pos[relay]
    for i, j in link_pairs():
        if i == relay or j == relay:
            continue
        if is_connected(pos[i][0], pos[i][1], pos[j][0], pos[j][1]):
            continue
        if (is_connected(pos[i][0], pos[i][1], rx, ry) and
                is_connected(rx, ry, pos[j][0], pos[j][1])):
            s.add((i, j))
    return s


# ── 병목 토폴로지 생성 ────────────────────────────────────────────────────────
def _can_bridge(pos, r, a=0, b=2) -> bool:
    """UAV r 이 a↔b 를 중계로 연결할 수 있나 (둘 다 직접연결되진 않을 때)."""
    return (is_connected(pos[a][0], pos[a][1], pos[r][0], pos[r][1]) and
            is_connected(pos[r][0], pos[r][1], pos[b][0], pos[b][1]))


def bottleneck_positions() -> np.ndarray:
    """양끝 UAV0(좌)·UAV2(우)를 R_EFF 보다 멀게 두어 직접연결 불가.
    중간에 UAV1,3,4 를 모아 각각 0·2 를 모두 연결 → 셋 중 누구든 (0,2) 브릿지 가능.
    → 릴레이 선택이 연결을 좌우 + 배터리 회전 여지(브릿지 후보 ≥2)."""
    R = R_EFF
    for _ in range(300):
        mx = np.random.uniform(R * 2, AREA_SIZE - R * 2)
        my = np.random.uniform(R * 2, AREA_SIZE - R * 2)
        ang = np.random.uniform(0, 2 * math.pi)
        half = np.random.uniform(R * 0.55, R * 0.72)   # gap=2*half ∈ (1.1R,1.44R) > R
        ux, uy = math.cos(ang), math.sin(ang)

        pos = np.zeros((N_UAVS, 2))
        pos[0] = [mx - half * ux, my - half * uy]      # 좌 끝
        pos[2] = [mx + half * ux, my + half * uy]      # 우 끝
        # 중간 브릿지 후보 3대: midpoint 근처 작은 군집
        for k in (1, 3, 4):
            pos[k] = [mx + np.random.uniform(-R * 0.18, R * 0.18),
                      my + np.random.uniform(-R * 0.18, R * 0.18)]
        pos = np.clip(pos, 0, AREA_SIZE)

        # 조건: 0-2 직접 단절 + 중간 3대 중 ≥2 가 브릿지 가능
        if is_connected(pos[0][0], pos[0][1], pos[2][0], pos[2][1]):
            continue
        n_bridge = sum(_can_bridge(pos, r) for r in (1, 3, 4))
        if n_bridge >= 2:
            return pos
    return np.clip(np.random.uniform(0, AREA_SIZE, (N_UAVS, 2)), 0, AREA_SIZE)


# ── 환경 ──────────────────────────────────────────────────────────────────────
class RelayEnvV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, (STATE_DIM,), np.float32)
        self.action_space = spaces.Discrete(N_UAVS)
        self.pos = np.zeros((N_UAVS, 2))
        self.relay = 0
        self.battery = np.ones(N_UAVS, dtype=np.float32)
        self.step_count = 0

    def _best_bridge_hint(self) -> np.ndarray:
        """각 UAV를 릴레이로 썼을 때 추가 연결쌍 수 (정규화) — 상태 힌트."""
        hint = np.zeros(N_UAVS, dtype=np.float32)
        for r in range(N_UAVS):
            hint[r] = len(relay_pairs(self.pos, r, self.battery))
        m = hint.max()
        return hint / m if m > 0 else hint

    def _observe(self) -> np.ndarray:
        rssi_v, snr_v, plr_v = [], [], []
        for i, j in link_pairs():
            rssi = compute_rssi(self.pos[i][0], self.pos[i][1],
                                self.pos[j][0], self.pos[j][1])
            snr = compute_snr(rssi)
            rssi_v.append(np.clip((rssi + 90.0) / 50.0, -1, 1))
            snr_v.append(np.clip((snr - 10.0) / 50.0, -1, 1))
            plr_v.append(compute_plr(snr))
        relay_oh = np.zeros(N_UAVS, dtype=np.float32)
        relay_oh[self.relay] = 1.0
        return np.concatenate([
            np.array(rssi_v, np.float32), np.array(snr_v, np.float32),
            np.array(plr_v, np.float32), relay_oh,
            self.battery.copy(), self._best_bridge_hint(),
        ]).astype(np.float32)

    def _move(self):
        self.pos = np.clip(self.pos + np.random.uniform(-SPEED, SPEED, (N_UAVS, 2)),
                           0, AREA_SIZE)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = bottleneck_positions()
        self.relay = random.randint(0, N_UAVS - 1)
        self.battery = np.ones(N_UAVS, dtype=np.float32)
        self.step_count = 0
        return self._observe(), {}

    def step(self, action: int):
        prev = self.relay
        self.relay = int(action)

        marginal = len(relay_pairs(self.pos, self.relay, self.battery))
        switched = self.relay != prev

        # 배터리 소모
        for u in range(N_UAVS):
            self.battery[u] -= DRAIN_RELAY if u == self.relay else DRAIN_IDLE
        self.battery = np.clip(self.battery, 0.0, 1.0)

        reward = (W_CONN * marginal
                  - (W_SWITCH if switched else 0.0)
                  - W_BATT * (1.0 - self.battery[self.relay]))

        self._move()
        self.step_count += 1
        truncated = self.step_count >= MAX_STEPS
        return self._observe(), float(reward), False, truncated, {
            "marginal": marginal, "switched": switched,
            "min_batt": float(self.battery.min()),
        }


# ── 비교 전략 (배터리 무시 = 휴리스틱의 한계) ────────────────────────────────
def heuristic_v2(pos, battery, _prev):
    """현재 추가 연결쌍 최대 UAV (그리디, 배터리 무시)."""
    best, bi = -1, 0
    for r in range(N_UAVS):
        c = len(relay_pairs(pos, r, np.ones(N_UAVS)))  # 배터리 안 봄
        if c > best:
            best, bi = c, r
    return bi


def rulebased_v2(pos, _battery, _prev):
    """스웜 중심(centroid)에 가장 가까운 UAV 선택.
    기하학적 고정 규칙 — 링크품질·배터리 미사용, 인덱스 의존 없음(공정한 baseline)."""
    cx, cy = pos[:, 0].mean(), pos[:, 1].mean()
    d = (pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2
    return int(np.argmin(d))


# ── 평가 ──────────────────────────────────────────────────────────────────────
def eval_strategy(kind, model, env, n_ep=500):
    tot_marg = tot_sw = tot_minbatt = tot_reward = 0.0
    for _ in range(n_ep):
        obs, _ = env.reset()
        relay = random.randint(0, N_UAVS - 1)
        lstm_state = None
        ep_start = True
        ep_marg = ep_sw = ep_r = 0.0
        for _ in range(MAX_STEPS):
            if kind == "ppo":
                action, lstm_state = model.predict(
                    obs, state=lstm_state, episode_start=np.array([ep_start]),
                    deterministic=True)
                action = int(action)
                ep_start = False
            elif kind == "heuristic":
                action = heuristic_v2(env.pos, env.battery, relay)
            else:
                action = rulebased_v2(env.pos, env.battery, relay)
            relay = action
            obs, reward, _, trunc, info = env.step(action)
            ep_marg += info["marginal"]
            ep_sw += int(info["switched"])
            ep_r += reward
            if trunc:
                break
        tot_marg += ep_marg / MAX_STEPS
        tot_sw += ep_sw
        tot_minbatt += info["min_batt"]
        tot_reward += ep_r / MAX_STEPS
    return {
        "marginal_pairs": round(tot_marg / n_ep, 3),
        "switches": round(tot_sw / n_ep, 2),
        "min_battery": round(tot_minbatt / n_ep, 3),
        "reward": round(tot_reward / n_ep, 3),
    }


def train(timesteps=300_000):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.env_util import make_vec_env

    print(f"RecurrentPPO 학습 시작 (timesteps={timesteps:,}, state={STATE_DIM})")
    vec = make_vec_env(RelayEnvV2, n_envs=4)
    model = RecurrentPPO(
        "MlpLstmPolicy", vec,
        n_steps=256, batch_size=128, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        ent_coef=0.01, verbose=1,
    )
    model.learn(total_timesteps=timesteps)
    model.save(str(MODEL_PATH))
    print(f"모델 저장: {MODEL_PATH}")
    return model


def compare(model, n_ep=500):
    env = RelayEnvV2()
    print(f"\n{'='*72}\n  PPO v2 비교 ({n_ep} 에피소드, 병목 토폴로지)\n{'='*72}")
    res = {}
    for name, kind in [("RecurrentPPO v2", "ppo"),
                       ("Heuristic(greedy)", "heuristic"),
                       ("Rule-based(center)", "rulebased")]:
        random.seed(0); np.random.seed(0)
        res[name] = eval_strategy(kind, model, env, n_ep)

    print(f"  {'전략':<22}{'연결쌍':>9}{'전환':>8}{'최저배터리':>12}{'종합보상':>10}")
    print(f"  {'-'*60}")
    for name, r in res.items():
        print(f"  {name:<22}{r['marginal_pairs']:>9.3f}{r['switches']:>8.2f}"
              f"{r['min_battery']:>12.3f}{r['reward']:>10.3f}")
    print(f"  {'-'*60}")
    ppo = res["RecurrentPPO v2"]
    best_other = max(res["Heuristic(greedy)"]["reward"],
                     res["Rule-based(center)"]["reward"])
    print(f"\n  종합보상 PPO우위: {(ppo['reward']-best_other):+.3f} "
          f"(양수면 PPO가 비-RL 기준선 추월)")
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장: {RESULTS}")


if __name__ == "__main__":
    import sys
    if "--eval-only" in sys.argv and MODEL_PATH.exists():
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(str(MODEL_PATH))
    else:
        model = train()
    compare(model)
