"""
rl_relay_agent.py — PPO 기반 릴레이 UAV 선택 에이전트
State : 링크 RSSI 특징 벡터 + 현재 릴레이 원-핫
Action: relay UAV 선택 (Discrete(N_UAVS))
Reward: 연결성 유지 보너스 - 릴레이 전환 패널티
비교  : PPO vs 휴리스틱(RSSI 합 최대) vs Rule-based(중앙 고정)
"""

import json
import math
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

# ── 상수 ─────────────────────────────────────────────────────────────────────
N_UAVS      = 5
COMM_RANGE  = 160.0        # m
RSSI_THRESH = -90.0        # dBm (DQN 학습 환경과 동일)
TX_POWER    = -5.0         # dBm — 실효 통신 거리 ~175m @2.4GHz
FREQ_MHZ    = 2400.0
MAX_STEPS   = 30           # 에피소드 최대 스텝
N_LINKS     = N_UAVS * (N_UAVS - 1) // 2   # 10
STATE_DIM   = N_LINKS + N_UAVS              # 15

AREA_SIZE   = 250.0        # 시뮬레이션 공간 (m) — 일부 쌍이 범위 밖
SPEED       = 5.0          # UAV 이동 속도 (m/step)

REWARD_CONN    =  1.0      # 연결 쌍당 보너스
REWARD_SWITCH  = -0.5      # 릴레이 전환 패널티

OUT_DIR = Path(__file__).parent / "models"
OUT_DIR.mkdir(exist_ok=True)
MODEL_PATH   = OUT_DIR / "ppo_relay_agent.zip"
RESULTS_PATH = OUT_DIR / "ppo_relay_results.json"


# ── RSSI / 연결 유틸 ──────────────────────────────────────────────────────────
def compute_rssi(ax, ay, bx, by) -> float:
    d = math.hypot(bx - ax, by - ay) or 0.01
    fspl = 20 * math.log10(d) + 20 * math.log10(FREQ_MHZ) - 27.55
    return TX_POWER - fspl


def is_connected(ax, ay, bx, by) -> bool:
    return (math.hypot(bx - ax, by - ay) <= COMM_RANGE and
            compute_rssi(ax, ay, bx, by) >= RSSI_THRESH)


def link_pairs():
    """(i, j) 링크 인덱스 순서 반환."""
    return [(i, j) for i in range(N_UAVS) for j in range(i + 1, N_UAVS)]


# ── 연결성 계산 ───────────────────────────────────────────────────────────────
def count_connected_pairs(pos: np.ndarray, relay: int) -> int:
    """직접 연결 + relay 경유 연결 쌍 수 반환."""
    connected = set()
    for i in range(N_UAVS):
        for j in range(i + 1, N_UAVS):
            if i == j:
                continue
            ax, ay = pos[i]
            bx, by = pos[j]
            if is_connected(ax, ay, bx, by):
                connected.add((i, j))
            elif (is_connected(ax, ay, pos[relay][0], pos[relay][1]) and
                  is_connected(pos[relay][0], pos[relay][1], bx, by)):
                connected.add((i, j))
    return len(connected)


# ── Gymnasium 환경 ────────────────────────────────────────────────────────────
class RelayEnv(gym.Env):
    """
    State : [rssi_link_0..9 (정규화), relay_onehot_0..4]  shape=(15,)
    Action: Discrete(5) — relay로 지정할 UAV 인덱스
    Reward: connected_pairs × REWARD_CONN + switch_penalty
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_UAVS)

        self.pos   = np.zeros((N_UAVS, 2))
        self.relay = 0
        self.step_count = 0

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────
    def _random_positions(self):
        """UAV 위치를 랜덤 배치 (일부 쌍이 직접 통신 불가하도록)."""
        while True:
            pos = np.random.uniform(0, AREA_SIZE, (N_UAVS, 2))
            # 적어도 한 쌍이 직접 연결 불가해야 의미 있는 시나리오
            has_disconnected = any(
                not is_connected(pos[i][0], pos[i][1], pos[j][0], pos[j][1])
                for i in range(N_UAVS) for j in range(i + 1, N_UAVS)
            )
            if has_disconnected:
                return pos

    def _observe(self) -> np.ndarray:
        rssi_vec = []
        for i, j in link_pairs():
            rssi = compute_rssi(
                self.pos[i][0], self.pos[i][1],
                self.pos[j][0], self.pos[j][1])
            # [-120, -20] → [-1, 1] 정규화
            rssi_vec.append(np.clip((rssi - RSSI_THRESH) / 50.0, -1.0, 1.0))
        relay_oh = np.zeros(N_UAVS, dtype=np.float32)
        relay_oh[self.relay] = 1.0
        return np.array(rssi_vec, dtype=np.float32)._concatenate(relay_oh) \
               if False else np.concatenate([
                   np.array(rssi_vec, dtype=np.float32), relay_oh])

    def _move_uavs(self):
        """UAV를 소량 랜덤 이동 (동적 환경 시뮬레이션)."""
        noise = np.random.uniform(-SPEED, SPEED, (N_UAVS, 2))
        self.pos = np.clip(self.pos + noise, 0, AREA_SIZE)

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos        = self._random_positions()
        self.relay      = random.randint(0, N_UAVS - 1)
        self.step_count = 0
        return self._observe(), {}

    def step(self, action: int):
        prev_relay = self.relay
        self.relay = int(action)

        pairs_before = count_connected_pairs(self.pos, prev_relay)
        self._move_uavs()
        pairs_after  = count_connected_pairs(self.pos, self.relay)

        reward  = pairs_after * REWARD_CONN
        reward += REWARD_SWITCH if self.relay != prev_relay else 0.0

        self.step_count += 1
        terminated = False
        truncated  = self.step_count >= MAX_STEPS
        return self._observe(), reward, terminated, truncated, {
            "connected_pairs": pairs_after,
            "relay_switched":  self.relay != prev_relay,
        }


# ── 전략 함수들 (비교용) ──────────────────────────────────────────────────────
def heuristic_relay(pos: np.ndarray, _prev: int) -> int:
    """RSSI 합 최대 UAV 선택 (현재 ml_server.py 방식)."""
    score = np.zeros(N_UAVS)
    for i in range(N_UAVS):
        for j in range(N_UAVS):
            if i == j:
                continue
            rssi = compute_rssi(pos[i][0], pos[i][1], pos[j][0], pos[j][1])
            if rssi >= RSSI_THRESH:
                score[i] += rssi
    return int(np.argmax(score))


def rulebased_relay(_pos: np.ndarray, _prev: int) -> int:
    """항상 중앙 인덱스 UAV 선택 (Rule-based)."""
    return N_UAVS // 2


# ── 평가 함수 ─────────────────────────────────────────────────────────────────
def evaluate(strategy_fn, n_episodes: int = 1000) -> dict:
    env = RelayEnv()
    total_pairs, total_switches, total_reward = 0.0, 0, 0.0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        relay  = random.randint(0, N_UAVS - 1)
        ep_pairs, ep_switches, ep_reward = 0.0, 0, 0.0
        for _ in range(MAX_STEPS):
            action = strategy_fn(env.pos, relay)
            if action != relay:
                ep_switches += 1
            relay = action
            obs, reward, _, truncated, info = env.step(action)
            ep_pairs  += info["connected_pairs"]
            ep_reward += reward
            if truncated:
                break
        total_pairs   += ep_pairs / MAX_STEPS
        total_switches += ep_switches
        total_reward  += ep_reward / MAX_STEPS

    return {
        "avg_connected_pairs": round(total_pairs / n_episodes, 3),
        "avg_switches_per_ep": round(total_switches / n_episodes, 2),
        "avg_reward":          round(total_reward / n_episodes, 3),
    }


def ppo_strategy(model):
    def _fn(pos, _prev):
        env_tmp = RelayEnv()
        env_tmp.pos   = pos.copy()
        env_tmp.relay = _prev
        obs = env_tmp._observe()
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return _fn


# ── 학습 ─────────────────────────────────────────────────────────────────────
def train(total_timesteps: int = 200_000):
    print("=" * 60)
    print("PPO 릴레이 에이전트 학습 시작")
    print(f"  State dim : {STATE_DIM}  |  Action : Discrete({N_UAVS})")
    print(f"  Timesteps : {total_timesteps:,}")
    print("=" * 60)

    vec_env = make_vec_env(RelayEnv, n_envs=4)
    model = PPO(
        "MlpPolicy", vec_env,
        n_steps=512, batch_size=64, n_epochs=10,
        learning_rate=3e-4, gamma=0.95,
        verbose=1,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(str(MODEL_PATH))
    print(f"\n모델 저장: {MODEL_PATH}")
    return model


# ── 비교 실험 ─────────────────────────────────────────────────────────────────
def compare(model, n_episodes: int = 1000):
    print("\n" + "=" * 60)
    print(f"비교 실험 ({n_episodes}회 에피소드)")
    print("=" * 60)

    strategies = {
        "PPO (RL)":    ppo_strategy(model),
        "Heuristic":   heuristic_relay,
        "Rule-based":  rulebased_relay,
    }

    results = {}
    for name, fn in strategies.items():
        print(f"  평가 중: {name}...", end=" ", flush=True)
        r = evaluate(fn, n_episodes)
        results[name] = r
        print(f"연결쌍={r['avg_connected_pairs']:.3f}  "
              f"전환={r['avg_switches_per_ep']:.2f}  "
              f"보상={r['avg_reward']:.3f}")

    max_pairs = N_UAVS * (N_UAVS - 1) // 2
    print("\n[ 결과 요약 ]")
    print(f"{'전략':<15} {'평균 연결쌍':>12} {'연결률':>8} {'릴레이전환':>10} {'평균보상':>10}")
    print("-" * 60)
    for name, r in results.items():
        conn_rate = r["avg_connected_pairs"] / max_pairs * 100
        print(f"{name:<15} {r['avg_connected_pairs']:>12.3f} "
              f"{conn_rate:>7.1f}% "
              f"{r['avg_switches_per_ep']:>10.2f} "
              f"{r['avg_reward']:>10.3f}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "n_episodes": n_episodes,
            "max_pairs":  max_pairs,
            "results":    results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {RESULTS_PATH}")
    return results


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # 환경 검증
    print("환경 검증 중...")
    check_env(RelayEnv(), warn=True)
    print("환경 검증 완료\n")

    if "--eval-only" in sys.argv and MODEL_PATH.exists():
        print("저장된 모델 로딩...")
        model = PPO.load(str(MODEL_PATH))
    else:
        model = train(total_timesteps=200_000)

    compare(model, n_episodes=1000)
