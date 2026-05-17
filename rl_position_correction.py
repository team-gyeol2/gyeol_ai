#!/usr/bin/env python3
"""
rl_position_correction.py
─────────────────────────
DQN 기반 UAV 위치 보정: 단절된 UAV의 최적 이동 방향·거리 학습

기존 방식: centroid 방향으로 항상 고정 20m 이동
RL 방식:   현재 상태를 보고 8방향 × 4거리 = 32개 행동 중 최적 선택

State (8차원):
  [dx_norm, dy_norm, dist_norm, rssi_margin_norm,
   nearest_uav_dist_norm, step_norm, building_blocked, init_rssi_norm]

Action (32개):
  direction ∈ {0°, 45°, ..., 315°} × step ∈ {10, 20, 30, 40}m

Reward:
  재연결 성공: +100 - step*2  (빠를수록 높음)
  RSSI 개선:   +delta_rssi * 0.1
  이동 비용:   -step_size * 0.01
  타임아웃:    -10

실행:
    python3 rl_position_correction.py
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"
OUT_DIR.mkdir(exist_ok=True)

# ── 하이퍼파라미터 ─────────────────────────────────────────────────────────────
COMM_RANGE    = 160.0   # m, 최대 통신 거리
RSSI_THRESH   = -90.0   # dBm, 연결 판단 임계값
TX_POWER      = -20.0   # dBm (free-space 기준)
FREQ_MHZ      = 2400.0

MAX_STEPS     = 25      # 에피소드 최대 스텝
GAMMA         = 0.95
LR            = 1e-3
BATCH_SIZE    = 64
REPLAY_SIZE   = 8000
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.995
EPISODES      = 4000
TARGET_UPDATE = 50      # target network 동기화 주기

DIRECTIONS = [i * (math.pi / 4) for i in range(8)]   # 8방향
STEP_SIZES = [10.0, 20.0, 30.0, 40.0]                # 4거리
N_ACTIONS  = len(DIRECTIONS) * len(STEP_SIZES)        # 32
STATE_DIM  = 8


# ── 데이터 로딩 ────────────────────────────────────────────────────────────────
def _load_data():
    obstacles: list[dict] = []
    with open(DATA_DIR / "obstacles.csv") as f:
        for r in csv.DictReader(f):
            obstacles.append({
                "x0": float(r["x_min_m"]), "x1": float(r["x_max_m"]),
                "y0": float(r["y_min_m"]), "y1": float(r["y_max_m"]),
                "atten": float(r["attenuation_db"]),
            })

    positions: dict[tuple, dict[int, tuple]] = {}
    with open(DATA_DIR / "uav_positions.csv") as f:
        for r in csv.DictReader(f):
            key = (r["scenario_id"], r["time_s"])
            positions.setdefault(key, {})[int(r["uav_id"])] = (
                float(r["x_m"]), float(r["y_m"]))

    links: dict[tuple, dict[tuple, str]] = {}
    with open(DATA_DIR / "link_metrics.csv") as f:
        for r in csv.DictReader(f):
            key = (r["scenario_id"], r["time_s"])
            links.setdefault(key, {})[(int(r["src_uav"]), int(r["dst_uav"]))] = r["link_state"]

    return obstacles, positions, links


# ── 링크 품질 계산 ─────────────────────────────────────────────────────────────
def _seg_hits_box(x0, y0, x1, y1, bx0, bx1, by0, by1) -> bool:
    dx, dy = x1 - x0, y1 - y0
    t_min, t_max = 0.0, 1.0
    for d, p, lo, hi in [(dx, x0, bx0, bx1), (dy, y0, by0, by1)]:
        if abs(d) < 1e-9:
            if p < lo or p > hi:
                return False
        else:
            t1, t2 = (lo - p) / d, (hi - p) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
    return True


def compute_rssi(ax, ay, bx, by, obstacles) -> float:
    d = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2) + 0.1
    fspl = 20 * math.log10(d) + 20 * math.log10(FREQ_MHZ) - 27.55
    atten = sum(
        b["atten"] for b in obstacles
        if _seg_hits_box(ax, ay, bx, by, b["x0"], b["x1"], b["y0"], b["y1"])
    )
    return TX_POWER - fspl - atten


def is_connected(ax, ay, bx, by, obstacles) -> bool:
    d = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
    return d <= COMM_RANGE and compute_rssi(ax, ay, bx, by, obstacles) >= RSSI_THRESH


# ── 단절 시나리오 추출 ────────────────────────────────────────────────────────
def extract_scenarios(positions, links, obstacles, min_streak=2) -> list[dict]:
    streak: dict[str, int] = defaultdict(int)
    result: list[dict] = []

    for key in sorted(positions, key=lambda k: (k[0], float(k[1]))):
        sid, t_s = key
        pos = positions[key]
        lnk = links.get(key, {})

        any_disc = any(v == "disconnected" for v in lnk.values())
        streak[sid] = (streak[sid] + 1) if any_disc else 0
        if streak[sid] < min_streak:
            continue

        adj: dict[int, set[int]] = defaultdict(set)
        for (a, b), state in lnk.items():
            if state != "disconnected":
                adj[a].add(b); adj[b].add(a)

        visited, comps = set(), []
        for uid in pos:
            if uid not in visited:
                comp, stack = set(), [uid]
                while stack:
                    n = stack.pop()
                    if n in visited:
                        continue
                    visited.add(n); comp.add(n)
                    stack.extend(adj[n] - visited)
                comps.append(comp)

        if len(comps) < 2:
            continue

        main_comp = max(comps, key=len)
        main_pos  = [pos[u] for u in main_comp if u in pos]

        for comp in comps:
            if comp == main_comp:
                continue
            for iso_uid in comp:
                ix, iy = pos[iso_uid]
                init_rssi = max(
                    compute_rssi(ix, iy, p[0], p[1], obstacles)
                    for p in main_pos
                )
                result.append({
                    "iso_pos":  (ix, iy),
                    "main_pos": main_pos,
                    "rssi":     init_rssi,
                })

    return result


# ── 환경 ──────────────────────────────────────────────────────────────────────
class UavCorrectionEnv:
    def __init__(self, obstacles: list[dict], scenarios: list[dict]):
        self.obstacles = obstacles
        self.scenarios = scenarios

    def _load_scenario(self, sc: dict):
        self.iso_x, self.iso_y = sc["iso_pos"]
        self.main_group        = list(sc["main_pos"])
        self.init_rssi         = sc["rssi"]
        self.step_count        = 0

    def reset(self) -> np.ndarray:
        self._load_scenario(random.choice(self.scenarios))
        return self._state()

    def reset_specific(self, sc: dict) -> np.ndarray:
        self._load_scenario(sc)
        return self._state()

    def _state(self) -> np.ndarray:
        cx = sum(p[0] for p in self.main_group) / len(self.main_group)
        cy = sum(p[1] for p in self.main_group) / len(self.main_group)

        dx   = (cx - self.iso_x) / COMM_RANGE
        dy   = (cy - self.iso_y) / COMM_RANGE
        dist = math.sqrt((cx - self.iso_x) ** 2 + (cy - self.iso_y) ** 2) / COMM_RANGE

        nearest = min(self.main_group,
                      key=lambda p: (p[0] - self.iso_x)**2 + (p[1] - self.iso_y)**2)
        nearest_d    = math.sqrt((nearest[0]-self.iso_x)**2 + (nearest[1]-self.iso_y)**2)
        nearest_rssi = compute_rssi(self.iso_x, self.iso_y,
                                    nearest[0], nearest[1], self.obstacles)

        blocked = 1.0 if any(
            _seg_hits_box(self.iso_x, self.iso_y, nearest[0], nearest[1],
                          b["x0"], b["x1"], b["y0"], b["y1"])
            for b in self.obstacles
        ) else 0.0

        return np.clip(np.array([
            dx, dy, dist,
            (nearest_rssi - RSSI_THRESH) / 30.0,
            nearest_d / COMM_RANGE,
            self.step_count / MAX_STEPS,
            blocked,
            (self.init_rssi - RSSI_THRESH) / 30.0,
        ], dtype=np.float32), -3.0, 3.0)

    def step(self, action: int):
        direction = DIRECTIONS[action // len(STEP_SIZES)]
        dist_m    = STEP_SIZES[action % len(STEP_SIZES)]

        self.iso_x += dist_m * math.cos(direction)
        self.iso_y += dist_m * math.sin(direction)
        self.step_count += 1

        reconnected = any(
            is_connected(self.iso_x, self.iso_y, p[0], p[1], self.obstacles)
            for p in self.main_group
        )

        if reconnected:
            reward = 100.0 - self.step_count * 2.0
            done   = True
        elif self.step_count >= MAX_STEPS:
            best_rssi = max(
                compute_rssi(self.iso_x, self.iso_y, p[0], p[1], self.obstacles)
                for p in self.main_group
            )
            reward = max(0.0, (best_rssi - self.init_rssi) * 0.5) - 10.0
            done   = True
        else:
            best_rssi = max(
                compute_rssi(self.iso_x, self.iso_y, p[0], p[1], self.obstacles)
                for p in self.main_group
            )
            reward = (best_rssi - self.init_rssi) * 0.1 - dist_m * 0.01
            done   = False

        return self._state(), reward, done

    def baseline_step(self) -> bool:
        """기존 방식: centroid 방향으로 고정 20m"""
        cx = sum(p[0] for p in self.main_group) / len(self.main_group)
        cy = sum(p[1] for p in self.main_group) / len(self.main_group)
        dx, dy = cx - self.iso_x, cy - self.iso_y
        d = math.sqrt(dx ** 2 + dy ** 2) + 1e-9
        self.iso_x += 20.0 * dx / d
        self.iso_y += 20.0 * dy / d
        self.step_count += 1
        return any(
            is_connected(self.iso_x, self.iso_y, p[0], p[1], self.obstacles)
            for p in self.main_group
        )


# ── DQN ───────────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(np.array(s2)),
                torch.FloatTensor(d))

    def __len__(self):
        return len(self.buf)


# ── 학습 ──────────────────────────────────────────────────────────────────────
def train(env: UavCorrectionEnv) -> tuple[DQN, list[dict]]:
    policy    = DQN()
    target    = DQN()
    target.load_state_dict(policy.state_dict())
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    replay    = ReplayBuffer(REPLAY_SIZE)
    epsilon   = EPSILON_START
    history:  list[dict] = []

    for ep in range(1, EPISODES + 1):
        state        = env.reset()
        total_reward = 0.0
        success      = False

        while True:
            if random.random() < epsilon:
                action = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    action = policy(torch.FloatTensor(state).unsqueeze(0)).argmax().item()

            next_state, reward, done = env.step(action)
            replay.push(state, action, reward, next_state, float(done))
            state        = next_state
            total_reward += reward

            if len(replay) >= BATCH_SIZE:
                s, a, r, s2, d = replay.sample(BATCH_SIZE)
                with torch.no_grad():
                    target_q = r + GAMMA * target(s2).max(1)[0] * (1 - d)
                current_q = policy(s).gather(1, a.unsqueeze(1)).squeeze()
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                success = total_reward > 50
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if ep % TARGET_UPDATE == 0:
            target.load_state_dict(policy.state_dict())

        history.append({
            "episode": ep,
            "reward":  round(total_reward, 2),
            "steps":   env.step_count,
            "success": success,
        })

        if ep % 500 == 0:
            recent = history[-500:]
            sr     = sum(1 for h in recent if h["success"]) / len(recent)
            avg_r  = sum(h["reward"] for h in recent) / len(recent)
            print(f"  ep {ep:4d}  ε={epsilon:.3f}  성공률={sr:.1%}  평균보상={avg_r:6.1f}")

    return policy, history


# ── 평가 ──────────────────────────────────────────────────────────────────────
def evaluate(env: UavCorrectionEnv, policy: DQN, n_eval: int = 500) -> dict:
    rl_suc,   rl_steps,   rl_dist   = 0, [], []
    base_suc, base_steps, base_dist = 0, [], []

    test_set = random.choices(env.scenarios, k=n_eval)

    for sc in test_set:
        # ── RL ──
        state  = env.reset_specific(sc)
        tot_d  = 0.0
        suc_rl = False
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                action = policy(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
            state, reward, done = env.step(action)
            tot_d += STEP_SIZES[action % len(STEP_SIZES)]
            if done:
                if reward > 50:
                    suc_rl = True
                break
        rl_suc += int(suc_rl)
        if suc_rl:
            rl_steps.append(env.step_count)
            rl_dist.append(tot_d)

        # ── Baseline (동일 시나리오) ──
        env.reset_specific(sc)
        suc_b  = False
        tot_db = 0.0
        for _ in range(MAX_STEPS):
            suc_b = env.baseline_step()
            tot_db += 20.0
            if suc_b:
                break
        base_suc += int(suc_b)
        if suc_b:
            base_steps.append(env.step_count)
            base_dist.append(tot_db)

    def _mean(lst):
        return round(float(np.mean(lst)), 2) if lst else None

    return {
        "rl": {
            "success_rate": round(rl_suc / n_eval, 4),
            "avg_steps":    _mean(rl_steps),
            "avg_dist_m":   _mean(rl_dist),
        },
        "baseline": {
            "success_rate": round(base_suc / n_eval, 4),
            "avg_steps":    _mean(base_steps),
            "avg_dist_m":   _mean(base_dist),
        },
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("데이터 로딩 중...")
    obstacles, positions, links = _load_data()

    print("단절 시나리오 추출 중...")
    scenarios = extract_scenarios(positions, links, obstacles)
    print(f"추출 완료: {len(scenarios)}개 에피소드")

    if not scenarios:
        print("단절 시나리오 없음 — 데이터를 확인하세요.")
        return

    env = UavCorrectionEnv(obstacles, scenarios)

    print(f"\nDQN 학습 시작 (에피소드: {EPISODES}, 행동: {N_ACTIONS}개)")
    print(f"  State: {STATE_DIM}차원  |  행동: {len(DIRECTIONS)}방향 × {len(STEP_SIZES)}거리\n")
    policy, history = train(env)

    print("\n평가 중 (500 에피소드)...")
    results = evaluate(env, policy, n_eval=500)

    r_rl   = results["rl"]
    r_base = results["baseline"]

    print("\n" + "=" * 58)
    print("  위치 보정: DQN RL vs 기존 고정 20m 방식")
    print("=" * 58)
    print(f"  {'항목':<22} {'기존 방식':>14} {'DQN RL':>12}")
    print("  " + "-" * 50)
    print(f"  {'재연결 성공률':<22} "
          f"{r_base['success_rate']:>13.1%} "
          f"{r_rl['success_rate']:>11.1%}")
    print(f"  {'평균 스텝 수':<22} "
          f"{r_base['avg_steps'] or 'N/A':>13} "
          f"{r_rl['avg_steps'] or 'N/A':>11}")
    print(f"  {'평균 이동 거리 (m)':<22} "
          f"{r_base['avg_dist_m'] or 'N/A':>13} "
          f"{r_rl['avg_dist_m'] or 'N/A':>11}")
    print("=" * 58)

    gain_sr = (r_rl["success_rate"] - r_base["success_rate"]) * 100
    print(f"\n  [결론] DQN RL 재연결 성공률 {gain_sr:+.1f}%p 향상")

    # 저장
    model_path  = OUT_DIR / "rl_correction_dqn.pt"
    result_path = OUT_DIR / "rl_correction_results.json"

    torch.save(policy.state_dict(), model_path)

    save_data = {
        "comparison": results,
        "config": {
            "episodes": EPISODES, "n_actions": N_ACTIONS,
            "directions": len(DIRECTIONS), "step_sizes": STEP_SIZES,
            "max_steps": MAX_STEPS, "comm_range": COMM_RANGE,
        },
        "history": [h for h in history if h["episode"] % 10 == 0],
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n  모델 저장: {model_path}")
    print(f"  결과 저장: {result_path}")


if __name__ == "__main__":
    main()
