#!/usr/bin/env python3
"""
ppo_v2_samsung_test.py
──────────────────────
삼성역 실제 맵(건물 43개)에서 PPO v2 vs 규칙기반 vs 휴리스틱 비교.

PPO v2는 '자유공간 병목'에서 학습했다. 여기서는 삼성역 실제 건물(감쇠)을 LOS
장애물로 넣은 환경에서 평가 → PPO v2 입장에선 OOD(건물 감쇠가 RSSI를 교란).
"릴레이 회전 전략이 실제 건물 환경에서도 통하나"를 본다.

물리: PPO v2와 동일 (TX=-5dBm, f=2400MHz, 임계=-90dBm, 실효거리~177m)
      + 건물 LOS 통과 시 감쇠(높이기반 6~18dB) 추가.
구조: 병목 토폴로지(양끝 0·2 + 중간 브릿지 1·3·4) + 배터리 모델, 삼성역 좌표에 배치.

실행:
    /opt/anaconda3/envs/capstone/bin/python ppo_v2_samsung_test.py
"""
from __future__ import annotations
import math, random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl_relay_agent import (N_UAVS, COMM_RANGE, RSSI_THRESH, TX_POWER,
                            FREQ_MHZ, NOISE_FLOOR, link_pairs)
import rl_relay_agent_v2 as V2
from rl_relay_agent_v2 import (STATE_DIM, MAX_STEPS, BATT_DEAD,
                               DRAIN_RELAY, DRAIN_IDLE, W_CONN, W_SWITCH, W_BATT)
from rl_position_correction import _seg_hits_box
import dqn_samsung_real_test as SRT

MAP = 5000.0
SPEED = 20.0


# ── 건물 인식 물리 ────────────────────────────────────────────────────────────
def rssi_obs(ax, ay, bx, by, obs):
    d = math.hypot(bx - ax, by - ay) or 0.01
    fspl = 20 * math.log10(d) + 20 * math.log10(FREQ_MHZ) - 27.55
    atten = sum(o["atten"] for o in obs
                if _seg_hits_box(ax, ay, bx, by, o["x0"], o["x1"], o["y0"], o["y1"]))
    return TX_POWER - fspl - atten

def connected(ax, ay, bx, by, obs):
    return (math.hypot(bx - ax, by - ay) <= COMM_RANGE and
            rssi_obs(ax, ay, bx, by, obs) >= RSSI_THRESH)

def relay_pairs(pos, r, battery, obs):
    if battery[r] <= BATT_DEAD:
        return set()
    s = set(); rx, ry = pos[r]
    for i, j in link_pairs():
        if i == r or j == r: continue
        if connected(pos[i][0], pos[i][1], pos[j][0], pos[j][1], obs): continue
        if (connected(pos[i][0], pos[i][1], rx, ry, obs) and
                connected(rx, ry, pos[j][0], pos[j][1], obs)):
            s.add((i, j))
    return s

def eff_range(obs):
    lo, hi = 1.0, COMM_RANGE
    for _ in range(40):
        m = (lo + hi) / 2
        if connected(0, 0, m, 0, obs): lo = m
        else: hi = m
    return lo


# ── 삼성역 병목 환경 ──────────────────────────────────────────────────────────
class SamsungRelayEnv(gym.Env):
    def __init__(self, obs):
        super().__init__()
        self.obs = obs
        self.R = eff_range(obs)
        self.observation_space = spaces.Box(-1, 1, (STATE_DIM,), np.float32)
        self.action_space = spaces.Discrete(N_UAVS)
        self.pos = np.zeros((N_UAVS, 2)); self.relay = 0
        self.battery = np.ones(N_UAVS, np.float32); self.step_count = 0

    def _in_b(self, x, y):
        return any(o["x0"] <= x <= o["x1"] and o["y0"] <= y <= o["y1"] for o in self.obs)

    def _gen(self):
        R = self.R
        for _ in range(400):
            mx = np.random.uniform(R*2, MAP - R*2); my = np.random.uniform(R*2, MAP - R*2)
            if self._in_b(mx, my): continue
            ang = np.random.uniform(0, 2*math.pi); half = np.random.uniform(0.55*R, 0.72*R)
            ux, uy = math.cos(ang), math.sin(ang)
            pos = np.zeros((N_UAVS, 2))
            pos[0] = [mx - half*ux, my - half*uy]; pos[2] = [mx + half*ux, my + half*uy]
            for k in (1, 3, 4):
                pos[k] = [mx + np.random.uniform(-0.18*R, 0.18*R),
                          my + np.random.uniform(-0.18*R, 0.18*R)]
            pos = np.clip(pos, 0, MAP)
            if connected(pos[0][0], pos[0][1], pos[2][0], pos[2][1], self.obs): continue
            nb = sum(connected(pos[0][0], pos[0][1], pos[r][0], pos[r][1], self.obs) and
                     connected(pos[r][0], pos[r][1], pos[2][0], pos[2][1], self.obs)
                     for r in (1, 3, 4))
            if nb >= 2:
                return pos
        return np.clip(np.random.uniform(0, MAP, (N_UAVS, 2)), 0, MAP)

    def _hint(self):
        h = np.array([len(relay_pairs(self.pos, r, self.battery, self.obs))
                      for r in range(N_UAVS)], np.float32)
        m = h.max(); return h/m if m > 0 else h

    def _observe(self):
        rv, sv, pv = [], [], []
        for i, j in link_pairs():
            r = rssi_obs(self.pos[i][0], self.pos[i][1], self.pos[j][0], self.pos[j][1], self.obs)
            s = r - NOISE_FLOOR
            rv.append(np.clip((r + 90)/50, -1, 1)); sv.append(np.clip((s - 10)/50, -1, 1))
            pv.append(1.0/(1.0 + math.exp((s - 10.0)*0.5)))
        oh = np.zeros(N_UAVS, np.float32); oh[self.relay] = 1.0
        return np.concatenate([np.array(rv, np.float32), np.array(sv, np.float32),
                               np.array(pv, np.float32), oh, self.battery.copy(),
                               self._hint()]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self._gen(); self.relay = random.randint(0, N_UAVS-1)
        self.battery = np.ones(N_UAVS, np.float32); self.step_count = 0
        return self._observe(), {}

    def step(self, action):
        prev = self.relay; self.relay = int(action)
        marg = len(relay_pairs(self.pos, self.relay, self.battery, self.obs))
        sw = self.relay != prev
        for u in range(N_UAVS):
            self.battery[u] -= DRAIN_RELAY if u == self.relay else DRAIN_IDLE
        self.battery = np.clip(self.battery, 0, 1)
        reward = W_CONN*marg - (W_SWITCH if sw else 0) - W_BATT*(1 - self.battery[self.relay])
        self.pos = np.clip(self.pos + np.random.uniform(-SPEED, SPEED, (N_UAVS, 2)), 0, MAP)
        self.step_count += 1
        return self._observe(), float(reward), False, self.step_count >= MAX_STEPS, {
            "marginal": marg, "switched": sw, "min_batt": float(self.battery.min())}


# 휴리스틱/규칙 (건물 인식)
def heuristic(pos, battery, obs, _prev):
    best, bi = -1, 0
    for r in range(N_UAVS):
        c = len(relay_pairs(pos, r, np.ones(N_UAVS), obs))
        if c > best: best, bi = c, r
    return bi

def rulebased(pos, _b, _o, _prev):
    cx, cy = pos[:, 0].mean(), pos[:, 1].mean()
    return int(np.argmin((pos[:, 0]-cx)**2 + (pos[:, 1]-cy)**2))


def oracle(pos, battery, obs):
    return max(len(relay_pairs(pos, r, battery, obs)) for r in range(N_UAVS))


def evaluate(kind, model, env, n_ep=400):
    sel_hit = sel_tot = kept = poss = 0
    tot_marg = tot_batt = tot_rew = 0.0
    for _ in range(n_ep):
        obs, _ = env.reset(); relay = random.randint(0, N_UAVS-1)
        lstm, ep0 = None, True; m = r = 0.0
        for _ in range(MAX_STEPS):
            orc = oracle(env.pos, env.battery, env.obs)
            if kind == "ppo":
                a, lstm = model.predict(obs, state=lstm, episode_start=np.array([ep0]),
                                        deterministic=True); a = int(a); ep0 = False
            elif kind == "heuristic":
                a = heuristic(env.pos, env.battery, env.obs, relay)
            else:
                a = rulebased(env.pos, env.battery, env.obs, relay)
            ach = len(relay_pairs(env.pos, a, env.battery, env.obs))
            if orc > 0:
                sel_tot += 1; sel_hit += int(ach >= orc); poss += orc; kept += ach
            relay = a
            obs, rw, _, tr, info = env.step(a); m += info["marginal"]; r += rw
            if tr: break
        tot_marg += m/MAX_STEPS; tot_batt += info["min_batt"]; tot_rew += r/MAX_STEPS
    return {"sel_acc": 100*sel_hit/sel_tot if sel_tot else 0,
            "retention": 100*kept/poss if poss else 0,
            "marginal": tot_marg/n_ep, "min_batt": tot_batt/n_ep, "reward": tot_rew/n_ep}


def main():
    from sb3_contrib import RecurrentPPO
    buildings = SRT.load_real_buildings()
    obs = SRT.to_obstacles(buildings, "realistic")
    model = RecurrentPPO.load("models/ppo_relay_agent_v2.zip")
    env = SamsungRelayEnv(obs)
    print(f"삼성역 건물 {len(buildings)}개 · 실효거리 {env.R:.0f}m\n")
    print(f"{'='*72}")
    print(f"  삼성역 실제 맵에서 PPO v2 vs 규칙기반 (OOD, 400ep)")
    print(f"{'='*72}")
    print(f"  {'전략':<18}{'선택정확도':>11}{'연결유지율':>11}{'연결쌍':>9}{'최저배터리':>11}{'종합보상':>10}")
    print(f"  {'-'*68}")
    for name, kind in [("PPO v2 (RL)", "ppo"), ("Heuristic", "heuristic"),
                       ("Rule-based", "rulebased")]:
        random.seed(0); np.random.seed(0)
        r = evaluate(kind, model, env)
        print(f"  {name:<18}{r['sel_acc']:>10.1f}%{r['retention']:>10.1f}%"
              f"{r['marginal']:>9.3f}{r['min_batt']:>11.3f}{r['reward']:>10.3f}")
    print(f"  {'-'*68}")


if __name__ == "__main__":
    main()
