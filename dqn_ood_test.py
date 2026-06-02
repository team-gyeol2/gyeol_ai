#!/usr/bin/env python3
"""
dqn_ood_test.py
───────────────
DQN 위치 보정 모델의 *환경(맵) 일반화* 측정 — Out-Of-Distribution 테스트.

패턴 홀드아웃(dqn_generalization_test.py)은 같은 삼성역 맵 안에서 새로운
'이동 패턴'에 대한 일반화를 봤다. 여기서는 한 발 더 나아가 삼성역과
'다른 도심 기하구조'(건물 배치·밀도·감쇠가 전혀 다른 맵)를 절차적으로
생성해, 거기서의 재연결 성공률을 측정한다.

  학습 : 삼성역 단절 시나리오 전체 (in-distribution)
  평가 : 절차 생성한 무작위 도심 환경 (OOD)
         - sparse  : 삼성역과 비슷한 희소 건물
         - dense   : 건물 많고 크고 감쇠 큰 도심 협곡
         - random  : 면적/밀도/감쇠 전부 무작위

물리 모델(통신거리·송신전력·RSSI식)은 동일 — '같은 무전기, 다른 도시' 설정.

실행:
    python3 dqn_ood_test.py
"""

from __future__ import annotations

import random

import numpy as np
import torch

import rl_position_correction as base
from rl_position_correction import (
    UavCorrectionEnv, compute_rssi, is_connected,
    COMM_RANGE, MAX_STEPS, STEP_SIZES,
)

# 메인 그룹 UAV 수 (격리 1대 + 메인 그룹)
MAIN_MIN, MAIN_MAX = 3, 4


# ── OOD 환경/시나리오 절차 생성 ───────────────────────────────────────────────
def _rand_obstacles(rng, area, n_obs, size_rng, atten_rng):
    obs = []
    for _ in range(n_obs):
        w = rng.uniform(*size_rng); h = rng.uniform(*size_rng)
        x0 = rng.uniform(0, area - w); y0 = rng.uniform(0, area - h)
        obs.append({
            "x0": x0, "x1": x0 + w, "y0": y0, "y1": y0 + h,
            "atten": rng.uniform(*atten_rng),
        })
    return obs


def _gen_scenario(rng, profile):
    """단절 1대 + 연결된 메인 그룹 1개 짜리 OOD 시나리오 생성.
    반환: dict(iso_pos, main_pos, rssi, obstacles) — 없으면 None(재시도)."""
    if profile == "sparse":
        area = rng.uniform(350, 500); n_obs = rng.randint(2, 5)
        size_rng, atten_rng = (8, 18), (4, 8)
    elif profile == "dense":
        area = rng.uniform(500, 800); n_obs = rng.randint(10, 20)
        size_rng, atten_rng = (20, 50), (8, 20)
    else:  # random
        area = rng.uniform(350, 800); n_obs = rng.randint(0, 18)
        size_rng, atten_rng = (8, 50), (3, 20)

    obstacles = _rand_obstacles(rng, area, n_obs, size_rng, atten_rng)

    # 메인 그룹: 한 중심 근처에 서로 연결되도록 배치
    n_main = rng.randint(MAIN_MIN, MAIN_MAX)
    cx, cy = rng.uniform(0, area), rng.uniform(0, area)
    main = []
    for _ in range(n_main):
        for _try in range(30):
            p = (cx + rng.uniform(-COMM_RANGE*0.4, COMM_RANGE*0.4),
                 cy + rng.uniform(-COMM_RANGE*0.4, COMM_RANGE*0.4))
            if not main or any(is_connected(p[0], p[1], q[0], q[1], obstacles) for q in main):
                main.append(p); break
    if len(main) < 2:
        return None

    # 격리 UAV: 메인 그룹 어느 누구와도 단절, 단 재연결 가능 거리(<=900m)
    for _try in range(60):
        ang = rng.uniform(0, 2*np.pi)
        dist = rng.uniform(COMM_RANGE*1.2, COMM_RANGE*5.0)  # ~190~800m
        ix, iy = cx + dist*np.cos(ang), cy + dist*np.sin(ang)
        disconnected = all(
            not is_connected(ix, iy, q[0], q[1], obstacles) for q in main
        )
        if disconnected:
            init_rssi = max(compute_rssi(ix, iy, q[0], q[1], obstacles) for q in main)
            return {"iso_pos": (ix, iy), "main_pos": main,
                    "rssi": init_rssi, "obstacles": obstacles}
    return None


def gen_ood(profile, n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        sc = _gen_scenario(rng, profile)
        if sc is not None:
            out.append(sc)
    return out


# ── OOD 평가 (시나리오마다 obstacles 교체) ────────────────────────────────────
def eval_ood(policy, scenarios, default_obs=None) -> dict:
    rl_suc, base_suc, n = 0, 0, len(scenarios)
    rl_steps = []
    for sc in scenarios:
        obs = sc.get("obstacles", default_obs)   # OOD는 자체 맵, in-dist는 삼성역 맵
        env = UavCorrectionEnv(obs, [sc])
        # RL
        state = env.reset_specific(sc)
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                action = policy(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
            state, reward, done = env.step(action)
            if done:
                if reward > 50:
                    rl_suc += 1; rl_steps.append(env.step_count)
                break
        # Baseline (고정 20m centroid)
        env.reset_specific(sc)
        for _ in range(MAX_STEPS):
            if env.baseline_step():
                base_suc += 1; break
    return {
        "n": n,
        "rl_success": rl_suc / n,
        "base_success": base_suc / n,
        "rl_avg_steps": round(float(np.mean(rl_steps)), 2) if rl_steps else None,
    }


def main():
    print("데이터 로딩 중 (삼성역, 학습용)...")
    obstacles, positions, links = base._load_data()
    train_sc = base.extract_scenarios(positions, links, obstacles)
    print(f"삼성역 단절 에피소드 {len(train_sc)}개로 DQN 학습\n")

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    train_env = UavCorrectionEnv(obstacles, train_sc)
    policy, _ = base.train(train_env)

    # in-distribution(삼성역) 재확인
    print("\n삼성역(in-distribution) 평가...")
    id_sc = random.choices(train_sc, k=500)
    id_res = eval_ood(policy, id_sc, default_obs=obstacles)

    # OOD 프로파일별 평가
    profiles = ["sparse", "dense", "random"]
    results = {"삼성역(in-dist)": id_res}
    for prof in profiles:
        scs = gen_ood(prof, n=500, seed=100 + profiles.index(prof))
        results[f"OOD-{prof}"] = eval_ood(policy, scs)

    # ── 출력 ──
    print(f"\n{'='*66}")
    print(f"  환경 일반화 (OOD) — 삼성역 학습 모델을 다른 도심 맵에서 평가")
    print(f"{'='*66}")
    print(f"  {'환경':<20}{'RL 성공률':>12}{'baseline':>12}{'RL avg steps':>14}")
    print(f"  {'-'*60}")
    for name, r in results.items():
        print(f"  {name:<20}{r['rl_success']:>11.1%}{r['base_success']:>12.1%}"
              f"{str(r['rl_avg_steps']):>14}")
    print(f"  {'-'*60}")
    id_rl = results['삼성역(in-dist)']['rl_success']
    print(f"\n  [해석] 삼성역 대비 OOD 성공률 하락폭:")
    for prof in profiles:
        drop = (results[f'OOD-{prof}']['rl_success'] - id_rl) * 100
        rl = results[f'OOD-{prof}']['rl_success']
        bl = results[f'OOD-{prof}']['base_success']
        print(f"    OOD-{prof:<8}: RL {rl:.1%}  ({drop:+.1f}%p)  |  baseline {bl:.1%}  "
              f"→ RL−base {(rl-bl)*100:+.1f}%p")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
