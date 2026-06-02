#!/usr/bin/env python3
"""
dqn_generalization_test.py
──────────────────────────
DQN 위치 보정 모델의 *진짜* 일반화 성능 측정 (패턴 홀드아웃).

기존 rl_position_correction.py 의 evaluate() 는 학습에 쓴 시나리오를 그대로
다시 평가(random.choices(env.scenarios))하므로 train=test 누수가 있고,
그래서 성공률이 100%로 나온다.

여기서는 30개 이동 시나리오를 scenario_id 단위로 train/test 로 분리한다.
  - test 시나리오의 단절 상황은 학습 중 한 번도 보지 못한다.
  - 같은 삼성역 맵·장애물이지만 '새로운 이동 패턴'에 대한 일반화를 측정.

여러 random seed 로 split 을 바꿔가며 반복해 평균/표준편차를 보고한다.

실행:
    python3 dqn_generalization_test.py
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch

import rl_position_correction as base
from rl_position_correction import (
    UavCorrectionEnv, compute_rssi, is_connected,
    MAX_STEPS, STEP_SIZES, EPISODES,
)


# ── scenario_id 를 유지하는 단절 시나리오 추출 ────────────────────────────────
def extract_scenarios_with_sid(positions, links, obstacles, min_streak=2):
    """base.extract_scenarios 와 동일하지만 각 에피소드에 scenario_id 를 태깅."""
    streak: dict[str, int] = defaultdict(int)
    by_sid: dict[str, list] = defaultdict(list)

    for key in sorted(positions, key=lambda k: (k[0], float(k[1]))):
        sid, _t = key
        pos = positions[key]
        lnk = links.get(key, {})

        any_disc = any(v == "disconnected" for v in lnk.values())
        streak[sid] = (streak[sid] + 1) if any_disc else 0
        if streak[sid] < min_streak:
            continue

        adj: dict[int, set] = defaultdict(set)
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
                    compute_rssi(ix, iy, p[0], p[1], obstacles) for p in main_pos
                )
                by_sid[sid].append({
                    "iso_pos":  (ix, iy),
                    "main_pos": main_pos,
                    "rssi":     init_rssi,
                })

    return by_sid


# ── 성공률 평가 (특정 시나리오 집합에 대해) ──────────────────────────────────
def eval_success(env: UavCorrectionEnv, policy, scenarios) -> dict:
    rl_suc, base_suc, n = 0, 0, len(scenarios)
    rl_steps = []
    for sc in scenarios:
        # RL
        state = env.reset_specific(sc)
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                action = policy(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
            state, reward, done = env.step(action)
            if done:
                if reward > 50:
                    rl_suc += 1
                    rl_steps.append(env.step_count)
                break
        # Baseline (고정 20m)
        env.reset_specific(sc)
        for _ in range(MAX_STEPS):
            if env.baseline_step():
                base_suc += 1
                break
    return {
        "n": n,
        "rl_success": rl_suc / n if n else 0.0,
        "base_success": base_suc / n if n else 0.0,
        "rl_avg_steps": round(float(np.mean(rl_steps)), 2) if rl_steps else None,
    }


def run_one_split(by_sid, obstacles, seed: int, test_frac: float = 0.3) -> dict:
    sids = sorted(by_sid.keys())
    rng = random.Random(seed)
    rng.shuffle(sids)
    n_test = max(1, int(len(sids) * test_frac))
    test_sids  = set(sids[:n_test])
    train_sids = set(sids[n_test:])

    train_sc = [sc for s in train_sids for sc in by_sid[s]]
    test_sc  = [sc for s in test_sids  for sc in by_sid[s]]

    # 재현성: seed 고정
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # train 시나리오로만 DQN 학습
    train_env = UavCorrectionEnv(obstacles, train_sc)
    policy, _ = base.train(train_env)

    # in-sample (train) vs out-of-sample (held-out test) 평가
    in_res  = eval_success(train_env, policy, train_sc)
    out_res = eval_success(train_env, policy, test_sc)

    return {
        "seed": seed,
        "n_train_sids": len(train_sids), "n_test_sids": len(test_sids),
        "test_sids": sorted(test_sids),
        "n_train_ep": len(train_sc), "n_test_ep": len(test_sc),
        "in_sample": in_res,
        "held_out":  out_res,
    }


def main():
    print("데이터 로딩 중...")
    obstacles, positions, links = base._load_data()
    by_sid = extract_scenarios_with_sid(positions, links, obstacles)

    total_ep = sum(len(v) for v in by_sid.values())
    print(f"시나리오 {len(by_sid)}개 · 단절 에피소드 {total_ep}개")
    print(f"DQN 학습 에피소드 수: {EPISODES} (split 당)\n")

    seeds = [0, 1, 2]
    runs = []
    for seed in seeds:
        print(f"\n{'='*60}\n  Split seed={seed}\n{'='*60}")
        r = run_one_split(by_sid, obstacles, seed)
        runs.append(r)
        print(f"  held-out 시나리오: {r['test_sids']}")
        print(f"  train ep={r['n_train_ep']}  test ep={r['n_test_ep']}")
        print(f"  [in-sample ] RL={r['in_sample']['rl_success']:.1%}  "
              f"base={r['in_sample']['base_success']:.1%}")
        print(f"  [held-out  ] RL={r['held_out']['rl_success']:.1%}  "
              f"base={r['held_out']['base_success']:.1%}  "
              f"(rl avg steps={r['held_out']['rl_avg_steps']})")

    # ── 요약 ──
    print(f"\n{'='*60}\n  요약 ({len(seeds)} splits 평균)\n{'='*60}")
    def agg(path_a, path_b):
        return np.array([run[path_a][path_b] for run in runs])
    in_rl   = agg("in_sample", "rl_success")
    out_rl  = agg("held_out",  "rl_success")
    out_base= agg("held_out",  "base_success")
    print(f"  {'':<28}{'평균':>10}{'표준편차':>10}")
    print(f"  {'-'*48}")
    print(f"  {'in-sample RL 성공률':<28}{in_rl.mean():>9.1%}{in_rl.std():>10.1%}")
    print(f"  {'held-out  RL 성공률':<28}{out_rl.mean():>9.1%}{out_rl.std():>10.1%}")
    print(f"  {'held-out  baseline 성공률':<28}{out_base.mean():>9.1%}{out_base.std():>10.1%}")
    print(f"  {'-'*48}")
    gap = (in_rl.mean() - out_rl.mean()) * 100
    print(f"  일반화 격차 (in − held-out): {gap:+.1f}%p")
    print(f"  held-out 에서 baseline 대비: {(out_rl.mean()-out_base.mean())*100:+.1f}%p")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
