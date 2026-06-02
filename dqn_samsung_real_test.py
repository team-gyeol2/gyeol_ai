#!/usr/bin/env python3
"""
dqn_samsung_real_test.py
────────────────────────
배포된 DQN 위치 보정 모델(models/rl_correction_dqn.pt)을
*실제 삼성역 5km×5km 맵의 진짜 건물 43개* 위에서 처음으로 직접 평가한다.

지금까지 'realistic environment' 탭은 작은 합성맵(440m·건물3개) 결과를 5km로
스케일해 그리기만 했고, 삼성/COEX/롯데타워 같은 건물은 RSSI·연결성·RL 계산에
전혀 들어가지 않았다(배경 그림). 이 스크립트는 그 건물들을 실제 LOS 차폐
장애물로 넣고, 모델 고유 물리(COMM_RANGE·스텝)를 유지한 채 재연결을 시킨다.

  맵      : 5000×5000 m, 삼성역=(2500,2500)
  건물    : dashboard.REAL_BUILDINGS 43개 (AST로 추출)
  장애물  : LOS가 건물을 통과하면 감쇠 적용
            - realistic : 높이 기반 감쇠 (clip(6+0.04*h, 6, 18) dB)
            - flat6dB   : 학습맵과 동일하게 전부 6 dB  (기하효과만 분리)
  물리    : rl_position_correction 의 상수 (COMM_RANGE=160, TX=-20, thr=-90)
  모델    : 학습된 그대로(small map). 재학습 없음.

시나리오: 5km 맵 곳곳에 무작위 단절 상황(메인 그룹 + 격리 1대)을 생성.
          건물이 LOS를 막는 경우(building-blocked)를 따로 집계.

실행:
    python3 dqn_samsung_real_test.py
"""

from __future__ import annotations

import ast
import math
import random
from pathlib import Path

import numpy as np
import torch

import rl_position_correction as base
from rl_position_correction import (
    DQN, UavCorrectionEnv, compute_rssi, is_connected,
    _seg_hits_box, COMM_RANGE, MAX_STEPS,
)

ROOT  = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "rl_correction_dqn.pt"
MAP   = 5000.0


# ── dashboard.REAL_BUILDINGS 추출 + 장애물 변환 ───────────────────────────────
def load_real_buildings() -> list[dict]:
    src = (ROOT / "dashboard.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        tgt = getattr(node, "target", None)
        if isinstance(node, ast.AnnAssign) and getattr(tgt, "id", None) == "REAL_BUILDINGS":
            return ast.literal_eval(node.value)
    raise RuntimeError("REAL_BUILDINGS 를 dashboard.py 에서 찾지 못함")


def to_obstacles(buildings: list[dict], mode: str) -> list[dict]:
    obs = []
    for b in buildings:
        if mode == "flat6dB":
            atten = 6.0
        else:  # realistic
            atten = float(np.clip(6.0 + 0.04 * b["height"], 6.0, 18.0))
        obs.append({"x0": b["x0"], "x1": b["x1"],
                    "y0": b["y0"], "y1": b["y1"], "atten": atten})
    return obs


def _in_building(x, y, obs) -> bool:
    return any(o["x0"] <= x <= o["x1"] and o["y0"] <= y <= o["y1"] for o in obs)


def _los_blocked(ax, ay, bx, by, obs) -> bool:
    return any(_seg_hits_box(ax, ay, bx, by, o["x0"], o["x1"], o["y0"], o["y1"])
               for o in obs)


# ── 삼성역 5km 맵 위 단절 시나리오 생성 ───────────────────────────────────────
def gen_samsung_scenarios(obs, n, seed) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    guard = 0
    while len(out) < n and guard < n * 200:
        guard += 1
        # 메인 그룹 중심 (건물 밖)
        cx, cy = rng.uniform(300, MAP - 300), rng.uniform(300, MAP - 300)
        if _in_building(cx, cy, obs):
            continue
        n_main = rng.randint(3, 4)
        main = []
        for _ in range(n_main):
            for _t in range(20):
                p = (cx + rng.uniform(-25, 25), cy + rng.uniform(-25, 25))
                if _in_building(p[0], p[1], obs):
                    continue
                if not main or any(is_connected(p[0], p[1], q[0], q[1], obs) for q in main):
                    main.append(p); break
        if len(main) < 2:
            continue
        # 격리 UAV: 35~250m, 단절, 건물 밖
        placed = None
        for _t in range(40):
            ang = rng.uniform(0, 2 * math.pi)
            d   = rng.uniform(35, 250)
            ix, iy = cx + d * math.cos(ang), cy + d * math.sin(ang)
            if not (0 < ix < MAP and 0 < iy < MAP) or _in_building(ix, iy, obs):
                continue
            if all(not is_connected(ix, iy, q[0], q[1], obs) for q in main):
                placed = (ix, iy); break
        if placed is None:
            continue
        ix, iy = placed
        nearest = min(main, key=lambda q: (q[0]-ix)**2 + (q[1]-iy)**2)
        out.append({
            "iso_pos":  (ix, iy),
            "main_pos": main,
            "rssi":     max(compute_rssi(ix, iy, q[0], q[1], obs) for q in main),
            "blocked":  _los_blocked(ix, iy, nearest[0], nearest[1], obs),
        })
    return out


# ── 평가 ──────────────────────────────────────────────────────────────────────
def evaluate(policy, obs, scenarios) -> dict:
    env = UavCorrectionEnv(obs, scenarios)
    rl_suc = base_suc = 0
    rl_b_suc = base_b_suc = n_blocked = 0
    rl_steps = []
    for sc in scenarios:
        blk = sc["blocked"]
        n_blocked += blk
        # RL
        state = env.reset_specific(sc)
        ok = False
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                a = policy(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
            state, reward, done = env.step(a)
            if done:
                ok = reward > 50
                break
        rl_suc += ok; rl_b_suc += (ok and blk)
        if ok:
            rl_steps.append(env.step_count)
        # Baseline
        env.reset_specific(sc)
        okb = False
        for _ in range(MAX_STEPS):
            if env.baseline_step():
                okb = True; break
        base_suc += okb; base_b_suc += (okb and blk)

    n = len(scenarios)
    nb = n_blocked
    return {
        "n": n, "n_blocked": nb,
        "rl": rl_suc / n, "base": base_suc / n,
        "rl_blocked":   (rl_b_suc / nb) if nb else None,
        "base_blocked": (base_b_suc / nb) if nb else None,
        "rl_open":   (rl_suc - rl_b_suc) / (n - nb) if n - nb else None,
        "base_open": (base_suc - base_b_suc) / (n - nb) if n - nb else None,
        "rl_steps": round(float(np.mean(rl_steps)), 2) if rl_steps else None,
    }


def gen_building_stress(buildings, obs, n, seed) -> list[dict]:
    """실제 삼성 건물을 드론 사이에 강제로 끼워 차폐시키는 하드 시나리오.
    메인 그룹은 건물 한 면 앞, 격리 UAV는 반대 면 앞 → LOS가 건물 관통.
    재연결하려면 건물 모서리를 돌아가야 함."""
    rng = random.Random(seed)
    out: list[dict] = []
    guard = 0
    while len(out) < n and guard < n * 300:
        guard += 1
        b = rng.choice(buildings)
        w, h = b["x1"] - b["x0"], b["y1"] - b["y0"]
        gap_m, gap_i = rng.uniform(15, 45), rng.uniform(15, 45)
        # 짧은 축을 가로지르는 방향으로 차폐 (우회가 step 예산 내 가능하도록)
        if w <= h:   # x축으로 가로지름
            my = rng.uniform(b["y0"] + 10, b["y1"] - 10)
            iy = rng.uniform(b["y0"] + 10, b["y1"] - 10)
            mcx, mcy = b["x0"] - gap_m, my
            ix,  iy  = b["x1"] + gap_i, iy
        else:        # y축으로 가로지름
            mx = rng.uniform(b["x0"] + 10, b["x1"] - 10)
            ixx = rng.uniform(b["x0"] + 10, b["x1"] - 10)
            mcx, mcy = mx, b["y0"] - gap_m
            ix,  iy  = ixx, b["y1"] + gap_i
        if not (0 < mcx < MAP and 0 < mcy < MAP and 0 < ix < MAP and 0 < iy < MAP):
            continue
        if _in_building(mcx, mcy, obs) or _in_building(ix, iy, obs):
            continue
        # 메인 그룹 (서로 연결, 건물 밖)
        main = []
        for _ in range(rng.randint(3, 4)):
            for _t in range(20):
                p = (mcx + rng.uniform(-22, 22), mcy + rng.uniform(-22, 22))
                if _in_building(p[0], p[1], obs):
                    continue
                if not main or any(is_connected(p[0], p[1], q[0], q[1], obs) for q in main):
                    main.append(p); break
        if len(main) < 2:
            continue
        if not all(not is_connected(ix, iy, q[0], q[1], obs) for q in main):
            continue
        nearest = min(main, key=lambda q: (q[0]-ix)**2 + (q[1]-iy)**2)
        if not _los_blocked(ix, iy, nearest[0], nearest[1], obs):
            continue   # 실제로 건물이 막아야만 채택
        out.append({
            "iso_pos": (ix, iy), "main_pos": main,
            "rssi": max(compute_rssi(ix, iy, q[0], q[1], obs) for q in main),
            "blocked": True,
            "detour_axis": min(w, h),
        })
    return out


def main():
    print("배포 모델 로딩:", MODEL.name)
    policy = DQN()
    policy.load_state_dict(torch.load(MODEL, map_location="cpu"))
    policy.eval()

    buildings = load_real_buildings()
    print(f"실제 삼성역 건물 {len(buildings)}개 로딩 (높이 "
          f"{min(b['height'] for b in buildings)}~{max(b['height'] for b in buildings)}m)\n")

    print(f"물리: COMM_RANGE={COMM_RANGE}m  (RSSI 임계로 실효 재연결거리 ~31m)\n")

    for mode in ["realistic", "flat6dB"]:
        obs = to_obstacles(buildings, mode)
        scs = gen_samsung_scenarios(obs, n=600, seed=7)
        res = evaluate(policy, obs, scs)
        atten_desc = ("높이기반 6~18dB" if mode == "realistic" else "전부 6dB(학습맵 동일)")
        print(f"{'='*64}")
        print(f"  삼성역 실제 맵 평가 — 감쇠={atten_desc}")
        print(f"{'='*64}")
        print(f"  시나리오 {res['n']}개  (건물차폐 {res['n_blocked']}개 / "
              f"개활 {res['n']-res['n_blocked']}개)")
        print(f"  {'구분':<16}{'DQN RL':>10}{'baseline':>12}")
        print(f"  {'-'*38}")
        print(f"  {'전체':<16}{res['rl']:>9.1%}{res['base']:>12.1%}")
        if res["rl_open"] is not None:
            print(f"  {'개활(건물無)':<16}{res['rl_open']:>9.1%}{res['base_open']:>12.1%}")
        if res["rl_blocked"] is not None:
            print(f"  {'건물차폐':<16}{res['rl_blocked']:>9.1%}{res['base_blocked']:>12.1%}")
        print(f"  {'-'*38}")
        print(f"  RL 성공시 평균스텝: {res['rl_steps']}")
        print(f"  RL − baseline (전체): {(res['rl']-res['base'])*100:+.1f}%p\n")

    # ── 건물 우회 하드 테스트 (실제 삼성 건물을 드론 사이에 강제 차폐) ──
    obs = to_obstacles(buildings, "realistic")
    hard = gen_building_stress(buildings, obs, n=400, seed=11)
    res = evaluate(policy, obs, hard)
    print(f"{'='*64}")
    print(f"  [하드] 실제 삼성 건물이 드론 사이를 가로막는 상황 (감쇠=현실값)")
    print(f"{'='*64}")
    print(f"  시나리오 {res['n']}개 (전부 건물차폐, 우회 필요)")
    print(f"  {'DQN RL':>12}{'baseline':>12}")
    print(f"  {res['rl']:>11.1%}{res['base']:>12.1%}")
    print(f"  RL 성공시 평균스텝: {res['rl_steps']}")
    print(f"  → 건물 우회가 필요한 단절에서의 실제 재연결률\n")

    print("참고: 학습맵(440m·건물3개) in-sample 성공률은 100% 였음.")


if __name__ == "__main__":
    main()
