#!/usr/bin/env python3
"""
dqn_domain_random.py
────────────────────
DQN 위치 보정을 *domain randomization* 으로 재학습한다.

기존 모델은 단일 합성맵(440m·건물3개)에서만 학습 → OOD 42%, 삼성 건물차폐 53%
(둘 다 단순 baseline 보다 낮음). 원인은 '한 맵만 외움 + 장애물 우회 미학습'.

처방: 매 에피소드 무작위 맵(면적·건물 배치/밀도/감쇠·단절거리 전부 랜덤,
건물이 LOS를 막는 하드 케이스 포함)을 생성해 학습. 그러면 특정 맵을 외울 수
없고 '단절 시 어디로 가야 재연결되는가 + 장애물 우회'의 일반 원리를 배운다.

재학습 후 동일 잣대로 before/after 비교:
  - in-sample(원본 작은맵)  - OOD(sparse/dense/random)  - 삼성역 실제 건물차폐

실행:
    python3 dqn_domain_random.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

import rl_position_correction as base
from rl_position_correction import DQN, UavCorrectionEnv, EPISODES

import dqn_ood_test as ood
import dqn_samsung_real_test as srt

ROOT = Path(__file__).resolve().parent
OUT  = ROOT / "models" / "rl_correction_dqn_domainrand.pt"


# ── 도메인 랜덤화 학습 환경 ───────────────────────────────────────────────────
class DomainRandEnv(UavCorrectionEnv):
    """매 reset 마다 완전히 새로운 무작위 맵+단절 시나리오를 생성."""
    PROFILES = ["sparse", "dense", "random", "random"]

    def __init__(self):
        super().__init__(obstacles=[], scenarios=[])

    def reset(self):
        sc = None
        for _ in range(80):
            sc = ood._gen_scenario(random, random.choice(self.PROFILES))
            if sc is not None:
                break
        if sc is None:                      # 폴백: 빈 맵 단순 단절
            sc = {"iso_pos": (0.0, 0.0),
                  "main_pos": [(300.0, 0.0), (300.0, 50.0)],
                  "rssi": -120.0, "obstacles": []}
        self.obstacles = sc["obstacles"]
        self._load_scenario(sc)
        return self._state()


# ── 평가 헬퍼 ─────────────────────────────────────────────────────────────────
def eval_insample(policy):
    obstacles, positions, links = base._load_data()
    scs = base.extract_scenarios(positions, links, obstacles)
    random.seed(1)
    return ood.eval_ood(policy, random.choices(scs, k=500), default_obs=obstacles)


def eval_all(policy):
    res = {}
    res["in-sample"] = eval_insample(policy)
    for prof in ["sparse", "dense", "random"]:
        res[f"OOD-{prof}"] = ood.eval_ood(policy, ood.gen_ood(prof, 500, 100 + len(prof)))
    # 삼성역 실제 건물 차폐
    buildings = srt.load_real_buildings()
    obs = srt.to_obstacles(buildings, "realistic")
    hard = srt.gen_building_stress(buildings, obs, 400, 11)
    res["삼성-건물차폐"] = srt.evaluate(policy, obs, hard)
    return res


def _rl(r):  # 통일된 성공률 키
    return r.get("rl", r.get("rl_success"))


def main():
    # ── before: 기존 단일맵 모델 ──
    before = DQN()
    before.load_state_dict(torch.load(ROOT / "models" / "rl_correction_dqn.pt",
                                      map_location="cpu"))
    before.eval()

    # ── domain randomization 재학습 ──
    print(f"Domain randomization 재학습 시작 (EPISODES={EPISODES})\n")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    env = DomainRandEnv()
    after, _ = base.train(env)
    torch.save(after.state_dict(), OUT)
    after.eval()
    print(f"\n재학습 모델 저장: {OUT.name}\n")

    print("평가 중 (before/after)...")
    rb = eval_all(before)
    ra = eval_all(after)

    print(f"\n{'='*70}")
    print(f"  DQN 위치보정 성공률: 단일맵 학습  vs  Domain Randomization 재학습")
    print(f"{'='*70}")
    print(f"  {'환경':<18}{'기존(단일맵)':>14}{'DR재학습':>12}{'baseline':>12}{'개선':>9}")
    print(f"  {'-'*64}")
    for k in rb:
        b, a = _rl(rb[k]), _rl(ra[k])
        bl = ra[k].get("base", ra[k].get("base_success"))
        print(f"  {k:<18}{b:>13.1%}{a:>12.1%}{bl:>12.1%}{(a-b)*100:>+8.1f}%p")
    print(f"  {'-'*64}")
    print("  (baseline=고정 20m centroid, after-환경 기준)")


if __name__ == "__main__":
    main()
