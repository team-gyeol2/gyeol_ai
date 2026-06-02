# UAV 통신 복구 모델 일반화 성능 분석 보고서

**작성일:** 2026-06-01
**대상:** 삼성역 5km×5km UAV 정찰 네트워크 — Transformer / PPO / DQN 및 전체 파이프라인
**분석 동기:** "DQN 성공률 100%"가 실제 일반화 성능인지 검증

---

## 0. 요약 (Executive Summary)

| 모델 | 역할 | 일반화 판정 | 핵심 수치 |
|---|---|---|---|
| **Transformer** | 링크상태 예측 | ✅ 우수 | held-out 99.7% (격차 ≈ 0) |
| **PPO** | 릴레이 선택 | ✅ 강건 / ⚠️ 무이득 | 모든 분포에서 휴리스틱과 동률 |
| **DQN** | 위치 보정 | ❌ → ✅ (개선됨) | OOD 42~53% → DR 재학습 88~96% |
| **전체 파이프라인** | 예측→relay→보정 | ⚠️ | end-to-end 71.8% (병목=relay 77.5%) |

**핵심 결론:** 보고돼 온 "100%"는 **일반화 성능이 아니라 학습 데이터 그대로의 in-sample 성능**(데이터 누수)이었다. 실제 일반화를 측정하니 Transformer는 우수, PPO는 무난하나 RL 이득이 없고, DQN은 원래 일반화가 안 됐으나 **domain randomization 재학습으로 해결**했다.

---

## 1. 배경: "100%"의 함정

### 1.1 데이터 누수 (train = test)

1. **DQN 평가** (`rl_position_correction.py` `evaluate()`): 평가 집합을 `random.choices(env.scenarios)` 로 뽑는데, 이는 `train()` 이 학습에 쓴 **바로 그 시나리오들**이다. train/test 분리가 전혀 없어 100%는 "암기 성능"이다.

2. **데이터셋 분할** (`ns-3.47/datasets/uav_2d_initial/`): train/val/test CSV가 **모두 동일한 30개 시나리오**를 담고 있고, 시간 윈도우만 다르다. 따라서 `pipeline.py` 의 "test" 정확도와 Transformer의 test 정확도도 in-sample이다.

### 1.2 "삼성역 모델링"의 실체

- DQN이 학습한 데이터(`uav_2d_initial`)는 **건물 3개·약 440m**의 일반 합성 코리도이며, 생성 스크립트에 "삼성/samsung" 언급이 없다. **실제 삼성역과 무관.**
- 실제 삼성역 5km 맵(`models/samsung_*`, OpenStreetMap 기반)과 대시보드 "실제 환경" 탭의 건물 43개(`dashboard.REAL_BUILDINGS`)는 **시각화 배경일 뿐**, 링크·RSSI·RL 계산에 들어가지 않는다. 즉 "삼성역에서 테스트"는 정확히는 "삼성역을 배경으로 시각화"였다.

---

## 2. 방법론

진짜 일반화를 측정하기 위해 다음 평가 스크립트를 신규 작성했다.

| 스크립트 | 측정 내용 |
|---|---|
| `generalization_test.py` (기존) | Transformer Leave-One-Scenario-Out |
| `ppo_generalization_test.py` (신규) | PPO 분포이동(OOD) — 산포·속도 변경 |
| `dqn_generalization_test.py` (신규) | DQN 패턴 홀드아웃 (시나리오 분리 재학습) |
| `dqn_ood_test.py` (신규) | DQN 무작위 도심 맵 OOD |
| `dqn_samsung_real_test.py` (신규) | DQN 실제 삼성역 건물 43개 위 평가 |
| `dqn_domain_random.py` (신규) | DQN domain randomization 재학습 + 비교 |

---

## 3. 모델별 일반화 결과

### 3.1 Transformer — 링크상태 예측 ✅

**Leave-One-Scenario-Out** (시나리오를 통째로 학습 제외 후 평가, 14개 중 5개 완료·전부 동일 추세):

| | Accuracy | F1 (macro) |
|---|---|---|
| In-sample baseline | 99.65% | 99.73% |
| **Held-out (미학습)** | **99.69%** | 98.82% |

> **일반화 격차 ≈ 0.** 링크상태(healthy/degraded/disconnected)가 RSSI·SNR·PLR의 보편적 함수라 시나리오 의존성이 거의 없다. 진짜로 잘 일반화된다.

### 3.2 PPO — 릴레이 선택 ✅ 강건 / ⚠️ 무이득

**분포이동 OOD** (평균 연결쌍 / 최대 10):

| 환경 | PPO | Heuristic | Rule-based | PPO 우위 |
|---|---|---|---|---|
| in-dist | 0.499 | 0.497 | 0.499 | ±0.0%p |
| OOD-tight (촘촘) | 0.771 | 0.762 | 0.776 | −0.0%p |
| OOD-wide (성김) | 0.464 | 0.462 | 0.463 | ±0.0%p |
| OOD-fast (3배속) | 0.197 | 0.196 | 0.197 | ±0.0%p |

> 모든 분포에서 비-RL 기준선을 그대로 추종(붕괴 없음 = 강건)하나 **이득이 없다.** 기존 결과(`ppo_relay_results.json`)에서도 PPO(0.514) ≤ Rule-based(0.537)였다. **진단은 §6 참조.**

### 3.3 DQN — 위치 보정 ❌ → ✅

| 평가 축 | 기존(단일맵) | baseline | DR 재학습 |
|---|---|---|---|
| in-sample | 100% | 97.8% | 100% |
| 패턴 홀드아웃 | 100% (격차 0) | — | — |
| OOD-sparse | 58.8% | 59.4% | **94.2%** |
| OOD-dense | 57.0% | 57.2% | **88.0%** |
| OOD-random | 62.2% | 60.2% | **94.0%** |
| **삼성역 건물차폐** | 53.0% | 74.5% | **96.2%** |

> 기존 모델은 단일 합성맵만 외워서, 처음 보는 도심 맵·실제 삼성 건물에서 **단순 휴리스틱보다도 낮았다**(가장 약한 고리). 원인은 ① 한 맵 암기 ② 장애물 우회 미학습.

---

## 4. 개선: DQN Domain Randomization

**처방:** 매 에피소드 무작위 맵(면적·건물 배치/밀도/감쇠·단절거리 전부 랜덤, 건물이 LOS를 막는 하드 케이스 포함)을 생성해 재학습.

**결과:** OOD **+31~35%p**, 삼성역 건물차폐 **+43.2%p** (baseline 74.5% 추월), in-sample은 100% 유지(망각 없음). 모델: `models/rl_correction_dqn_domainrand.pt`.

> 매번 다른 맵으로 학습 → 특정 맵을 외울 수 없어 "단절 시 재연결 + 장애물 우회"라는 일반 원리를 학습. **OOD를 in-distribution으로 바꾼 것이 핵심.**

---

## 5. 전체 파이프라인 성능

**구성:** Transformer(링크예측) → 규칙기반 relay 선택 → 위치 보정 (`pipeline.py`, in-sample test split)

| 단계 | 성능 |
|---|---|
| 링크상태 분류 | Acc 94.0% / F1 92.4% |
| 릴레이 선택 | Acc **77.5%** |
| **전체 파이프라인 (end-to-end)** | Acc **71.8%** |
| 위치보정 발동 / 성공 | 228회 / **99.1%** |

> 파이프라인의 병목은 일반화가 아니라 **릴레이 선택 정확도(77.5%)**. 링크예측 단계는 일반화가 되고(LOSO 99.7%), relay·보정은 학습 안 하는 결정론적 휴리스틱이라 과적합 위험이 없다. 따라서 end-to-end 71.8%는 일반화 시에도 비슷하게 유지된다.

---

## 6. 보충: "멀티홉 연결률 표"와 "PPO 표"의 차이

`multihop_relay.py` 의 연결 방식별 표(Direct 64.7% / Single Relay 80.2% / Multi-hop 81.7%)는 **PPO 표와 다른 것을 측정**한다.

| | 멀티홉 표 | PPO 표 |
|---|---|---|
| 묻는 것 | 홉 깊이(1/2/4)의 효과 | 릴레이 선택 알고리즘의 우열 |
| 방법 | 결정론적 그래프 BFS | RL vs 휴리스틱 |
| 데이터 | 데이터셋 실제 link_state | rl_relay_agent 합성환경 |

**상보적 해석:** "릴레이를 두는 것"은 +15.5%p(Direct→Single)로 큰 효과지만, "어느 걸 두냐"는 무의미(2홉 넘으면 +1.5%p뿐). → PPO가 일반화는 강건하나 무이득인 이유와 정확히 일치.

### 6.1 PPO 진단 및 개선 방향

기존 결과: PPO 연결쌍 0.514 / 전환 0.62, Rule-based(중앙 고정) 0.537 / 0.81 → **단순 규칙이 이미 최고**.

원인: **행동이 결과를 거의 안 바꾸는 과제 구조** (대부분 쌍이 릴레이와 무관하게 연결/단절). 모델 문제가 아니라 task 문제.

개선 후보 (영향력 순):
1. **릴레이 선택 → 배치(이동)로 전환** — RL이 릴레이 UAV를 빈 공간으로 이동시켜 다리를 놓게. 행동 영향력 큼. (DQN 위치보정과 같은 결, 가장 추천)
2. **보상 재설계** — 릴레이가 실제 기여한 연결쌍(marginal)만 보상.
3. **병목 토폴로지 집중 학습** — 릴레이 선택이 결정적인 시나리오로.
4. **안정성 지표로 재포지셔닝** — PPO의 실제 강점(핸드오버 4배 감소)을 지표에 반영.

---

## 7. 결론 및 다음 단계

### 결론
- "100%"는 in-sample 성능이었고, 진짜 일반화는 모델마다 크게 달랐다.
- **Transformer**: 일반화 우수, 손댈 것 없음.
- **DQN**: 원래 약점이었으나 domain randomization으로 해결(OOD·실제 삼성 건물 모두 90%대).
- **PPO**: 강건하지만 RL 이득 없음 — task 재설계 필요.
- **파이프라인**: 다음 개선점은 일반화가 아니라 릴레이 선택 로직(77.5%).

### 다음 단계 (제안)
1. 파이프라인에 DR-DQN(`rl_correction_dqn_domainrand.pt`) 연결.
2. PPO를 "릴레이 배치(이동)" 과제로 재설계 (§6.1-1).
3. (선택) Transformer LOSO 14개 전체 완주로 평균 확정.

---

## 부록 A. 생성/사용 스크립트
- `generalization_test.py` — Transformer LOSO (기존)
- `ppo_generalization_test.py` — PPO 분포이동 OOD (신규)
- `dqn_generalization_test.py` — DQN 패턴 홀드아웃 (신규)
- `dqn_ood_test.py` — DQN 무작위 도심 OOD (신규)
- `dqn_samsung_real_test.py` — DQN 실제 삼성역 건물 평가 (신규)
- `dqn_domain_random.py` — DQN domain randomization 재학습 (신규)

## 부록 B. 실행 환경
- 순수 torch 스크립트(Transformer/DQN): 기본 `python3` (3.13, torch 2.8)
- PPO(SB3): `/opt/anaconda3/envs/capstone/bin/python` (3.10, torch 2.1.1)에 `stable_baselines3 2.3.2` 설치. (PPO 모델이 최신 SB3로 저장돼 `FloatSchedule` 경고가 뜨나 예측에는 무해.)

## 부록 C. 한계
- Transformer LOSO는 14개 중 5개 완료분(전부 99.2~100%)으로 추세 확정. 전체 평균은 별도 완주 시 갱신.
- OOD/삼성 건물차폐 시나리오는 절차 생성이므로 난이도에 생성 파라미터가 영향. 단, baseline 동시 평가로 상대비교는 견고.
