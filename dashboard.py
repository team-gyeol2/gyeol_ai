#!/usr/bin/env python3
"""
dashboard.py
────────────
UAV 군집 통신 장애 예측 & 복구 대시보드

① 애니메이션 지도: Play/Pause, 속도 조절, 타임스텝 슬라이더
② 중계 전환 시각화: 현재 relay UAV 별표 표시, 전환 시 화살표 + 이벤트 로그
③ 위치 보정 시각화: 단절 감지 시 격리 UAV 원형 표시 + 보정 방향 점선 화살표
④ 링크 상태 패널: 현재 프레임 healthy/degraded/disconnected 수

실행:
    python3 dashboard.py
    브라우저: http://127.0.0.1:8050
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import dash
from dash import Input, Output, State, callback, dcc, html, no_update, dash_table
import plotly.graph_objects as go
from collections import deque

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
DATA_DIR3 = ROOT / "ns-3.47" / "datasets" / "uav_3d"

STATE_COLOR = {"healthy": "#27ae60", "degraded": "#e67e22", "disconnected": "#e74c3c"}
UAV_NORMAL  = "#2980b9"
UAV_RELAY   = "#e74c3c"
UAV_ISOLATED= "#8e44ad"
BLDG_COLOR  = "rgba(120,120,120,0.30)"

HYSTERESIS  = 2   # 연속 disconnected 스텝 수 >= 이 값 → 위치 보정 발동
STEP_ARROW  = 20  # 위치 보정 화살표 길이(m)


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def _load_obstacles() -> list[dict]:
    rows = []
    with open(DATA_DIR / "obstacles.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "id": r["building_id"],
                "x0": float(r["x_min_m"]), "x1": float(r["x_max_m"]),
                "y0": float(r["y_min_m"]), "y1": float(r["y_max_m"]),
                "atten": float(r["attenuation_db"]),
            })
    return rows


def _load_all() -> tuple[dict, dict]:
    pos: dict[str, dict[str, dict[int, tuple]]] = defaultdict(lambda: defaultdict(dict))
    with open(DATA_DIR / "uav_positions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pos[r["scenario_id"]][r["time_s"]][int(r["uav_id"])] = (
                float(r["x_m"]), float(r["y_m"]), r.get("role", ""))

    lnk: dict[str, dict[str, dict[tuple, dict]]] = defaultdict(lambda: defaultdict(dict))
    with open(DATA_DIR / "link_metrics.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pair = (int(r["src_uav"]), int(r["dst_uav"]))
            lnk[r["scenario_id"]][r["time_s"]][pair] = {
                "state": r["link_state"],
                "rssi":  float(r["rssi_dbm_est"]),
                "plr":   float(r["plr_pct_est"]),
                "relay": int(r["optimal_relay_uav"]),
            }
    return dict(pos), dict(lnk)


print("데이터 로딩 중...")
OBSTACLES = _load_obstacles()
POSITIONS, LINKS = _load_all()
SCENARIOS = sorted(POSITIONS.keys())
OUT_DIR = ROOT / "models"
print(f"시나리오 {len(SCENARIOS)}개 로딩 완료")

# Initial buildings for the configurator DataTable
BUILDINGS_INIT = [
    {"id": b["id"], "x0": b["x0"], "x1": b["x1"],
     "y0": b["y0"], "y1": b["y1"], "height": 30.0, "atten": b["atten"]}
    for b in OBSTACLES
]


# ── 3D 데이터 로딩 ────────────────────────────────────────────────────────────

def _load_3d_data() -> tuple[dict, dict, list, list]:
    if not (DATA_DIR3 / "uav_positions_3d.csv").exists():
        return {}, {}, [], []
    obs3d: list[dict] = []
    with open(DATA_DIR3 / "obstacles_3d.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            obs3d.append({
                "id":    r["building_id"],
                "x0":    float(r["x_min_m"]), "x1": float(r["x_max_m"]),
                "y0":    float(r["y_min_m"]), "y1": float(r["y_max_m"]),
                "height":float(r["height_m"]),
                "atten": float(r["attenuation_db"]),
            })
    pos3d: dict[str, dict[str, dict[int, tuple]]] = defaultdict(lambda: defaultdict(dict))
    with open(DATA_DIR3 / "uav_positions_3d.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pos3d[r["scenario_id"]][r["time_s"]][int(r["uav_id"])] = (
                float(r["x_m"]), float(r["y_m"]),
                float(r["z_m"]), r.get("role", ""))
    lnk3d: dict[str, dict[str, dict[tuple, dict]]] = defaultdict(lambda: defaultdict(dict))
    with open(DATA_DIR3 / "link_metrics_3d.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pair = (int(r["src_uav"]), int(r["dst_uav"]))
            lnk3d[r["scenario_id"]][r["time_s"]][pair] = {
                "state": r["link_state"],
                "rssi":  float(r["rssi_dbm_est"]),
                "plr":   float(r["plr_pct_est"]),
                "relay": int(r["optimal_relay_uav"]),
            }
    scenarios3d = sorted(pos3d.keys())
    return dict(pos3d), dict(lnk3d), scenarios3d, obs3d


print("3D 데이터 로딩 중...")
POSITIONS_3D, LINKS_3D, SCENARIOS_3D, OBSTACLES_3D = _load_3d_data()
HAS_3D = bool(SCENARIOS_3D)
print(f"3D 시나리오 {len(SCENARIOS_3D)}개 로딩 완료" if HAS_3D else "3D 데이터 없음")


# ── 실제 환경 (삼성역 5km×5km) 데이터 ──────────────────────────────────────────
# 좌표계: (0,0)=남서모퉁이, (5000,5000)=북동모퉁이, 삼성역=(2500,2500), 단위 m
UAV_ALTITUDE_M = 80.0   # 정찰 비행 고도 (m) — 이 고도에서 대부분 건물 위로 비행

# waypoint 시나리오만 사용 (5km×5km 정찰 시나리오)
REAL_SCENARIOS = sorted(s for s in SCENARIOS if 'waypoint' in s.lower())

# waypoint 좌표 범위 기반 동적 스케일링 → 5km×5km 맵 꽉 채움
_wp_xs = [v[0] for sid in REAL_SCENARIOS
          for d in POSITIONS.get(sid, {}).values() for v in d.values()]
_wp_ys = [v[1] for sid in REAL_SCENARIOS
          for d in POSITIONS.get(sid, {}).values() for v in d.values()]
_WP_XMIN, _WP_XMAX = min(_wp_xs), max(_wp_xs)
_WP_YMIN, _WP_YMAX = min(_wp_ys), max(_wp_ys)
_MAP_MARGIN = 180  # m (가장자리 여백)
REAL_X_SCALE  = (5000 - 2 * _MAP_MARGIN) / max(_WP_XMAX - _WP_XMIN, 1)
REAL_X_OFFSET = _MAP_MARGIN - _WP_XMIN * REAL_X_SCALE
REAL_Y_SCALE  = (5000 - 2 * _MAP_MARGIN) / max(_WP_YMAX - _WP_YMIN, 1)
REAL_Y_OFFSET = _MAP_MARGIN - _WP_YMIN * REAL_Y_SCALE
print(f"실제 환경 스케일: X×{REAL_X_SCALE:.1f}  Y×{REAL_Y_SCALE:.1f}  "
      f"({len(REAL_SCENARIOS)}개 waypoint 시나리오)")


def _to_real(x_uav: float, y_uav: float) -> tuple[float, float]:
    return x_uav * REAL_X_SCALE + REAL_X_OFFSET, y_uav * REAL_Y_SCALE + REAL_Y_OFFSET


REAL_BUILDINGS: list[dict] = [
    # ─── COEX 복합단지 ────────────────────────────────────────────────────────
    {"id": "COEX_CONV",    "x0": 1750, "x1": 2350, "y0": 2650, "y1": 3100,
     "label": "COEX 컨벤션센터",           "color": "#e67e22", "height": 55},
    {"id": "COEX_MALL",    "x0": 1750, "x1": 2350, "y0": 3100, "y1": 3350,
     "label": "COEX 몰",                   "color": "#f39c12", "height": 30},
    {"id": "HYD_COEX",     "x0": 2350, "x1": 2600, "y0": 2800, "y1": 3300,
     "label": "현대백화점 무역센터",        "color": "#8e44ad", "height": 160},
    {"id": "COEX_IHG",     "x0": 1600, "x1": 1950, "y0": 3200, "y1": 3600,
     "label": "인터컨티넨탈 호텔",          "color": "#3498db", "height": 100},
    {"id": "COEX_GRAND",   "x0": 1950, "x1": 2250, "y0": 3300, "y1": 3620,
     "label": "그랜드 인터컨티넨탈",        "color": "#2980b9", "height": 140},
    # ─── 파르나스/무역센터 타워 ──────────────────────────────────────────────
    {"id": "PARNAS_TOWER", "x0": 2600, "x1": 2820, "y0": 2850, "y1": 3200,
     "label": "파르나스 타워",              "color": "#c0392b", "height": 245},
    {"id": "TRADE_TOWER",  "x0": 2350, "x1": 2600, "y0": 2550, "y1": 2800,
     "label": "무역센터 타워",              "color": "#e74c3c", "height": 228},
    # ─── GBC / 현대차 방면 ──────────────────────────────────────────────────
    {"id": "GBC_SITE",     "x0": 1900, "x1": 2350, "y0": 2050, "y1": 2500,
     "label": "현대 GBC (공사중)",          "color": "#c0392b", "height": 569},
    {"id": "HYUNDAI_MS",   "x0": 2350, "x1": 2600, "y0": 2100, "y1": 2450,
     "label": "현대 모터스튜디오",          "color": "#8e44ad", "height": 120},
    # ─── 삼성전자 / 삼성물산 방면 ────────────────────────────────────────────
    {"id": "SAMSUNG_CT",   "x0": 2820, "x1": 3100, "y0": 2700, "y1": 3100,
     "label": "삼성물산 본관",              "color": "#c0392b", "height": 130},
    {"id": "SAMSUNG_ELE",  "x0": 3100, "x1": 3400, "y0": 2600, "y1": 3000,
     "label": "삼성전자 서초사옥",          "color": "#e74c3c", "height": 100},
    {"id": "SAMSUNG_SEC",  "x0": 3400, "x1": 3650, "y0": 2600, "y1": 2950,
     "label": "삼성생명 빌딩",              "color": "#e67e22", "height": 80},
    {"id": "POSCO_CTR",    "x0": 3650, "x1": 3900, "y0": 2550, "y1": 2900,
     "label": "포스코센터",                 "color": "#c0392b", "height": 134},
    # ─── 봉은사 ────────────────────────────────────────────────────────────
    {"id": "BONGEUNSA",    "x0": 1500, "x1": 1900, "y0": 3050, "y1": 3600,
     "label": "봉은사",                     "color": "#16a085", "height": 15},
    # ─── 테헤란로 북쪽 ─────────────────────────────────────────────────────
    {"id": "GS_TOWER",     "x0": 2350, "x1": 2580, "y0": 3300, "y1": 3600,
     "label": "GS타워",                     "color": "#c0392b", "height": 185},
    {"id": "AVENUE_EL",    "x0": 2820, "x1": 3050, "y0": 3100, "y1": 3400,
     "label": "에비뉴엘 월드타워",          "color": "#e67e22", "height": 70},
    # ─── 테헤란로 남쪽 오피스 벨트 ──────────────────────────────────────────
    {"id": "TH_OFF_1",     "x0": 2600, "x1": 2820, "y0": 2100, "y1": 2500,
     "label": "포스코P&S타워",              "color": "#e67e22", "height": 90},
    {"id": "TH_OFF_2",     "x0": 3100, "x1": 3300, "y0": 2200, "y1": 2550,
     "label": "역삼 오피스A",               "color": "#e67e22", "height": 60},
    {"id": "TH_OFF_3",     "x0": 3300, "x1": 3500, "y0": 2200, "y1": 2550,
     "label": "역삼 오피스B",               "color": "#e67e22", "height": 55},
    {"id": "TH_OFF_4",     "x0": 3500, "x1": 3700, "y0": 2200, "y1": 2550,
     "label": "역삼 오피스C",               "color": "#e67e22", "height": 65},
    {"id": "TH_OFF_5",     "x0": 3700, "x1": 3950, "y0": 2200, "y1": 2550,
     "label": "선릉 오피스A",               "color": "#e67e22", "height": 60},
    {"id": "TH_OFF_6",     "x0": 3950, "x1": 4200, "y0": 2200, "y1": 2550,
     "label": "선릉 오피스B",               "color": "#e67e22", "height": 58},
    {"id": "TH_OFF_W1",    "x0": 1400, "x1": 1650, "y0": 2350, "y1": 2650,
     "label": "역삼 서쪽 오피스",           "color": "#27ae60", "height": 45},
    {"id": "TH_OFF_W2",    "x0": 1100, "x1": 1400, "y0": 2350, "y1": 2650,
     "label": "강남 대로변 빌딩",           "color": "#27ae60", "height": 40},
    # ─── 청담 / 압구정 ──────────────────────────────────────────────────────
    {"id": "GALLERIA",     "x0": 1200, "x1": 1600, "y0": 3400, "y1": 3750,
     "label": "갤러리아백화점",             "color": "#8e44ad", "height": 50},
    {"id": "CHEONGDAM_1",  "x0": 1600, "x1": 1950, "y0": 3700, "y1": 4050,
     "label": "청담동 상권",                "color": "#27ae60", "height": 35},
    {"id": "CHEONGDAM_2",  "x0": 1950, "x1": 2250, "y0": 3700, "y1": 4050,
     "label": "청담 명품거리",              "color": "#27ae60", "height": 30},
    # ─── 주거 단지 ──────────────────────────────────────────────────────────
    {"id": "APT_W1",       "x0": 600,  "x1": 1000, "y0": 2050, "y1": 2500,
     "label": "서초구 아파트",              "color": "#bdc3c7", "height": 80},
    {"id": "APT_W2",       "x0": 600,  "x1": 1000, "y0": 1550, "y1": 2000,
     "label": "반포 아파트",                "color": "#bdc3c7", "height": 60},
    {"id": "APT_E1",       "x0": 4200, "x1": 4600, "y0": 2200, "y1": 2700,
     "label": "선릉 아파트",                "color": "#bdc3c7", "height": 70},
    {"id": "APT_N1",       "x0": 900,  "x1": 1300, "y0": 3500, "y1": 4000,
     "label": "압구정 아파트",              "color": "#bdc3c7", "height": 50},
    {"id": "APT_N2",       "x0": 2550, "x1": 2950, "y0": 3650, "y1": 4100,
     "label": "청담 아파트단지",            "color": "#bdc3c7", "height": 55},
    {"id": "APT_S1",       "x0": 1500, "x1": 1900, "y0": 1200, "y1": 1700,
     "label": "서초 주거단지",              "color": "#bdc3c7", "height": 65},
    {"id": "APT_S2",       "x0": 2800, "x1": 3200, "y0": 1350, "y1": 1850,
     "label": "강남 주거지구",              "color": "#bdc3c7", "height": 50},
    # ─── 잠실 / 롯데타워 방면 ───────────────────────────────────────────────
    {"id": "LOTTE_TWR",    "x0": 4400, "x1": 4700, "y0": 2300, "y1": 2700,
     "label": "롯데월드타워",               "color": "#c0392b", "height": 555},
    {"id": "LOTTE_MALL",   "x0": 4100, "x1": 4400, "y0": 2200, "y1": 2600,
     "label": "롯데월드몰",                 "color": "#8e44ad", "height": 45},
    # ─── 공공 / 기타 ────────────────────────────────────────────────────────
    {"id": "KANGNAM_GU",   "x0": 3100, "x1": 3400, "y0": 3200, "y1": 3550,
     "label": "강남구청",                   "color": "#16a085", "height": 25},
    {"id": "KSPO_DOME",    "x0": 4200, "x1": 4600, "y0": 3250, "y1": 3750,
     "label": "KSPO 돔",                    "color": "#16a085", "height": 30},
    # ─── 강남역 방면 ────────────────────────────────────────────────────────
    {"id": "GANGNAM_A",    "x0": 1500, "x1": 1800, "y0": 1850, "y1": 2200,
     "label": "강남역 상권A",               "color": "#27ae60", "height": 35},
    {"id": "GANGNAM_B",    "x0": 1800, "x1": 2100, "y0": 1850, "y1": 2200,
     "label": "강남역 상권B",               "color": "#27ae60", "height": 40},
    # ─── 남부 주거 ──────────────────────────────────────────────────────────
    {"id": "RES_S1",       "x0": 700,  "x1": 1200, "y0": 800,  "y1": 1400,
     "label": "서초 주거지",                "color": "#bdc3c7", "height": 40},
    {"id": "RES_S2",       "x0": 2000, "x1": 2500, "y0": 700,  "y1": 1300,
     "label": "양재 주거단지",              "color": "#bdc3c7", "height": 45},
    {"id": "RES_E1",       "x0": 3800, "x1": 4200, "y0": 3350, "y1": 3850,
     "label": "잠실 주거단지",              "color": "#bdc3c7", "height": 60},
]

REAL_ROADS: list[dict] = [
    {"name": "테헤란로",  "xs": [500,  4700], "ys": [2500, 2500], "w": 9,  "c": "#d5dbdb"},
    {"name": "영동대로",  "xs": [2500, 2500], "ys": [300,  4800], "w": 11, "c": "#d5dbdb"},
    {"name": "봉은사로",  "xs": [1900, 1900], "ys": [1500, 4200], "w": 5,  "c": "#e5e8e8"},
    {"name": "강남대로",  "xs": [1500, 1500], "ys": [300,  2500], "w": 8,  "c": "#d5dbdb"},
    {"name": "언주로",    "xs": [3200, 3200], "ys": [1500, 4200], "w": 5,  "c": "#e5e8e8"},
    {"name": "학동로",    "xs": [3800, 3800], "ys": [2000, 4500], "w": 5,  "c": "#e5e8e8"},
    {"name": "올림픽로",  "xs": [500,  4800], "ys": [1500, 1500], "w": 7,  "c": "#d5dbdb"},
    {"name": "압구정로",  "xs": [500,  3500], "ys": [3600, 3600], "w": 5,  "c": "#e5e8e8"},
    {"name": "도산대로",  "xs": [1200, 1200], "ys": [2500, 4500], "w": 5,  "c": "#e5e8e8"},
    {"name": "삼성로",    "xs": [2820, 2820], "ys": [2100, 3700], "w": 5,  "c": "#e5e8e8"},
]


# ── 성능 지표 데이터 로딩 & 계산 ─────────────────────────────────────────────
def _load_perf_data() -> dict:
    # 1. 모델 비교
    ens = json.load(open(OUT_DIR / "ensemble_results.json"))
    lstm_hist = json.load(open(OUT_DIR / "lstm_history.json"))
    trf_hist  = json.load(open(OUT_DIR / "transformer_history.json"))

    model_compare = {
        "models": ["LSTM", "Transformer", "Ensemble (평균)"],
        "test_acc": [
            ens["test"]["lstm"]["accuracy"] * 100,
            ens["test"]["transformer"]["accuracy"] * 100,
            ens["test"]["ensemble_avg"]["accuracy"] * 100,
        ],
        "test_f1": [
            ens["test"]["lstm"]["f1"] * 100,
            ens["test"]["transformer"]["f1"] * 100,
            ens["test"]["ensemble_avg"]["f1"] * 100,
        ],
        "val_acc": [
            ens["val"]["lstm"]["accuracy"] * 100,
            ens["val"]["transformer"]["accuracy"] * 100,
            ens["val"]["ensemble_avg"]["accuracy"] * 100,
        ],
        "train_history": {
            "lstm": lstm_hist.get("history", []),
            "transformer": trf_hist.get("history", []),
        },
    }

    # 2. 멀티홉 결과 계산
    NUM_UAVS = 5
    results = defaultdict(lambda: {"total": 0, "direct": 0, "single": 0, "multi": 0})
    for scenario, ts_dict in LINKS.items():
        for t_s, lnks in ts_dict.items():
            adj = defaultdict(set)
            for (a, b), info in lnks.items():
                if info["state"] != "disconnected":
                    adj[a].add(b); adj[b].add(a)
            for src in range(NUM_UAVS):
                for dst in range(src + 1, NUM_UAVS):
                    r = results[scenario]
                    r["total"] += 1
                    state = lnks.get((src, dst), {}).get("state", "disconnected")
                    direct = state != "disconnected"
                    if direct:
                        r["direct"] += 1; r["single"] += 1; r["multi"] += 1
                        continue
                    # single relay
                    relays = (adj.get(src, set()) & adj.get(dst, set())) - {src, dst}
                    if relays:
                        r["single"] += 1; r["multi"] += 1
                        continue
                    # BFS multi-hop
                    queue = deque([[src]]); visited = {src}; found = False
                    while queue and not found:
                        path = queue.popleft()
                        if len(path) > 4: break
                        for nb in adj.get(path[-1], set()):
                            if nb in visited: continue
                            if nb == dst: found = True; break
                            visited.add(nb); queue.append(path + [nb])
                    if found:
                        r["multi"] += 1

    multihop = {
        "scenarios": [],
        "direct": [], "single": [], "multi": [],
        "total_direct": 0, "total_single": 0, "total_multi": 0, "total_all": 0,
    }
    for sid in sorted(results):
        r = results[sid]
        if r["total"] == 0: continue
        multihop["scenarios"].append(sid)
        multihop["direct"].append(r["direct"] / r["total"] * 100)
        multihop["single"].append(r["single"] / r["total"] * 100)
        multihop["multi"].append(r["multi"] / r["total"] * 100)
        multihop["total_direct"] += r["direct"]
        multihop["total_single"] += r["single"]
        multihop["total_multi"]  += r["multi"]
        multihop["total_all"]    += r["total"]

    # 3. 온라인 학습 (하드코딩 — online_learning.py 실행 결과)
    online = {
        "scenarios": ["wave_disconnect", "slow_separation", "split_and_rejoin"],
        "before":    [97.94, 98.24, 98.24],
        "after":     [98.73, 98.53, 98.82],
    }

    # 4. 위치 보정 성공률
    correction = {
        "labels":  ["기존 (단순 모델)", "개선 후 (노이즈+감쇠)"],
        "success": [100.0, 90.0],
        "failed":  [0.0, 10.0],
        "note": "high_speed_scatter: 0% (드론 이미 100m+ 이탈)",
    }

    # 5. DQN 위치 보정 결과
    rl_correction = None
    _rl_path = OUT_DIR / "rl_correction_results.json"
    if _rl_path.exists():
        try:
            _rl = json.load(open(_rl_path))
            rl_correction = {
                "rl_success":   _rl["comparison"]["rl"]["success_rate"] * 100,
                "bl_success":   _rl["comparison"]["baseline"]["success_rate"] * 100,
                "rl_steps":     _rl["comparison"]["rl"]["avg_steps"],
                "bl_steps":     _rl["comparison"]["baseline"]["avg_steps"],
                "history":      _rl.get("history", []),
            }
        except Exception:
            pass

    return {
        "model_compare": model_compare,
        "multihop": multihop,
        "online": online,
        "correction": correction,
        "rl_correction": rl_correction,
    }


print("성능 지표 계산 중...")
PERF = _load_perf_data()
print("완료")


# ── DQN 위치 보정 상수 ────────────────────────────────────────────────────────
_DQN_COMM_RANGE  = 160.0
_DQN_RSSI_THRESH = -90.0
_DQN_TX_POWER    = -20.0
_DQN_FREQ_MHZ    = 2400.0
_DQN_DIRECTIONS  = [i * (math.pi / 4) for i in range(8)]
_DQN_STEP_SIZES  = [10.0, 20.0, 30.0, 40.0]

# ── DQN 모델 정의 + 가중치 로딩 ──────────────────────────────────────────────
_dqn_model = None

if _TORCH_OK:
    class _CorrDQN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 32),
            )
        def forward(self, x):
            return self.net(x)

    _dqn_path = OUT_DIR / "rl_correction_dqn.pt"
    if _dqn_path.exists():
        try:
            _dqn_model = _CorrDQN()
            _dqn_model.load_state_dict(
                torch.load(str(_dqn_path), map_location="cpu"))
            _dqn_model.eval()
            print("DQN 위치 보정 모델 로딩 완료")
        except Exception as _e:
            print(f"DQN 로딩 실패: {_e}")
            _dqn_model = None
    else:
        print("DQN 모델 파일 없음 — 고정 20m 방식 사용")


# ── 성능 지표 그래프 함수 ──────────────────────────────────────────────────────
def _make_model_compare() -> go.Figure:
    p = PERF["model_compare"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    fig = go.Figure()
    for metric, label, pattern in [("test_acc", "Accuracy", ""), ("test_f1", "F1 (macro)", "/")]:
        fig.add_trace(go.Bar(
            name=label, x=p["models"],
            y=[p[metric][i] for i in range(3)],
            marker=dict(color=colors, pattern_shape=pattern),
            text=[f"{v:.2f}%" for v in [p[metric][i] for i in range(3)]],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group", height=280,
        yaxis=dict(range=[85, 105], title="%"),
        margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def _make_train_curve() -> go.Figure:
    hist = PERF["model_compare"]["train_history"]
    fig = go.Figure()
    for model, color in [("lstm", "#3498db"), ("transformer", "#e74c3c")]:
        h = hist.get(model, [])
        if h:
            fig.add_trace(go.Scatter(
                x=[r["epoch"] for r in h],
                y=[r["val_acc"] * 100 for r in h],
                name=model.upper(), line=dict(color=color, width=2),
                mode="lines",
            ))
    fig.update_layout(
        height=280, xaxis_title="Epoch", yaxis_title="Val Accuracy (%)",
        margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def _make_multihop() -> go.Figure:
    mh = PERF["multihop"]
    tot = mh["total_all"]
    overall = {
        "Direct":    mh["total_direct"] / tot * 100,
        "Single":    mh["total_single"] / tot * 100,
        "Multi-hop": mh["total_multi"]  / tot * 100,
    }
    fig = go.Figure()
    colors = {"Direct": "#95a5a6", "Single": "#3498db", "Multi-hop": "#e74c3c"}
    # 전체 요약 바 (왼쪽) + 시나리오별 (오른쪽) — 시나리오별만 표시
    for method, key, color in [
        ("Direct",    "direct", "#95a5a6"),
        ("Single Relay", "single", "#3498db"),
        ("Multi-hop", "multi",  "#e74c3c"),
    ]:
        fig.add_trace(go.Bar(
            name=method, x=mh["scenarios"], y=mh[key],
            marker_color=color, opacity=0.85,
        ))
    fig.add_annotation(
        x=0.01, y=0.98, xref="paper", yref="paper", showarrow=False,
        text=(f"전체 평균: Direct {overall['Direct']:.1f}%  "
              f"Single {overall['Single']:.1f}%  "
              f"Multi {overall['Multi-hop']:.1f}%"),
        bgcolor="white", bordercolor="#ccc", borderwidth=1,
        font=dict(size=11), align="left",
    )
    fig.update_layout(
        barmode="group", height=300,
        yaxis=dict(range=[0, 110], title="연결 성공률 (%)"),
        xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
        margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.35),
    )
    return fig


def _make_online() -> go.Figure:
    ol = PERF["online"]
    fig = go.Figure()
    x = ol["scenarios"]
    fig.add_trace(go.Bar(name="적응 전", x=x, y=ol["before"],
                         marker_color="#95a5a6",
                         text=[f"{v:.2f}%" for v in ol["before"]],
                         textposition="inside"))
    fig.add_trace(go.Bar(name="적응 후", x=x, y=ol["after"],
                         marker_color="#27ae60",
                         text=[f"{v:.2f}%" for v in ol["after"]],
                         textposition="inside"))
    for i, (b, a) in enumerate(zip(ol["before"], ol["after"])):
        fig.add_annotation(x=x[i], y=a + 0.3, text=f"+{a-b:.2f}%p",
                           showarrow=False, font=dict(size=10, color="#27ae60"))
    fig.update_layout(
        barmode="group", height=220,
        yaxis=dict(range=[96, 100], title="Accuracy (%)"),
        margin=dict(l=8, r=8, t=8, b=40),
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.3),
        xaxis=dict(tickfont=dict(size=10)),
    )
    return fig


def _make_correction() -> go.Figure:
    cr = PERF["correction"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="성공", x=cr["labels"], y=cr["success"],
        marker_color=["#e74c3c", "#27ae60"],
        text=[f"{v:.0f}%" for v in cr["success"]],
        textposition="outside",
    ))
    fig.add_annotation(
        x=0.5, y=0.05, xref="paper", yref="paper", showarrow=False,
        text=cr["note"], font=dict(size=10, color="#555"),
        bgcolor="white", bordercolor="#ccc", borderwidth=1,
    )
    fig.update_layout(
        height=200, yaxis=dict(range=[0, 115], title="성공률 (%)"),
        margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def _make_rl_correction():
    rl = PERF.get("rl_correction")
    if rl is None:
        return go.Figure()
    fig = go.Figure()
    labels = ["기존 (고정 20m)", "DQN (강화학습)"]
    fig.add_trace(go.Bar(
        name="재연결 성공률",
        x=labels, y=[rl["bl_success"], rl["rl_success"]],
        marker_color=["#e74c3c", "#27ae60"],
        text=[f"{rl['bl_success']:.1f}%", f"{rl['rl_success']:.1f}%"],
        textposition="outside", yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        name="평균 이동 스텝",
        x=labels, y=[rl["bl_steps"], rl["rl_steps"]],
        mode="lines+markers+text",
        text=[f"{rl['bl_steps']:.1f}회", f"{rl['rl_steps']:.1f}회"],
        textposition="top center",
        line=dict(color="#8e44ad", width=2),
        marker=dict(size=10, color="#8e44ad"),
        yaxis="y2",
    ))
    step_max = max(rl["bl_steps"], rl["rl_steps"]) * 1.6
    fig.update_layout(
        height=220,
        yaxis=dict(range=[0, 115], title="성공률 (%)"),
        yaxis2=dict(range=[0, step_max], title="평균 스텝 수",
                    overlaying="y", side="right"),
        margin=dict(l=8, r=8, t=8, b=40),
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.3),
        xaxis=dict(tickfont=dict(size=10)),
    )
    return fig


def _make_rl_history():
    rl = PERF.get("rl_correction")
    if rl is None or not rl.get("history"):
        return go.Figure()
    hist = rl["history"]
    episodes = [h["episode"] for h in hist]
    rewards  = [h["reward"]  for h in hist]
    fig = go.Figure()
    # 이동 평균 (window=50)
    w = 50
    smoothed = [
        sum(rewards[max(0, i-w):i+1]) / len(rewards[max(0, i-w):i+1])
        for i in range(len(rewards))
    ]
    fig.add_trace(go.Scatter(
        x=episodes, y=rewards, name="에피소드 보상",
        mode="lines", line=dict(color="#95a5a6", width=1), opacity=0.4,
    ))
    fig.add_trace(go.Scatter(
        x=episodes, y=smoothed, name=f"이동 평균 (w={w})",
        mode="lines", line=dict(color="#2980b9", width=2),
    ))
    fig.update_layout(
        height=220,
        xaxis_title="Episode", yaxis_title="보상 (Reward)",
        margin=dict(l=8, r=8, t=8, b=40),
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


def get_ts(scenario: str) -> list[str]:
    return sorted(POSITIONS.get(scenario, {}).keys(), key=float)


# ── DQN 위치 보정 헬퍼 ────────────────────────────────────────────────────────

def _corr_seg_hits(x0, y0, x1, y1, bx0, bx1, by0, by1) -> bool:
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


def _corr_compute_rssi(ax, ay, bx, by, obstacles) -> float:
    d = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2) + 0.1
    fspl = 20 * math.log10(d) + 20 * math.log10(_DQN_FREQ_MHZ) - 27.55
    atten = sum(
        b["atten"] for b in obstacles
        if _corr_seg_hits(ax, ay, bx, by, b["x0"], b["x1"], b["y0"], b["y1"])
    )
    return _DQN_TX_POWER - fspl - atten


def _dqn_correction(iso_x: float, iso_y: float,
                    main_pos: list[tuple], obstacles: list[dict]
                    ) -> tuple[float, float]:
    """DQN 추론으로 격리 UAV의 이동 후 위치 반환. 모델 없으면 고정 20m centroid."""
    if _dqn_model is None or not main_pos:
        cx = sum(p[0] for p in main_pos) / max(len(main_pos), 1)
        cy = sum(p[1] for p in main_pos) / max(len(main_pos), 1)
        dx, dy = cx - iso_x, cy - iso_y
        d = math.sqrt(dx ** 2 + dy ** 2) + 1e-9
        return iso_x + STEP_ARROW * dx / d, iso_y + STEP_ARROW * dy / d

    cx = sum(p[0] for p in main_pos) / len(main_pos)
    cy = sum(p[1] for p in main_pos) / len(main_pos)

    nearest = min(main_pos, key=lambda p: (p[0] - iso_x) ** 2 + (p[1] - iso_y) ** 2)
    nearest_d    = math.sqrt((nearest[0] - iso_x) ** 2 + (nearest[1] - iso_y) ** 2)
    nearest_rssi = _corr_compute_rssi(iso_x, iso_y, nearest[0], nearest[1], obstacles)
    init_rssi    = max(_corr_compute_rssi(iso_x, iso_y, p[0], p[1], obstacles)
                       for p in main_pos)

    blocked = 1.0 if any(
        _corr_seg_hits(iso_x, iso_y, nearest[0], nearest[1],
                       b["x0"], b["x1"], b["y0"], b["y1"])
        for b in obstacles
    ) else 0.0

    state = np.clip(np.array([
        (cx - iso_x) / _DQN_COMM_RANGE,
        (cy - iso_y) / _DQN_COMM_RANGE,
        math.sqrt((cx - iso_x) ** 2 + (cy - iso_y) ** 2) / _DQN_COMM_RANGE,
        (nearest_rssi - _DQN_RSSI_THRESH) / 30.0,
        nearest_d / _DQN_COMM_RANGE,
        0.0,   # step_count = 0 (첫 번째 보정)
        blocked,
        (init_rssi - _DQN_RSSI_THRESH) / 30.0,
    ], dtype=np.float32), -3.0, 3.0)

    with torch.no_grad():
        action = int(_dqn_model(torch.FloatTensor(state).unsqueeze(0)).argmax().item())

    direction = _DQN_DIRECTIONS[action // len(_DQN_STEP_SIZES)]
    dist_m    = _DQN_STEP_SIZES[action % len(_DQN_STEP_SIZES)]
    return (iso_x + dist_m * math.cos(direction),
            iso_y + dist_m * math.sin(direction))


# ── 연결 컴포넌트 (위치 보정 트리거 판단) ─────────────────────────────────────
def _components(pos: dict[int, tuple], lnks: dict[tuple, dict]) -> list[set[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for (a, b), info in lnks.items():
        if info["state"] != "disconnected":
            adj[a].add(b); adj[b].add(a)
    visited, comps = set(), []
    for uid in pos:
        if uid not in visited:
            comp, stack = set(), [uid]
            while stack:
                n = stack.pop()
                if n in visited: continue
                visited.add(n); comp.add(n)
                stack.extend(adj[n] - visited)
            comps.append(comp)
    return comps


def _centroid(pos: dict[int, tuple], ids: set[int]) -> tuple[float, float]:
    xs = [pos[u][0] for u in ids]
    ys = [pos[u][1] for u in ids]
    return sum(xs)/len(xs), sum(ys)/len(ys)


# ── 지도 Figure 생성 ─────────────────────────────────────────────────────────
def make_figure(scenario: str, t_s: str,
                prev_relay: int | None, bad_streak: int,
                buildings: list[dict] | None = None) -> go.Figure:
    pos  = POSITIONS.get(scenario, {}).get(t_s, {})
    lnks = LINKS.get(scenario, {}).get(t_s, {})

    if buildings is None:
        buildings = OBSTACLES

    traces = []

    # ── 건물 ────────────────────────────────────────────────────────────────
    for b in buildings:
        xs = [b["x0"], b["x1"], b["x1"], b["x0"], b["x0"]]
        ys = [b["y0"], b["y0"], b["y1"], b["y1"], b["y0"]]
        traces.append(go.Scatter(
            x=xs, y=ys, fill="toself",
            fillcolor=BLDG_COLOR,
            line=dict(color="#666", width=1),
            mode="lines", hoverinfo="skip",
            name=f"{b['id']} ({b['atten']}dB)",
            legendgroup="buildings",
            showlegend=True,
        ))

    # ── 연결 컴포넌트 & 위치 보정 판단 ──────────────────────────────────────
    comps = _components(pos, lnks)
    need_correction = len(comps) > 1 and bad_streak >= HYSTERESIS
    isolated_uavs: set[int] = set()
    correction_arrows: list[tuple] = []  # (x0,y0,dx,dy)

    if need_correction:
        main_comp = max(comps, key=len)
        main_pos_list = [(pos[u][0], pos[u][1]) for u in main_comp if u in pos]
        for comp in comps:
            if comp == main_comp: continue
            for uid in comp:
                isolated_uavs.add(uid)
                x, y, _ = pos[uid]
                nx, ny = _dqn_correction(x, y, main_pos_list, buildings)
                correction_arrows.append((x, y, nx - x, ny - y))

    # ── 위치 보정 화살표 (점선) ──────────────────────────────────────────────
    for x0, y0, dx, dy in correction_arrows:
        traces.append(go.Scatter(
            x=[x0, x0 + dx], y=[y0, y0 + dy],
            mode="lines",
            line=dict(color=UAV_ISOLATED, width=2, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))
        traces.append(go.Scatter(
            x=[x0 + dx], y=[y0 + dy],
            mode="markers",
            marker=dict(symbol="arrow", size=14, angle=math.degrees(math.atan2(dy, dx)),
                        color=UAV_ISOLATED),
            hoverinfo="skip", showlegend=False,
        ))

    # ── 링크 ────────────────────────────────────────────────────────────────
    for (src, dst), info in lnks.items():
        if src not in pos or dst not in pos: continue
        x0, y0, _ = pos[src]; x1, y1, _ = pos[dst]
        traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(color=STATE_COLOR[info["state"]], width=2.5),
            hovertemplate=(f"UAV{src}↔UAV{dst}<br>"
                           f"state: {info['state']}<br>"
                           f"RSSI: {info['rssi']:.1f}dBm  PLR: {info['plr']:.1f}%"
                           "<extra></extra>"),
            showlegend=False,
        ))

    # ── 현재 relay 판별 ──────────────────────────────────────────────────────
    relay_vals = [v["relay"] for v in lnks.values()]
    cur_relay  = max(set(relay_vals), key=relay_vals.count) if relay_vals else None

    # ── Relay 전환 화살표 ────────────────────────────────────────────────────
    if (prev_relay is not None and cur_relay is not None
            and prev_relay != cur_relay
            and prev_relay in pos and cur_relay in pos):
        px, py, _ = pos[prev_relay]
        nx, ny, _ = pos[cur_relay]
        traces.append(go.Scatter(
            x=[px, (px+nx)/2, nx], y=[py, (py+ny)/2+8, ny],
            mode="lines",
            line=dict(color="#f39c12", width=2.5, dash="dash"),
            hoverinfo="skip", showlegend=False,
        ))
        traces.append(go.Scatter(
            x=[nx], y=[ny],
            mode="markers",
            marker=dict(symbol="arrow", size=16,
                        angle=math.degrees(math.atan2(ny-py, nx-px)),
                        color="#f39c12"),
            hoverinfo="skip", showlegend=False,
        ))

    # ── UAV 노드 ────────────────────────────────────────────────────────────
    for uid in sorted(pos):
        x, y, role = pos[uid]
        is_relay    = (uid == cur_relay)
        is_isolated = (uid in isolated_uavs)
        color  = UAV_RELAY if is_relay else (UAV_ISOLATED if is_isolated else UAV_NORMAL)
        symbol = "star" if is_relay else ("circle-open" if is_isolated else "circle")
        size   = 22 if is_relay else 18
        label  = f"UAV{uid}" + (" ⭐" if is_relay else (" ⚠" if is_isolated else ""))
        hover  = (f"<b>UAV{uid}</b><br>role: {role}<br>"
                  f"({x:.1f}, {y:.1f})<br>"
                  + ("현재 Relay<br>" if is_relay else "")
                  + ("격리됨 → 보정 중<br>" if is_isolated else "")
                  + "<extra></extra>")
        traces.append(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=size, color=color,
                        symbol=symbol,
                        line=dict(color="white", width=2)),
            text=[label],
            textposition="top center",
            textfont=dict(size=10, color="#2c3e50"),
            hovertemplate=hover,
            name=f"UAV{uid}",
            showlegend=False,
        ))

    # ── 위치 보정 격리 원 ────────────────────────────────────────────────────
    if need_correction:
        for uid in isolated_uavs:
            x, y, _ = pos[uid]
            theta = [i * math.pi / 18 for i in range(37)]
            r = 10
            traces.append(go.Scatter(
                x=[x + r*math.cos(a) for a in theta],
                y=[y + r*math.sin(a) for a in theta],
                mode="lines",
                line=dict(color=UAV_ISOLATED, width=1.5, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))

    # ── 레이아웃 ────────────────────────────────────────────────────────────
    all_x = [v[0] for d in POSITIONS.get(scenario, {}).values() for v in d.values()]
    all_y = [v[1] for d in POSITIONS.get(scenario, {}).values() for v in d.values()]
    pad = 20

    annotations = []
    if need_correction:
        annotations.append(dict(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text="⚠ 위치 보정 발동", showarrow=False,
            bgcolor="#8e44ad", font=dict(color="white", size=12),
            borderpad=4,
        ))
    if prev_relay is not None and cur_relay != prev_relay:
        annotations.append(dict(
            x=0.02, y=0.90, xref="paper", yref="paper",
            text=f"🔀 Relay 전환: UAV{prev_relay}→UAV{cur_relay}",
            showarrow=False,
            bgcolor="#e67e22", font=dict(color="white", size=12),
            borderpad=4,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(range=[min(all_x)-pad, max(all_x)+pad],
                   showgrid=True, gridcolor="#eee", title="X (m)"),
        yaxis=dict(range=[min(all_y)-pad, max(all_y)+pad],
                   showgrid=True, gridcolor="#eee", title="Y (m)", scaleanchor="x"),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        height=480,
        annotations=annotations,
        legend=dict(x=1.01, y=1, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#ccc", borderwidth=1),
    )
    return fig


# ── 3D Figure ────────────────────────────────────────────────────────────────

def _sf(v, default=0.0) -> float:
    """Safe float conversion for DataTable values (may be None or str)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _building_box_traces(b: dict) -> list[go.Scatter3d]:
    """Return Scatter3d wireframe traces for a 3D building box."""
    x0, x1 = _sf(b["x0"]), _sf(b["x1"])
    y0, y1 = _sf(b["y0"]), _sf(b["y1"])
    h       = _sf(b.get("height", 30.0), 30.0)

    def _seg(pts):
        xs = [p[0] for p in pts] + [None]
        ys = [p[1] for p in pts] + [None]
        zs = [p[2] for p in pts] + [None]
        return xs, ys, zs

    corners_bot = [(x0,y0,0),(x1,y0,0),(x1,y1,0),(x0,y1,0),(x0,y0,0)]
    corners_top = [(x0,y0,h),(x1,y0,h),(x1,y1,h),(x0,y1,h),(x0,y0,h)]
    verticals   = [(x0,y0,0),(x0,y0,h),(None,None,None),
                   (x1,y0,0),(x1,y0,h),(None,None,None),
                   (x1,y1,0),(x1,y1,h),(None,None,None),
                   (x0,y1,0),(x0,y1,h)]

    all_pts = corners_bot + [(None,None,None)] + corners_top + [(None,None,None)] + verticals
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    zs = [p[2] for p in all_pts]
    return [go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color="rgba(100,100,100,0.55)", width=3),
        hovertemplate=f"<b>{b.get('id','?')}</b><br>높이: {h}m<br>감쇠: {_sf(b.get('atten',6.0),6.0)}dB<extra></extra>",
        name=f"{b['id']}",
        legendgroup="buildings3d",
        showlegend=True,
    )]


def make_figure_3d(scenario: str, t_s: str,
                   prev_relay: int | None,
                   buildings: list[dict] | None = None) -> go.Figure:
    pos  = POSITIONS_3D.get(scenario, {}).get(t_s, {})
    lnks = LINKS_3D.get(scenario, {}).get(t_s, {})

    # 건물 목록: Tab1 DataTable → 3D 공유
    if not buildings:
        buildings = OBSTACLES_3D

    traces: list = []

    # Buildings (Tab1과 동일한 건물 데이터 사용)
    for b in buildings:
        traces.extend(_building_box_traces(b))

    # Links
    for (src, dst), info in lnks.items():
        if src not in pos or dst not in pos:
            continue
        x0, y0, z0, _ = pos[src]
        x1, y1, z1, _ = pos[dst]
        color = STATE_COLOR[info["state"]]
        traces.append(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode="lines",
            line=dict(color=color, width=4),
            hovertemplate=(f"UAV{src}↔UAV{dst}<br>state: {info['state']}<br>"
                           f"RSSI: {info['rssi']:.1f}dBm<extra></extra>"),
            showlegend=False,
        ))

    # Current relay
    relay_vals = [v["relay"] for v in lnks.values()]
    cur_relay  = max(set(relay_vals), key=relay_vals.count) if relay_vals else None

    # UAV nodes
    for uid in sorted(pos):
        x, y, z, role = pos[uid]
        is_relay = (uid == cur_relay)
        color  = UAV_RELAY if is_relay else UAV_NORMAL
        symbol = "diamond" if is_relay else "circle"
        size   = 12 if is_relay else 8
        label  = f"UAV{uid}" + (" ⭐" if is_relay else "")
        traces.append(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="markers+text",
            marker=dict(size=size, color=color, symbol=symbol,
                        line=dict(color="white", width=1)),
            text=[label],
            textposition="top center",
            textfont=dict(size=9, color="#2c3e50"),
            hovertemplate=(f"<b>UAV{uid}</b><br>role: {role}<br>"
                           f"({x:.1f}, {y:.1f}, {z:.1f}m)<extra></extra>"),
            name=f"UAV{uid}",
            showlegend=False,
        ))

    # Relay switch arrow
    if (prev_relay is not None and cur_relay is not None
            and prev_relay != cur_relay
            and prev_relay in pos and cur_relay in pos):
        px, py, pz, _ = pos[prev_relay]
        nx, ny, nz, _ = pos[cur_relay]
        traces.append(go.Scatter3d(
            x=[px, (px+nx)/2, nx], y=[py, (py+ny)/2, ny], z=[pz, (pz+nz)/2+5, nz],
            mode="lines",
            line=dict(color="#f39c12", width=3, dash="dash"),
            hoverinfo="skip", showlegend=False,
        ))

    # ── 축 범위: 시나리오 전체 기준으로 고정 (매 프레임 재계산 방지) ────────────
    all_data  = POSITIONS_3D.get(scenario, {})
    all_x_all = [v[0] for d in all_data.values() for v in d.values()]
    all_y_all = [v[1] for d in all_data.values() for v in d.values()]
    all_z_all = [v[2] for d in all_data.values() for v in d.values()]
    pad = 20
    x_range = [min(all_x_all) - pad, max(all_x_all) + pad]
    y_range = [min(all_y_all) - pad, max(all_y_all) + pad]
    z_range = [0, max(all_z_all) + pad]

    fig = go.Figure(data=traces)
    fig.update_layout(
        # uirevision=scenario → 시나리오가 같으면 카메라 각도 보존
        uirevision=scenario,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            # 축 범위 고정 (autorange=False) → 배경이 늘어나거나 축이 변하지 않음
            xaxis=dict(title="X (m)", range=x_range, autorange=False,
                       backgroundcolor="#f4f6f8", gridcolor="#ccc",
                       showspikes=False),
            yaxis=dict(title="Y (m)", range=y_range, autorange=False,
                       backgroundcolor="#f4f6f8", gridcolor="#ccc",
                       showspikes=False),
            zaxis=dict(title="고도 (m)", range=z_range, autorange=False,
                       backgroundcolor="#e8eeff", gridcolor="#aab",
                       showspikes=False),
            bgcolor="#f4f6f8",
            # 드론 이동(주로 +X 방향)이 가장 잘 보이는 각도:
            # 남쪽(Y-)에서 약간 높게 바라보는 뷰 → X·Z 동시 관찰
            camera=dict(
                eye=dict(x=0.4, y=-2.4, z=0.7),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
                projection=dict(type="perspective"),
            ),
            dragmode="turntable",   # 수평 회전만 허용 (위아래 뒤집힘 방지)
            aspectmode="manual",
            aspectratio=dict(x=2.5, y=1.0, z=0.6),  # X축을 길게 → 이동 경로 강조
        ),
        paper_bgcolor="white",
        height=500,
        legend=dict(x=1.01, y=1, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#ccc", borderwidth=1),
    )
    return fig


# ── 멀티홉 연결성 계산 ────────────────────────────────────────────────────────
def _multihop_connectivity(lnks: dict) -> dict:
    """현재 타임스텝 링크 상태로 멀티홉 연결성 계산."""
    uavs: set[int] = set()
    for (a, b) in lnks:
        uavs.add(a); uavs.add(b)
    if len(uavs) < 2:
        return {"all_direct": True, "all_multihop": True, "multihop_gain": 0}

    adj: dict[int, set[int]] = defaultdict(set)
    for (a, b), info in lnks.items():
        if info["state"] != "disconnected":
            adj[a].add(b); adj[b].add(a)

    # 직접 연결 확인 (모든 쌍)
    all_direct = all(
        lnks.get((a, b), lnks.get((b, a), {})).get("state", "disconnected") != "disconnected"
        for a in uavs for b in uavs if a < b
    )

    # BFS 전체 연결 확인
    start = next(iter(uavs))
    visited = {start}
    queue = [start]
    while queue:
        node = queue.pop()
        for nb in adj.get(node, set()):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    all_multihop = (visited == uavs)

    # 멀티홉으로 추가 연결된 쌍 수
    multihop_gain = 0
    for a in uavs:
        for b in uavs:
            if a >= b:
                continue
            direct = lnks.get((a, b), lnks.get((b, a), {})).get("state", "disconnected") != "disconnected"
            if not direct and b in visited and a in visited:
                multihop_gain += 1

    return {"all_direct": all_direct, "all_multihop": all_multihop,
            "multihop_gain": multihop_gain, "n_uavs": len(uavs)}


# ── 실제 환경 Figure ─────────────────────────────────────────────────────────

def _hex_fill(hex_color: str, alpha: float = 0.50) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_figure_real(scenario: str, t_s: str,
                     prev_relay: int | None,
                     bad_streak: int) -> go.Figure:
    pos  = POSITIONS.get(scenario, {}).get(t_s, {})
    lnks = LINKS.get(scenario, {}).get(t_s, {})

    pos_real: dict[int, tuple[float, float, str]] = {}
    for uid, (x, y, role) in pos.items():
        rx, ry = _to_real(x, y)
        pos_real[uid] = (rx, ry, role)

    traces: list = []

    # 도로 (하단 레이어)
    for road in REAL_ROADS:
        traces.append(go.Scatter(
            x=road["xs"], y=road["ys"],
            mode="lines",
            line=dict(color=road["c"], width=road["w"]),
            hoverinfo="skip", showlegend=False,
        ))

    # 건물
    for b in REAL_BUILDINGS:
        xs = [b["x0"], b["x1"], b["x1"], b["x0"], b["x0"]]
        ys = [b["y0"], b["y0"], b["y1"], b["y1"], b["y0"]]
        traces.append(go.Scatter(
            x=xs, y=ys, fill="toself",
            fillcolor=_hex_fill(b["color"], 0.55),
            line=dict(color=b["color"], width=1.5),
            mode="lines",
            hovertemplate=f"<b>{b['label']}</b><br>높이: {b['height']}m<extra></extra>",
            showlegend=False,
        ))
        area = (b["x1"] - b["x0"]) * (b["y1"] - b["y0"])
        if area > 55000:
            traces.append(go.Scatter(
                x=[(b["x0"]+b["x1"])/2], y=[(b["y0"]+b["y1"])/2],
                mode="text", text=[b["label"]],
                textfont=dict(size=7, color="#2c3e50"),
                hoverinfo="skip", showlegend=False,
            ))

    # 위치 보정 판단
    comps = _components(pos, lnks)
    need_correction = len(comps) > 1 and bad_streak >= HYSTERESIS
    isolated_uavs: set[int] = set()
    correction_arrows: list[tuple] = []

    if need_correction:
        main_comp = max(comps, key=len)
        main_pos_list = [(pos[u][0], pos[u][1]) for u in main_comp if u in pos]
        for comp in comps:
            if comp == main_comp:
                continue
            for uid in comp:
                isolated_uavs.add(uid)
                rx, ry, _ = pos_real[uid]
                vx, vy, _ = pos[uid]
                # DQN은 가상 좌표계에서 추론 → 결과를 실제 좌표로 변환
                nx_v, ny_v = _dqn_correction(vx, vy, main_pos_list, OBSTACLES)
                nx_r, ny_r = _to_real(nx_v, ny_v)
                correction_arrows.append((rx, ry, nx_r - rx, ny_r - ry))

    for x0, y0, dx, dy in correction_arrows:
        traces.append(go.Scatter(
            x=[x0, x0+dx], y=[y0, y0+dy], mode="lines",
            line=dict(color=UAV_ISOLATED, width=2, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))
        traces.append(go.Scatter(
            x=[x0+dx], y=[y0+dy], mode="markers",
            marker=dict(symbol="arrow", size=14,
                        angle=math.degrees(math.atan2(dy, dx)),
                        color=UAV_ISOLATED),
            hoverinfo="skip", showlegend=False,
        ))

    # 링크
    for (src, dst), info in lnks.items():
        if src not in pos_real or dst not in pos_real:
            continue
        x0, y0, _ = pos_real[src]
        x1, y1, _ = pos_real[dst]
        traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(color=STATE_COLOR[info["state"]], width=3),
            hovertemplate=(f"UAV{src}↔UAV{dst}<br>state: {info['state']}<br>"
                           f"RSSI: {info['rssi']:.1f}dBm  PLR: {info['plr']:.1f}%"
                           "<extra></extra>"),
            showlegend=False,
        ))

    relay_vals = [v["relay"] for v in lnks.values()]
    cur_relay  = max(set(relay_vals), key=relay_vals.count) if relay_vals else None

    # Relay 전환 화살표
    if (prev_relay is not None and cur_relay is not None
            and prev_relay != cur_relay
            and prev_relay in pos_real and cur_relay in pos_real):
        px, py, _ = pos_real[prev_relay]
        nx, ny, _ = pos_real[cur_relay]
        traces.append(go.Scatter(
            x=[px, (px+nx)/2, nx], y=[py, (py+ny)/2+80, ny], mode="lines",
            line=dict(color="#f39c12", width=2.5, dash="dash"),
            hoverinfo="skip", showlegend=False,
        ))
        traces.append(go.Scatter(
            x=[nx], y=[ny], mode="markers",
            marker=dict(symbol="arrow", size=16,
                        angle=math.degrees(math.atan2(ny-py, nx-px)),
                        color="#f39c12"),
            hoverinfo="skip", showlegend=False,
        ))

    # UAV 노드
    for uid in sorted(pos_real):
        x, y, role = pos_real[uid]
        is_relay    = (uid == cur_relay)
        is_isolated = (uid in isolated_uavs)
        color  = UAV_RELAY if is_relay else (UAV_ISOLATED if is_isolated else UAV_NORMAL)
        symbol = "star" if is_relay else ("circle-open" if is_isolated else "circle")
        size   = 22 if is_relay else 18
        label  = f"UAV{uid}" + (" ⭐" if is_relay else (" ⚠" if is_isolated else ""))
        orig   = pos.get(uid, (0, 0, ""))
        traces.append(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=size, color=color, symbol=symbol,
                        line=dict(color="white", width=2)),
            text=[label], textposition="top center",
            textfont=dict(size=10, color="#2c3e50"),
            hovertemplate=(f"<b>UAV{uid}</b><br>role: {role}<br>"
                           f"실제위치: ({x:.0f}m, {y:.0f}m)<br>"
                           f"고도: {UAV_ALTITUDE_M:.0f}m (정찰 비행)<br>"
                           + ("현재 Relay<br>" if is_relay else "")
                           + ("격리됨 → 보정 중<br>" if is_isolated else "")
                           + "<extra></extra>"),
            name=f"UAV{uid}", showlegend=False,
        ))

    # 격리 원
    if need_correction:
        for uid in isolated_uavs:
            x, y, _ = pos_real[uid]
            theta = [i * math.pi / 18 for i in range(37)]
            r = 120
            traces.append(go.Scatter(
                x=[x + r*math.cos(a) for a in theta],
                y=[y + r*math.sin(a) for a in theta],
                mode="lines",
                line=dict(color=UAV_ISOLATED, width=1.5, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))

    # 삼성역 마커
    traces.append(go.Scatter(
        x=[2500], y=[2500], mode="markers+text",
        marker=dict(size=13, color="#f39c12", symbol="diamond",
                    line=dict(color="white", width=2)),
        text=["삼성역"], textposition="top center",
        textfont=dict(size=11, color="#f39c12"),
        hovertemplate="<b>삼성역</b><br>지도 중심점<extra></extra>",
        showlegend=False,
    ))

    annotations = []
    if need_correction:
        annotations.append(dict(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text="⚠ 위치 보정 발동", showarrow=False,
            bgcolor="#8e44ad", font=dict(color="white", size=12), borderpad=4,
        ))
    if prev_relay is not None and cur_relay != prev_relay:
        annotations.append(dict(
            x=0.02, y=0.90, xref="paper", yref="paper",
            text=f"🔀 Relay 전환: UAV{prev_relay}→UAV{cur_relay}",
            showarrow=False, bgcolor="#e67e22",
            font=dict(color="white", size=12), borderpad=4,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(
            range=[0, 5000], showgrid=False, zeroline=False,
            tickvals=[0, 1000, 2000, 3000, 4000, 5000],
            ticktext=["0", "1km", "2km", "3km", "4km", "5km"],
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            range=[0, 5000], showgrid=False, zeroline=False,
            tickvals=[0, 1000, 2000, 3000, 4000, 5000],
            ticktext=["0", "1km", "2km", "3km", "4km", "5km"],
            tickfont=dict(size=10), scaleanchor="x",
        ),
        plot_bgcolor="#eaecee",
        paper_bgcolor="white",
        height=560,
        annotations=annotations,
    )
    return fig


# ── 앱 ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="UAV 통신 대시보드")

CARD = {"background": "white", "borderRadius": 8, "padding": 16,
        "boxShadow": "0 1px 4px rgba(0,0,0,.1)", "marginBottom": 16}

app.layout = html.Div([
    dcc.Store(id="frame-store", data=0),
    dcc.Store(id="prev-relay-store", data=None),
    dcc.Store(id="bad-streak-store", data=0),
    dcc.Store(id="frame-store-3d", data=0),
    dcc.Store(id="prev-relay-store-3d", data=None),
    dcc.Interval(id="interval", interval=600, n_intervals=0, disabled=True),
    dcc.Interval(id="interval-3d", interval=600, n_intervals=0, disabled=True),
    dcc.Store(id="frame-store-real", data=0),
    dcc.Store(id="prev-relay-store-real", data=None),
    dcc.Store(id="bad-streak-store-real", data=0),
    dcc.Interval(id="interval-real", interval=600, n_intervals=0, disabled=True),
    dcc.Interval(id="ns3-interval", interval=2000, n_intervals=0),

    # ── 헤더 ─────────────────────────────────────────────────────────────────
    html.Div([
        html.H2("UAV 군집 통신 장애 예측 대시보드",
                style={"margin": 0, "color": "white", "fontSize": 20}),
        html.Span("중계 전환 & 위치 보정 시각화",
                  style={"color": "#bdc3c7", "fontSize": 13}),
    ], style={"background": "#2c3e50", "padding": "14px 24px",
              "display": "flex", "flexDirection": "column"}),

    dcc.Tabs(id="main-tabs", value="tab-real", children=[
    # ════════════════════════════════════════════════════════════════════════
    # TAB 1: 실제 환경 시뮬레이션 (삼성역 5km×5km)
    # ════════════════════════════════════════════════════════════════════════
    dcc.Tab(label="실제 환경 시뮬레이션", value="tab-real", children=[

    html.Div([
        html.Div([
            html.Label("시나리오", style={"fontSize": 12, "fontWeight": "bold"}),
            dcc.Dropdown(
                id="scenario-dd-real", clearable=False,
                options=[{"label": s, "value": s} for s in REAL_SCENARIOS],
                value=REAL_SCENARIOS[0] if REAL_SCENARIOS else None,
                style={"fontSize": 13, "width": 260},
            ),
        ]),
        html.Div([
            html.Label("재생 속도", style={"fontSize": 12, "fontWeight": "bold"}),
            dcc.Slider(id="speed-slider-real", min=1, max=5, step=1, value=2,
                       marks={1:"느림", 3:"보통", 5:"빠름"},
                       tooltip={"placement": "bottom"}, updatemode="drag"),
        ], style={"width": 200, "marginLeft": 20}),
        html.Div([
            html.Button("▶ Play",  id="play-btn-real",  n_clicks=0,
                        style={"marginRight": 8, "padding": "6px 18px",
                               "background": "#27ae60", "color": "white",
                               "border": "none", "borderRadius": 4,
                               "cursor": "pointer", "fontSize": 14}),
            html.Button("⏸ Pause", id="pause-btn-real", n_clicks=0,
                        style={"padding": "6px 18px",
                               "background": "#e74c3c", "color": "white",
                               "border": "none", "borderRadius": 4,
                               "cursor": "pointer", "fontSize": 14}),
        ], style={"marginLeft": 20, "alignSelf": "flex-end"}),
        html.Div(id="time-label-real",
                 style={"marginLeft": 20, "alignSelf": "flex-end",
                        "fontSize": 13, "color": "#555", "minWidth": 140}),
        html.Div([
            html.Span(f"정찰 고도 {UAV_ALTITUDE_M:.0f}m | 삼성역 중심 5km×5km",
                      style={"fontSize": 11, "color": "#888"}),
        ], style={"marginLeft": "auto", "alignSelf": "flex-end", "paddingRight": 8}),
    ], style={"display": "flex", "alignItems": "flex-end", "gap": 0,
              "padding": "14px 24px", "background": "#ecf0f1",
              "borderBottom": "1px solid #ddd"}),

    html.Div([
        dcc.Slider(id="frame-slider-real", min=0, max=1, step=1, value=0,
                   marks={}, updatemode="drag",
                   tooltip={"placement": "bottom", "always_visible": False}),
    ], style={"padding": "8px 24px", "background": "#ecf0f1",
              "borderBottom": "1px solid #ddd"}),

    html.Div([
        # 지도
        html.Div([
            dcc.Graph(id="map-graph-real", config={"displayModeBar": False}),
        ], style={"flex": "3", "background": "white", "borderRadius": 8,
                  "padding": 12, "marginRight": 12,
                  "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

        # 사이드 패널
        html.Div([
            # ── 멀티홉 결과 카드 (최상단 강조) ──────────────────────────────
            html.Div(id="multihop-result-real",
                     style={"marginBottom": 12}),

            html.Div([
                html.H4("링크 상태", style={"margin": "0 0 8px", "fontSize": 13,
                                           "color": "#2c3e50"}),
                html.Div(id="state-summary-real"),
            ], style={"background": "white", "borderRadius": 8, "padding": 12,
                      "marginBottom": 12, "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

            html.Div([
                html.H4("Relay 전환 이력", style={"margin": "0 0 8px", "fontSize": 13,
                                                  "color": "#e67e22"}),
                html.Div(id="relay-log-real",
                         style={"maxHeight": 100, "overflowY": "auto", "fontSize": 12}),
            ], style={"background": "white", "borderRadius": 8, "padding": 12,
                      "marginBottom": 12, "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

            html.Div([
                html.H4("위치 보정 이력", style={"margin": "0 0 8px", "fontSize": 13,
                                               "color": "#8e44ad"}),
                html.Div(id="correction-log-real",
                         style={"maxHeight": 85, "overflowY": "auto", "fontSize": 12}),
            ], style={"background": "white", "borderRadius": 8, "padding": 12,
                      "marginBottom": 12, "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

            html.Div([
                html.Div("링크", style={"fontWeight": "bold", "fontSize": 11,
                                       "marginBottom": 4}),
                *[html.Div([
                    html.Span("━━", style={"color": STATE_COLOR[s], "marginRight": 6}),
                    html.Span(s, style={"fontSize": 11}),
                ]) for s in ["healthy", "degraded", "disconnected"]],
                html.Div("UAV", style={"fontWeight": "bold", "fontSize": 11,
                                       "margin": "8px 0 4px"}),
                html.Div("⭐ 현재 Relay",   style={"color": UAV_RELAY,    "fontSize": 11}),
                html.Div("⚠ 격리/보정중",  style={"color": UAV_ISOLATED, "fontSize": 11}),
                html.Div("◆ 삼성역 중심",  style={"color": "#f39c12",    "fontSize": 11}),
                html.Div("건물 유형", style={"fontWeight": "bold", "fontSize": 11,
                                           "margin": "8px 0 4px"}),
                *[html.Div([
                    html.Span("■", style={"color": c, "marginRight": 5}),
                    html.Span(lbl, style={"fontSize": 10}),
                ]) for lbl, c in [
                    ("초고층 오피스",  "#c0392b"),
                    ("고층 오피스",   "#e67e22"),
                    ("호텔/컨벤션",   "#3498db"),
                    ("백화점",        "#8e44ad"),
                    ("상업/문화",     "#27ae60"),
                    ("주거",          "#bdc3c7"),
                ]],
            ], style={"background": "white", "borderRadius": 8, "padding": 12,
                      "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

        ], style={"flex": "1.2", "display": "flex", "flexDirection": "column",
                  "overflowY": "auto", "maxHeight": "calc(100vh - 150px)"}),

    ], style={"display": "flex", "padding": "14px 24px",
              "background": "#f5f6fa", "minHeight": "calc(100vh - 150px)"}),

    ]),  # end Tab 1 (실제 환경)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2: 2D 시뮬레이션
    # ════════════════════════════════════════════════════════════════════════
    dcc.Tab(label="2D 시뮬레이션", value="tab-sim", children=[

    # ── 컨트롤 바 ────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Label("시나리오", style={"fontSize": 12, "fontWeight": "bold"}),
            dcc.Dropdown(
                id="scenario-dd", clearable=False,
                options=[{"label": s, "value": s} for s in SCENARIOS],
                value=SCENARIOS[0],
                style={"fontSize": 13, "width": 260},
            ),
        ]),
        html.Div([
            html.Label("재생 속도", style={"fontSize": 12, "fontWeight": "bold"}),
            dcc.Slider(id="speed-slider", min=1, max=5, step=1, value=2,
                       marks={1:"느림", 3:"보통", 5:"빠름"},
                       tooltip={"placement": "bottom"}, updatemode="drag"),
        ], style={"width": 200, "marginLeft": 20}),
        html.Div([
            html.Button("▶ Play",  id="play-btn",  n_clicks=0,
                        style={"marginRight": 8, "padding": "6px 18px",
                               "background": "#27ae60", "color": "white",
                               "border": "none", "borderRadius": 4, "cursor": "pointer",
                               "fontSize": 14}),
            html.Button("⏸ Pause", id="pause-btn", n_clicks=0,
                        style={"padding": "6px 18px",
                               "background": "#e74c3c", "color": "white",
                               "border": "none", "borderRadius": 4, "cursor": "pointer",
                               "fontSize": 14}),
        ], style={"marginLeft": 20, "alignSelf": "flex-end"}),
        html.Div(id="time-label",
                 style={"marginLeft": 20, "alignSelf": "flex-end",
                        "fontSize": 13, "color": "#555", "minWidth": 140}),
    ], style={"display": "flex", "alignItems": "flex-end", "gap": 0,
              "padding": "14px 24px", "background": "#ecf0f1",
              "borderBottom": "1px solid #ddd"}),

    # ── 타임스텝 슬라이더 ─────────────────────────────────────────────────────
    html.Div([
        dcc.Slider(id="frame-slider", min=0, max=1, step=1, value=0,
                   marks={}, updatemode="drag",
                   tooltip={"placement": "bottom", "always_visible": False}),
    ], style={"padding": "8px 24px", "background": "#ecf0f1",
              "borderBottom": "1px solid #ddd"}),

    # ── 메인 ─────────────────────────────────────────────────────────────────
    html.Div([
        # 지도
        html.Div([
            dcc.Graph(id="map-graph", config={"displayModeBar": False}),
        ], style={"flex": "3", "background": "white", "borderRadius": 8,
                  "padding": 12, "marginRight": 12,
                  "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

        # 사이드 패널
        html.Div([
            # 링크 상태 요약
            html.Div([
                html.H4("링크 상태", style={"margin": "0 0 8px", "fontSize": 13,
                                           "color": "#2c3e50"}),
                html.Div(id="state-summary"),
            ], style={"background": "white", "borderRadius": 8, "padding": 12,
                      "marginBottom": 12, "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

            # Relay 이력
            html.Div([
                html.H4("Relay 전환 이력", style={"margin": "0 0 8px", "fontSize": 13,
                                                  "color": "#e67e22"}),
                html.Div(id="relay-log",
                         style={"maxHeight": 130, "overflowY": "auto",
                                "fontSize": 12}),
            ], style={"background": "white", "borderRadius": 8, "padding": 12,
                      "marginBottom": 12, "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

            # 위치 보정 이력
            html.Div([
                html.H4("위치 보정 이력", style={"margin": "0 0 8px", "fontSize": 13,
                                               "color": "#8e44ad"}),
                html.Div(id="correction-log",
                         style={"maxHeight": 110, "overflowY": "auto",
                                "fontSize": 12}),
            ], style={"background": "white", "borderRadius": 8, "padding": 12,
                      "marginBottom": 12, "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

            # 범례
            html.Div([
                html.Div("링크", style={"fontWeight": "bold", "fontSize": 11,
                                       "marginBottom": 4}),
                *[html.Div([
                    html.Span("━━", style={"color": STATE_COLOR[s], "marginRight": 6}),
                    html.Span(s, style={"fontSize": 11}),
                ]) for s in ["healthy", "degraded", "disconnected"]],
                html.Div("UAV", style={"fontWeight": "bold", "fontSize": 11,
                                       "margin": "8px 0 4px"}),
                html.Div("⭐ 현재 Relay", style={"color": UAV_RELAY,   "fontSize": 11}),
                html.Div("⚠ 격리/보정중", style={"color": UAV_ISOLATED,"fontSize": 11}),
                html.Div("🔀 Relay 전환 화살표", style={"color":"#e67e22","fontSize": 11}),
                html.Div("……▶ 위치 보정 방향", style={"color":UAV_ISOLATED,"fontSize":11}),
            ], style={"background": "white", "borderRadius": 8, "padding": 12,
                      "marginBottom": 12, "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

            # ── 건물 설정 ─────────────────────────────────────────────────────
            html.Div([
                html.Div([
                    html.H4("🏢 건물 설정",
                            style={"margin": "0 0 4px", "fontSize": 13, "color": "#2c3e50",
                                   "display": "inline-block"}),
                    html.Span("(시각화 전용)",
                              style={"fontSize": 10, "color": "#999", "marginLeft": 6}),
                ]),
                html.Div("셀을 클릭하면 직접 수정할 수 있어요. 2D·3D 동시 반영",
                         style={"fontSize": 10, "color": "#888", "marginBottom": 5}),
                dash_table.DataTable(
                    id="building-table",
                    columns=[
                        {"name": "ID",      "id": "id",     "editable": False},
                        {"name": "X min",   "id": "x0",     "editable": True, "type": "numeric", "format": {"specifier": ".1f"}},
                        {"name": "X max",   "id": "x1",     "editable": True, "type": "numeric", "format": {"specifier": ".1f"}},
                        {"name": "Y min",   "id": "y0",     "editable": True, "type": "numeric", "format": {"specifier": ".1f"}},
                        {"name": "Y max",   "id": "y1",     "editable": True, "type": "numeric", "format": {"specifier": ".1f"}},
                        {"name": "높이(m)", "id": "height", "editable": True, "type": "numeric", "format": {"specifier": ".0f"}},
                        {"name": "감쇠(dB)","id": "atten",  "editable": True, "type": "numeric", "format": {"specifier": ".1f"}},
                    ],
                    data=BUILDINGS_INIT,
                    editable=True,
                    row_deletable=True,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#2c3e50", "color": "white",
                        "fontWeight": "bold", "fontSize": 11, "padding": "5px 6px",
                    },
                    style_cell={
                        "fontSize": 11, "padding": "4px 5px",
                        "minWidth": 44, "maxWidth": 64,
                        "textAlign": "center",
                    },
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
                        {"if": {"state": "active"},  # 편집 중인 셀 강조
                         "backgroundColor": "#fff3cd",
                         "border": "1px solid #e67e22"},
                    ],
                    tooltip_header={
                        "x0": "건물 서쪽 끝 X 좌표 (m)",
                        "x1": "건물 동쪽 끝 X 좌표 (m)",
                        "y0": "건물 남쪽 끝 Y 좌표 (m)",
                        "y1": "건물 북쪽 끝 Y 좌표 (m)",
                        "height": "건물 높이 — 3D 차폐에 사용",
                        "atten": "전파 감쇠량 (dB)",
                    },
                    tooltip_delay=0,
                    tooltip_duration=None,
                ),
                html.Button("+ 건물 추가", id="add-building-btn", n_clicks=0,
                            style={"marginTop": 6, "padding": "4px 12px",
                                   "background": "#3498db", "color": "white",
                                   "border": "none", "borderRadius": 4,
                                   "cursor": "pointer", "fontSize": 12}),
                html.Button("↺ 초기화", id="reset-building-btn", n_clicks=0,
                            style={"marginTop": 6, "marginLeft": 6,
                                   "padding": "4px 12px",
                                   "background": "#95a5a6", "color": "white",
                                   "border": "none", "borderRadius": 4,
                                   "cursor": "pointer", "fontSize": 12}),
            ], style={"background": "white", "borderRadius": 8, "padding": 12,
                      "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

        ], style={"flex": "1.2", "display": "flex", "flexDirection": "column",
                  "overflowY": "auto", "maxHeight": "calc(100vh - 150px)"}),

    ], style={"display": "flex", "padding": "14px 24px",
              "background": "#f5f6fa", "minHeight": "calc(100vh - 150px)"}),

    ]),  # end Tab 2 (2D 시뮬레이션)


    # ════════════════════════════════════════════════════════════════════════
    # TAB 3: 3D 시뮬레이션
    # ════════════════════════════════════════════════════════════════════════
    dcc.Tab(label="3D 시뮬레이션", value="tab-3d", children=[

    html.Div([
        html.Div([
            html.Label("시나리오 (3D)", style={"fontSize": 12, "fontWeight": "bold"}),
            dcc.Dropdown(
                id="scenario-dd-3d", clearable=False,
                options=[{"label": s, "value": s} for s in SCENARIOS_3D],
                value=SCENARIOS_3D[0] if SCENARIOS_3D else None,
                style={"fontSize": 13, "width": 260},
            ),
        ]),
        html.Div([
            html.Label("재생 속도", style={"fontSize": 12, "fontWeight": "bold"}),
            dcc.Slider(id="speed-slider-3d", min=1, max=5, step=1, value=2,
                       marks={1:"느림", 3:"보통", 5:"빠름"},
                       tooltip={"placement": "bottom"}, updatemode="drag"),
        ], style={"width": 200, "marginLeft": 20}),
        html.Div([
            html.Button("▶ Play", id="play-btn-3d", n_clicks=0,
                        style={"marginRight": 8, "padding": "6px 18px",
                               "background": "#27ae60", "color": "white",
                               "border": "none", "borderRadius": 4,
                               "cursor": "pointer", "fontSize": 14}),
            html.Button("⏸ Pause", id="pause-btn-3d", n_clicks=0,
                        style={"padding": "6px 18px",
                               "background": "#e74c3c", "color": "white",
                               "border": "none", "borderRadius": 4,
                               "cursor": "pointer", "fontSize": 14}),
        ], style={"marginLeft": 20, "alignSelf": "flex-end"}),
        html.Div(id="time-label-3d",
                 style={"marginLeft": 20, "alignSelf": "flex-end",
                        "fontSize": 13, "color": "#555", "minWidth": 140}),
    ], style={"display": "flex", "alignItems": "flex-end", "gap": 0,
              "padding": "14px 24px", "background": "#ecf0f1",
              "borderBottom": "1px solid #ddd"}),

    html.Div([
        dcc.Slider(id="frame-slider-3d", min=0, max=1, step=1, value=0,
                   marks={}, updatemode="drag",
                   tooltip={"placement": "bottom", "always_visible": False}),
    ], style={"padding": "8px 24px", "background": "#ecf0f1",
              "borderBottom": "1px solid #ddd"}),

    html.Div([
        # 3D 지도
        html.Div([
            dcc.Graph(id="map-graph-3d", config={"displayModeBar": True}),
        ], style={"flex": "3", "background": "white", "borderRadius": 8,
                  "padding": 12, "marginRight": 12,
                  "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

        # 3D 사이드 패널
        html.Div([
            html.Div([
                html.H4("링크 상태 (3D)", style={"margin": "0 0 8px", "fontSize": 13,
                                               "color": "#2c3e50"}),
                html.Div(id="state-summary-3d"),
            ], style={"background": "white", "borderRadius": 8, "padding": 12,
                      "marginBottom": 12, "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

            html.Div([
                html.H4("Relay 전환 이력", style={"margin": "0 0 8px", "fontSize": 13,
                                                  "color": "#e67e22"}),
                html.Div(id="relay-log-3d",
                         style={"maxHeight": 130, "overflowY": "auto", "fontSize": 12}),
            ], style={"background": "white", "borderRadius": 8, "padding": 12,
                      "marginBottom": 12, "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

            # 3D 안내
            html.Div([
                html.Div("💡 3D 조작법", style={"fontWeight": "bold", "fontSize": 11,
                                               "marginBottom": 6}),
                html.Div("• 마우스 드래그: 회전", style={"fontSize": 11, "color": "#555"}),
                html.Div("• 스크롤: 확대/축소", style={"fontSize": 11, "color": "#555"}),
                html.Div("• 더블클릭: 뷰 초기화", style={"fontSize": 11, "color": "#555"}),
                html.Div("", style={"height": 8}),
                html.Div("건물 색상 의미", style={"fontWeight": "bold", "fontSize": 11,
                                               "marginBottom": 4}),
                html.Div("━ 회색 박스: 3D 건물 (높이 포함)",
                         style={"fontSize": 11, "color": "#888"}),
                html.Div("", style={"height": 8}),
                html.Div("링크 색상", style={"fontWeight": "bold", "fontSize": 11,
                                           "marginBottom": 4}),
                *[html.Div([
                    html.Span("━━", style={"color": STATE_COLOR[s], "marginRight": 6}),
                    html.Span(s, style={"fontSize": 11}),
                ]) for s in ["healthy", "degraded", "disconnected"]],
            ], style={"background": "white", "borderRadius": 8, "padding": 12,
                      "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

        ], style={"flex": "1.2", "display": "flex", "flexDirection": "column"}),
    ], style={"display": "flex", "padding": "14px 24px",
              "background": "#f5f6fa", "minHeight": "calc(100vh - 150px)"}),

    ] if HAS_3D else [
        html.Div([
            html.Div("🌐 3D 데이터가 없습니다.",
                     style={"fontSize": 18, "color": "#999", "textAlign": "center",
                            "marginTop": 60}),
            html.Div("generate_uav_3d_dataset.py 를 실행하면 3D 탭이 활성화됩니다.",
                     style={"fontSize": 13, "color": "#bbb", "textAlign": "center",
                            "marginTop": 12}),
        ], style={"padding": 40}),
    ]),  # end Tab 2

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3: 성능 지표
    # ════════════════════════════════════════════════════════════════════════
    dcc.Tab(label="성능 지표", value="tab-perf", children=[
        html.Div([

            # 행 1: 모델 비교 + 학습 곡선
            html.Div([
                # 모델 성능 비교
                html.Div([
                    html.H4("모델 성능 비교 (Test Set)",
                            style={"margin": "0 0 8px", "fontSize": 14, "color": "#2c3e50"}),
                    dcc.Graph(id="model-compare-graph",
                              config={"displayModeBar": False},
                              figure=_make_model_compare()),
                ], style={**CARD, "flex": 1, "marginRight": 12}),

                # 학습 곡선
                html.Div([
                    html.H4("학습 곡선 (Val Accuracy)",
                            style={"margin": "0 0 8px", "fontSize": 14, "color": "#2c3e50"}),
                    dcc.Graph(id="train-curve-graph",
                              config={"displayModeBar": False},
                              figure=_make_train_curve()),
                ], style={**CARD, "flex": 1}),
            ], style={"display": "flex", "marginBottom": 0}),

            # 행 2: 멀티홉 + 온라인 학습 + 위치 보정
            html.Div([
                # 멀티홉
                html.Div([
                    html.H4("멀티홉 Relay 연결 성공률",
                            style={"margin": "0 0 8px", "fontSize": 14, "color": "#2c3e50"}),
                    dcc.Graph(id="multihop-graph",
                              config={"displayModeBar": False},
                              figure=_make_multihop()),
                ], style={**CARD, "flex": 2, "marginRight": 12}),

                # 온라인 학습 + 위치 보정 (우측 2개)
                html.Div([
                    html.Div([
                        html.H4("온라인 학습 적응 효과",
                                style={"margin": "0 0 8px", "fontSize": 14, "color": "#2c3e50"}),
                        dcc.Graph(id="online-graph",
                                  config={"displayModeBar": False},
                                  figure=_make_online()),
                    ], style={**CARD}),

                    html.Div([
                        html.H4("위치 보정 성공률 개선",
                                style={"margin": "0 0 8px", "fontSize": 14, "color": "#2c3e50"}),
                        dcc.Graph(id="correction-graph",
                                  config={"displayModeBar": False},
                                  figure=_make_correction()),
                    ], style={**CARD, "marginBottom": 0}),
                ], style={"flex": 1, "display": "flex", "flexDirection": "column"}),
            ], style={"display": "flex"}),

            # 행 3: DQN 위치 보정 성능
            html.Div([
                html.Div([
                    html.H4("DQN 위치 보정 성능 (강화학습 vs 기존)",
                            style={"margin": "0 0 8px", "fontSize": 14,
                                   "color": "#2c3e50"}),
                    html.Div("성공률 (막대) · 평균 이동 스텝 수 (선, 우측 축)",
                             style={"fontSize": 11, "color": "#888", "marginBottom": 4}),
                    dcc.Graph(id="rl-correction-graph",
                              config={"displayModeBar": False},
                              figure=_make_rl_correction()),
                ], style={**CARD, "flex": 1, "marginRight": 12}),

                html.Div([
                    html.H4("DQN 학습 곡선 (에피소드 보상)",
                            style={"margin": "0 0 8px", "fontSize": 14,
                                   "color": "#2c3e50"}),
                    html.Div("회색=에피소드 보상, 파랑=이동 평균",
                             style={"fontSize": 11, "color": "#888", "marginBottom": 4}),
                    dcc.Graph(id="rl-history-graph",
                              config={"displayModeBar": False},
                              figure=_make_rl_history()),
                ], style={**CARD, "flex": 1, "marginBottom": 0}),
            ], style={"display": "flex"}),

        ], style={"padding": "16px 24px", "background": "#f5f6fa",
                  "minHeight": "calc(100vh - 120px)"}),
    ]),  # end 성능 지표 Tab

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5: NS-3 실시간 연동
    # ════════════════════════════════════════════════════════════════════════
    dcc.Tab(label="NS-3 실시간 연동", value="tab-ns3", children=[
        html.Div([

            # 상단 상태 카드
            html.Div(id="ns3-status-card", style={"marginBottom": 16}),

            # 행 1: 릴레이 타임라인 + 지연 시간
            html.Div([
                html.Div([
                    html.H4("릴레이 노드 선택 타임라인",
                            style={"margin": "0 0 8px", "fontSize": 14, "color": "#2c3e50"}),
                    dcc.Graph(id="ns3-relay-graph",
                              config={"displayModeBar": False},
                              style={"height": 260}),
                ], style={**CARD, "flex": 1, "marginRight": 12}),

                html.Div([
                    html.H4("ML 서버 응답 지연 (ms)",
                            style={"margin": "0 0 8px", "fontSize": 14, "color": "#2c3e50"}),
                    dcc.Graph(id="ns3-latency-graph",
                              config={"displayModeBar": False},
                              style={"height": 260}),
                ], style={**CARD, "flex": 1}),
            ], style={"display": "flex", "marginBottom": 16}),

            # 행 2: 위치 보정 이력
            html.Div([
                html.H4("DQN 위치 보정 이력",
                        style={"margin": "0 0 8px", "fontSize": 14, "color": "#2c3e50"}),
                dcc.Graph(id="ns3-correction-graph",
                          config={"displayModeBar": False},
                          style={"height": 240}),
            ], style={**CARD, "marginBottom": 0}),

        ], style={"padding": "16px 24px", "background": "#f5f6fa",
                  "minHeight": "calc(100vh - 120px)"}),
    ]),  # end NS-3 Tab

    ]),  # end Tabs

], style={"fontFamily": "sans-serif"})


# ── 콜백 ─────────────────────────────────────────────────────────────────────

@callback(
    Output("frame-slider", "max"),
    Output("frame-slider", "marks"),
    Output("frame-slider", "value"),
    Output("frame-store",  "data"),
    Output("prev-relay-store", "data"),
    Output("bad-streak-store", "data"),
    Input("scenario-dd", "value"),
)
def reset_on_scenario(scenario):
    ts = get_ts(scenario)
    n  = len(ts) - 1
    step = max(1, n // 12)
    marks = {i: f"{float(ts[i]):.0f}s" for i in range(0, n+1, step)}
    return n, marks, 0, 0, None, 0


@callback(
    Output("interval", "disabled"),
    Output("interval", "interval"),
    Input("play-btn",    "n_clicks"),
    Input("pause-btn",   "n_clicks"),
    Input("speed-slider","value"),
    State("interval",    "disabled"),
)
def toggle_play(play_n, pause_n, speed, is_disabled):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, 600
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    interval_ms = max(100, 600 - (speed - 1) * 120)
    if btn == "play-btn":
        return False, interval_ms
    if btn == "pause-btn":
        return True, interval_ms
    return is_disabled, interval_ms


@callback(
    Output("frame-store",      "data",    allow_duplicate=True),
    Output("frame-slider",     "value",   allow_duplicate=True),
    Output("prev-relay-store", "data",    allow_duplicate=True),
    Output("bad-streak-store", "data",    allow_duplicate=True),
    Input("interval",          "n_intervals"),
    State("frame-store",       "data"),
    State("scenario-dd",       "value"),
    State("prev-relay-store",  "data"),
    State("bad-streak-store",  "data"),
    prevent_initial_call=True,
)
def advance_frame(n, frame_idx, scenario, prev_relay, bad_streak):
    ts  = get_ts(scenario)
    nxt = (frame_idx + 1) % len(ts)
    t_s = ts[nxt]

    lnks = LINKS.get(scenario, {}).get(t_s, {})
    relay_vals = [v["relay"] for v in lnks.values()]
    cur_relay  = max(set(relay_vals), key=relay_vals.count) if relay_vals else prev_relay

    any_disc = any(v["state"] == "disconnected" for v in lnks.values())
    new_streak = (bad_streak + 1) if any_disc else 0

    return nxt, nxt, cur_relay, new_streak


@callback(
    Output("frame-store",      "data",    allow_duplicate=True),
    Output("prev-relay-store", "data",    allow_duplicate=True),
    Output("bad-streak-store", "data",    allow_duplicate=True),
    Input("frame-slider",      "value"),
    State("scenario-dd",       "value"),
    prevent_initial_call=True,
)
def slider_moved(slider_val, scenario):
    ts  = get_ts(scenario)
    t_s = ts[min(slider_val, len(ts)-1)]
    lnks = LINKS.get(scenario, {}).get(t_s, {})
    relay_vals = [v["relay"] for v in lnks.values()]
    cur_relay  = max(set(relay_vals), key=relay_vals.count) if relay_vals else None
    return slider_val, cur_relay, 0


@callback(
    Output("map-graph",       "figure"),
    Output("time-label",      "children"),
    Output("state-summary",   "children"),
    Output("relay-log",       "children"),
    Output("correction-log",  "children"),
    Input("frame-store",      "data"),
    Input("scenario-dd",      "value"),
    Input("building-table",   "data"),
    State("prev-relay-store", "data"),
    State("bad-streak-store", "data"),
)
def update_view(frame_idx, scenario, buildings, prev_relay, bad_streak):
    ts  = get_ts(scenario)
    idx = min(frame_idx, len(ts) - 1)
    t_s = ts[idx]

    fig = make_figure(scenario, t_s, prev_relay, bad_streak, buildings=buildings)

    time_label = f"t = {float(t_s):.2f}s  ({idx+1}/{len(ts)})"

    # 링크 상태 요약
    lnks = LINKS.get(scenario, {}).get(t_s, {})
    cnt  = {"healthy": 0, "degraded": 0, "disconnected": 0}
    for v in lnks.values():
        if v["state"] in cnt: cnt[v["state"]] += 1
    state_summary = [
        html.Div([
            html.Span("●", style={"color": STATE_COLOR[s], "fontSize": 18,
                                   "marginRight": 6}),
            html.Span(f"{s}: {cnt[s]}건",
                      style={"fontSize": 12}),
        ], style={"marginBottom": 4})
        for s in ["healthy", "degraded", "disconnected"]
    ]

    # Relay 전환 이력 (전체 시나리오 순회)
    relay_events = []
    prev_r = None
    for t in ts[:idx+1]:
        lk = LINKS.get(scenario, {}).get(t, {})
        rv = [v["relay"] for v in lk.values()]
        cur = max(set(rv), key=rv.count) if rv else None
        if prev_r is not None and cur != prev_r:
            relay_events.append(
                html.Div(f"t={float(t):.1f}s  UAV{prev_r}→UAV{cur}",
                         style={"color": "#e67e22", "borderBottom": "1px solid #fde",
                                "padding": "2px 0"}))
        prev_r = cur
    relay_log = relay_events[-10:] if relay_events else [
        html.Span("전환 없음", style={"color": "#999", "fontSize": 11})]

    # 위치 보정 이력
    corr_events = []
    streak = 0
    for t in ts[:idx+1]:
        lk = LINKS.get(scenario, {}).get(t, {})
        any_disc = any(v["state"] == "disconnected" for v in lk.values())
        streak = (streak + 1) if any_disc else 0
        pos_t = POSITIONS.get(scenario, {}).get(t, {})
        comps = _components(pos_t, lk)
        if len(comps) > 1 and streak >= HYSTERESIS:
            n_iso = sum(len(c) for c in comps if c != max(comps, key=len))
            corr_events.append(
                html.Div(f"t={float(t):.1f}s  격리 {n_iso}대 보정 발동",
                         style={"color": "#8e44ad", "borderBottom": "1px solid #ede",
                                "padding": "2px 0"}))
    correction_log = corr_events[-10:] if corr_events else [
        html.Span("보정 없음", style={"color": "#999", "fontSize": 11})]

    return fig, time_label, state_summary, relay_log, correction_log


# ── 건물 설정 콜백 ────────────────────────────────────────────────────────────

@callback(
    Output("building-table", "data"),
    Input("add-building-btn",   "n_clicks"),
    Input("reset-building-btn", "n_clicks"),
    State("building-table",     "data"),
    prevent_initial_call=True,
)
def manage_buildings(add_n, reset_n, current_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "reset-building-btn":
        return BUILDINGS_INIT
    # add-building-btn
    new_id = f"B{len(current_data)}"
    new_row = {"id": new_id, "x0": 100.0, "x1": 120.0,
               "y0": 55.0, "y1": 65.0, "height": 30.0, "atten": 6.0}
    return (current_data or []) + [new_row]


# ── 실제 환경 탭 콜백 ────────────────────────────────────────────────────────

@callback(
    Output("frame-slider-real", "max"),
    Output("frame-slider-real", "marks"),
    Output("frame-slider-real", "value"),
    Output("frame-store-real",  "data"),
    Output("prev-relay-store-real", "data"),
    Output("bad-streak-store-real", "data"),
    Input("scenario-dd-real", "value"),
)
def reset_on_scenario_real(scenario):
    ts = get_ts(scenario)
    n  = len(ts) - 1
    step = max(1, n // 12)
    marks = {i: f"{float(ts[i]):.0f}s" for i in range(0, n+1, step)}
    return n, marks, 0, 0, None, 0


@callback(
    Output("interval-real", "disabled"),
    Output("interval-real", "interval"),
    Input("play-btn-real",    "n_clicks"),
    Input("pause-btn-real",   "n_clicks"),
    Input("speed-slider-real","value"),
    State("interval-real",    "disabled"),
)
def toggle_play_real(play_n, pause_n, speed, is_disabled):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, 600
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    interval_ms = max(100, 600 - (speed - 1) * 120)
    if btn == "play-btn-real":
        return False, interval_ms
    if btn == "pause-btn-real":
        return True, interval_ms
    return is_disabled, interval_ms


@callback(
    Output("frame-store-real",      "data",    allow_duplicate=True),
    Output("frame-slider-real",     "value",   allow_duplicate=True),
    Output("prev-relay-store-real", "data",    allow_duplicate=True),
    Output("bad-streak-store-real", "data",    allow_duplicate=True),
    Input("interval-real",          "n_intervals"),
    State("frame-store-real",       "data"),
    State("scenario-dd-real",       "value"),
    State("prev-relay-store-real",  "data"),
    State("bad-streak-store-real",  "data"),
    prevent_initial_call=True,
)
def advance_frame_real(n, frame_idx, scenario, prev_relay, bad_streak):
    ts  = get_ts(scenario)
    nxt = (frame_idx + 1) % len(ts)
    t_s = ts[nxt]
    lnks = LINKS.get(scenario, {}).get(t_s, {})
    relay_vals = [v["relay"] for v in lnks.values()]
    cur_relay  = max(set(relay_vals), key=relay_vals.count) if relay_vals else prev_relay
    any_disc   = any(v["state"] == "disconnected" for v in lnks.values())
    new_streak = (bad_streak + 1) if any_disc else 0
    return nxt, nxt, cur_relay, new_streak


@callback(
    Output("frame-store-real",      "data",    allow_duplicate=True),
    Output("prev-relay-store-real", "data",    allow_duplicate=True),
    Output("bad-streak-store-real", "data",    allow_duplicate=True),
    Input("frame-slider-real",      "value"),
    State("scenario-dd-real",       "value"),
    prevent_initial_call=True,
)
def slider_moved_real(slider_val, scenario):
    ts  = get_ts(scenario)
    t_s = ts[min(slider_val, len(ts)-1)]
    lnks = LINKS.get(scenario, {}).get(t_s, {})
    relay_vals = [v["relay"] for v in lnks.values()]
    cur_relay  = max(set(relay_vals), key=relay_vals.count) if relay_vals else None
    return slider_val, cur_relay, 0


@callback(
    Output("map-graph-real",       "figure"),
    Output("time-label-real",      "children"),
    Output("state-summary-real",   "children"),
    Output("relay-log-real",       "children"),
    Output("correction-log-real",  "children"),
    Output("multihop-result-real", "children"),
    Input("frame-store-real",      "data"),
    Input("scenario-dd-real",      "value"),
    State("prev-relay-store-real", "data"),
    State("bad-streak-store-real", "data"),
)
def update_view_real(frame_idx, scenario, prev_relay, bad_streak):
    ts  = get_ts(scenario)
    idx = min(frame_idx, len(ts) - 1)
    t_s = ts[idx]

    fig = make_figure_real(scenario, t_s, prev_relay, bad_streak)
    time_label = f"t = {float(t_s):.2f}s  ({idx+1}/{len(ts)})"

    lnks = LINKS.get(scenario, {}).get(t_s, {})
    cnt  = {"healthy": 0, "degraded": 0, "disconnected": 0}
    for v in lnks.values():
        if v["state"] in cnt:
            cnt[v["state"]] += 1
    state_summary = [
        html.Div([
            html.Span("●", style={"color": STATE_COLOR[s], "fontSize": 18,
                                   "marginRight": 6}),
            html.Span(f"{s}: {cnt[s]}건", style={"fontSize": 12}),
        ], style={"marginBottom": 4})
        for s in ["healthy", "degraded", "disconnected"]
    ]

    relay_events = []
    prev_r = None
    for t in ts[:idx+1]:
        lk = LINKS.get(scenario, {}).get(t, {})
        rv = [v["relay"] for v in lk.values()]
        cur = max(set(rv), key=rv.count) if rv else None
        if prev_r is not None and cur != prev_r:
            relay_events.append(
                html.Div(f"t={float(t):.1f}s  UAV{prev_r}→UAV{cur}",
                         style={"color": "#e67e22", "borderBottom": "1px solid #fde",
                                "padding": "2px 0"}))
        prev_r = cur
    relay_log = relay_events[-10:] if relay_events else [
        html.Span("전환 없음", style={"color": "#999", "fontSize": 11})]

    corr_events = []
    streak = 0
    for t in ts[:idx+1]:
        lk = LINKS.get(scenario, {}).get(t, {})
        any_disc = any(v["state"] == "disconnected" for v in lk.values())
        streak = (streak + 1) if any_disc else 0
        pos_t = POSITIONS.get(scenario, {}).get(t, {})
        comps = _components(pos_t, lk)
        if len(comps) > 1 and streak >= HYSTERESIS:
            n_iso = sum(len(c) for c in comps if c != max(comps, key=len))
            corr_events.append(
                html.Div(f"t={float(t):.1f}s  격리 {n_iso}대 보정 발동",
                         style={"color": "#8e44ad", "borderBottom": "1px solid #ede",
                                "padding": "2px 0"}))
    correction_log = corr_events[-10:] if corr_events else [
        html.Span("보정 없음", style={"color": "#999", "fontSize": 11})]

    # ── 멀티홉 알고리즘 결과 카드 ────────────────────────────────────────────
    mh = _multihop_connectivity(lnks)
    if mh["all_multihop"]:
        if mh["all_direct"]:
            bg, icon, title = "#27ae60", "✅", "직접 연결 성공"
            body = "모든 UAV 쌍이 직접 링크로 연결됩니다."
        else:
            bg, icon, title = "#2980b9", "🔗", "멀티홉 알고리즘 성공!"
            gain = mh.get("multihop_gain", 0)
            body = (f"직접 연결이 불가능한 {gain}개 쌍을 멀티홉 중계로 복구했습니다. "
                    f"전체 {mh['n_uavs']}대 UAV 통신망 유지 중.")
    else:
        bg, icon, title = "#e74c3c", "❌", "연결 실패"
        body = "멀티홉으로도 일부 UAV 간 경로를 찾지 못했습니다. 위치 보정 발동 대기 중."

    multihop_card = html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": 18, "marginRight": 8}),
            html.Span(title, style={"fontWeight": "bold", "fontSize": 13}),
        ], style={"marginBottom": 6}),
        html.Div(body, style={"fontSize": 11, "lineHeight": "1.5"}),
    ], style={
        "background": bg, "color": "white",
        "borderRadius": 8, "padding": "10px 14px",
        "boxShadow": "0 2px 6px rgba(0,0,0,.2)",
    })

    return fig, time_label, state_summary, relay_log, correction_log, multihop_card


# ── 3D 탭 콜백 ───────────────────────────────────────────────────────────────

if HAS_3D:
    def get_ts_3d(scenario: str) -> list[str]:
        return sorted(POSITIONS_3D.get(scenario, {}).keys(), key=float)

    @callback(
        Output("frame-slider-3d", "max"),
        Output("frame-slider-3d", "marks"),
        Output("frame-slider-3d", "value"),
        Output("frame-store-3d",  "data"),
        Output("prev-relay-store-3d", "data"),
        Input("scenario-dd-3d",   "value"),
    )
    def reset_on_scenario_3d(scenario):
        ts = get_ts_3d(scenario)
        n  = len(ts) - 1
        step = max(1, n // 12)
        marks = {i: f"{float(ts[i]):.0f}s" for i in range(0, n+1, step)}
        return n, marks, 0, 0, None

    @callback(
        Output("interval-3d", "disabled"),
        Output("interval-3d", "interval"),
        Input("play-btn-3d",    "n_clicks"),
        Input("pause-btn-3d",   "n_clicks"),
        Input("speed-slider-3d","value"),
        State("interval-3d",    "disabled"),
    )
    def toggle_play_3d(play_n, pause_n, speed, is_disabled):
        ctx = dash.callback_context
        if not ctx.triggered:
            return True, 600
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        interval_ms = max(100, 600 - (speed - 1) * 120)
        if btn == "play-btn-3d":
            return False, interval_ms
        if btn == "pause-btn-3d":
            return True, interval_ms
        return is_disabled, interval_ms

    @callback(
        Output("frame-store-3d",      "data",    allow_duplicate=True),
        Output("frame-slider-3d",     "value",   allow_duplicate=True),
        Output("prev-relay-store-3d", "data",    allow_duplicate=True),
        Input("interval-3d",          "n_intervals"),
        State("frame-store-3d",       "data"),
        State("scenario-dd-3d",       "value"),
        State("prev-relay-store-3d",  "data"),
        prevent_initial_call=True,
    )
    def advance_frame_3d(n, frame_idx, scenario, prev_relay):
        ts  = get_ts_3d(scenario)
        nxt = (frame_idx + 1) % len(ts)
        t_s = ts[nxt]
        lnks = LINKS_3D.get(scenario, {}).get(t_s, {})
        relay_vals = [v["relay"] for v in lnks.values()]
        cur_relay  = max(set(relay_vals), key=relay_vals.count) if relay_vals else prev_relay
        return nxt, nxt, cur_relay

    @callback(
        Output("frame-store-3d",      "data",    allow_duplicate=True),
        Output("prev-relay-store-3d", "data",    allow_duplicate=True),
        Input("frame-slider-3d",      "value"),
        State("scenario-dd-3d",       "value"),
        prevent_initial_call=True,
    )
    def slider_moved_3d(slider_val, scenario):
        ts  = get_ts_3d(scenario)
        t_s = ts[min(slider_val, len(ts)-1)]
        lnks = LINKS_3D.get(scenario, {}).get(t_s, {})
        relay_vals = [v["relay"] for v in lnks.values()]
        cur_relay  = max(set(relay_vals), key=relay_vals.count) if relay_vals else None
        return slider_val, cur_relay

    @callback(
        Output("map-graph-3d",       "figure"),
        Output("time-label-3d",      "children"),
        Output("state-summary-3d",   "children"),
        Output("relay-log-3d",       "children"),
        Input("frame-store-3d",      "data"),
        Input("scenario-dd-3d",      "value"),
        Input("building-table",      "data"),   # Tab1 건물 변경 시 3D도 갱신
        State("prev-relay-store-3d", "data"),
    )
    def update_view_3d(frame_idx, scenario, buildings, prev_relay):
        ts  = get_ts_3d(scenario)
        idx = min(frame_idx, len(ts) - 1)
        t_s = ts[idx]

        fig = make_figure_3d(scenario, t_s, prev_relay, buildings=buildings)

        time_label = f"t = {float(t_s):.2f}s  ({idx+1}/{len(ts)})"

        lnks = LINKS_3D.get(scenario, {}).get(t_s, {})
        cnt  = {"healthy": 0, "degraded": 0, "disconnected": 0}
        for v in lnks.values():
            if v["state"] in cnt:
                cnt[v["state"]] += 1
        state_summary = [
            html.Div([
                html.Span("●", style={"color": STATE_COLOR[s], "fontSize": 18,
                                       "marginRight": 6}),
                html.Span(f"{s}: {cnt[s]}건", style={"fontSize": 12}),
            ], style={"marginBottom": 4})
            for s in ["healthy", "degraded", "disconnected"]
        ]

        relay_events = []
        prev_r = None
        for t in ts[:idx+1]:
            lk = LINKS_3D.get(scenario, {}).get(t, {})
            rv = [v["relay"] for v in lk.values()]
            cur = max(set(rv), key=rv.count) if rv else None
            if prev_r is not None and cur != prev_r:
                relay_events.append(
                    html.Div(f"t={float(t):.1f}s  UAV{prev_r}→UAV{cur}",
                             style={"color": "#e67e22",
                                    "borderBottom": "1px solid #fde",
                                    "padding": "2px 0"}))
            prev_r = cur
        relay_log = relay_events[-8:] if relay_events else [
            html.Span("전환 없음", style={"color": "#999", "fontSize": 11})]

        return fig, time_label, state_summary, relay_log


# ── NS-3 실시간 연동 콜백 ─────────────────────────────────────────────────────
_NS3_ML_CSV = ROOT / "ns-3.47" / "uav-ml.csv"


def _read_ns3_log() -> list[dict]:
    """uav-ml.csv를 읽어 레코드 리스트 반환.
    response 필드에 개행이 포함된 경우도 안전하게 처리.
    """
    rows = []
    if not _NS3_ML_CSV.exists():
        return rows
    try:
        with open(_NS3_ML_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rid = row["relay_id"].strip()
                    relay = int(rid) if rid not in ("-1", "offline", "") else -1
                    # response 필드: 개행·따옴표 제거 후 JSON 파싱 시도
                    raw_resp = row.get("response", "").strip().strip('"').strip()
                    rows.append({
                        "t":     float(row["time_s"]),
                        "lat":   float(row["latency_ms"]),
                        "relay": relay,
                        "resp":  raw_resp,
                    })
                except (ValueError, KeyError):
                    pass
    except Exception:
        pass
    return rows


@callback(
    Output("ns3-status-card",     "children"),
    Output("ns3-relay-graph",     "figure"),
    Output("ns3-latency-graph",   "figure"),
    Output("ns3-correction-graph","figure"),
    Input("ns3-interval",         "n_intervals"),
)
def update_ns3_tab(_):
    rows = _read_ns3_log()

    # ── 상태 카드 ──────────────────────────────────────────────────────────
    if not rows:
        card = html.Div(
            "uav-ml.csv 없음 — NS-3와 ML 서버를 함께 실행하면 데이터가 표시됩니다.",
            style={"background": "#fff3cd", "border": "1px solid #ffc107",
                   "borderRadius": 6, "padding": "10px 16px",
                   "color": "#856404", "fontSize": 13},
        )
        empty = go.Figure().update_layout(
            paper_bgcolor="#f9f9f9", plot_bgcolor="#f9f9f9",
            margin=dict(l=30, r=20, t=20, b=30),
            annotations=[dict(text="데이터 없음", showarrow=False,
                              font=dict(size=14, color="#aaa"),
                              xref="paper", yref="paper", x=0.5, y=0.5)],
        )
        return card, empty, empty, empty

    online = [r for r in rows if r["relay"] >= 0]
    corr_rows = []
    for r in online:
        try:
            d = json.loads(r["resp"])
            if d.get("correction"):
                corr_rows.append({**r, **d["correction"]})
        except Exception:
            pass

    total   = len(online)
    avg_lat = sum(r["lat"] for r in online) / total if total else 0
    max_lat = max((r["lat"] for r in online), default=0)
    n_corr  = len(corr_rows)

    stat_style = {"background": "#fff", "borderRadius": 8,
                  "padding": "10px 20px", "boxShadow": "0 1px 4px rgba(0,0,0,.1)",
                  "textAlign": "center", "flex": 1, "marginRight": 12}
    card = html.Div([
        html.Div([html.Div(str(total), style={"fontSize": 26, "fontWeight": "bold", "color": "#2980b9"}),
                  html.Div("총 요청 수", style={"fontSize": 12, "color": "#7f8c8d"})], style=stat_style),
        html.Div([html.Div(f"{avg_lat:.2f} ms", style={"fontSize": 26, "fontWeight": "bold", "color": "#27ae60"}),
                  html.Div("평균 지연", style={"fontSize": 12, "color": "#7f8c8d"})], style=stat_style),
        html.Div([html.Div(f"{max_lat:.2f} ms", style={"fontSize": 26, "fontWeight": "bold", "color": "#e67e22"}),
                  html.Div("최대 지연", style={"fontSize": 12, "color": "#7f8c8d"})], style=stat_style),
        html.Div([html.Div(str(n_corr), style={"fontSize": 26, "fontWeight": "bold", "color": "#8e44ad"}),
                  html.Div("DQN 위치 보정 횟수", style={"fontSize": 12, "color": "#7f8c8d"})],
                 style={**stat_style, "marginRight": 0}),
    ], style={"display": "flex"})

    ts     = [r["t"]    for r in online]
    relays = [r["relay"] for r in online]
    lats   = [r["lat"]   for r in online]

    # ── 릴레이 타임라인 ────────────────────────────────────────────────────
    relay_fig = go.Figure()
    relay_fig.add_trace(go.Scatter(
        x=ts, y=relays, mode="lines+markers",
        line=dict(color="#2980b9", width=2),
        marker=dict(size=7),
        name="릴레이 ID",
    ))
    relay_fig.update_layout(
        margin=dict(l=40, r=20, t=10, b=30),
        xaxis=dict(title="시뮬레이션 시간 (s)", gridcolor="#eee"),
        yaxis=dict(title="릴레이 UAV ID", dtick=1,
                   tickvals=list(range(5)),
                   ticktext=[f"UAV{i}" for i in range(5)],
                   gridcolor="#eee"),
        plot_bgcolor="white", paper_bgcolor="white",
    )

    # ── 지연 시간 ──────────────────────────────────────────────────────────
    lat_fig = go.Figure()
    lat_fig.add_trace(go.Scatter(
        x=ts, y=lats, mode="lines+markers",
        line=dict(color="#27ae60", width=2),
        marker=dict(size=6),
        name="지연 (ms)",
        fill="tozeroy", fillcolor="rgba(39,174,96,0.08)",
    ))
    lat_fig.add_hline(y=500, line_dash="dot", line_color="#e74c3c",
                      annotation_text="목표 500ms", annotation_position="top left")
    lat_fig.update_layout(
        margin=dict(l=40, r=20, t=10, b=30),
        xaxis=dict(title="시뮬레이션 시간 (s)", gridcolor="#eee"),
        yaxis=dict(title="응답 지연 (ms)", gridcolor="#eee"),
        plot_bgcolor="white", paper_bgcolor="white",
    )

    # ── 위치 보정 이력 ─────────────────────────────────────────────────────
    corr_fig = go.Figure()
    if corr_rows:
        corr_ts  = [r["t"] for r in corr_rows]
        corr_uid = [r["uav_id"] for r in corr_rows]
        corr_dx  = [r["dx"] for r in corr_rows]
        corr_dy  = [r["dy"] for r in corr_rows]
        corr_mag = [math.hypot(dx, dy) for dx, dy in zip(corr_dx, corr_dy)]

        corr_fig.add_trace(go.Bar(
            x=corr_ts, y=corr_mag,
            marker_color=[f"hsl({uid * 60}, 70%, 55%)" for uid in corr_uid],
            text=[f"UAV{uid}<br>({dx:.1f},{dy:.1f})"
                  for uid, dx, dy in zip(corr_uid, corr_dx, corr_dy)],
            textposition="auto",
            name="보정 크기 (m)",
        ))
        corr_fig.update_layout(
            margin=dict(l=40, r=20, t=10, b=30),
            xaxis=dict(title="시뮬레이션 시간 (s)", gridcolor="#eee"),
            yaxis=dict(title="이동 거리 (m)", gridcolor="#eee"),
            plot_bgcolor="white", paper_bgcolor="white",
        )
    else:
        corr_fig.update_layout(
            margin=dict(l=40, r=20, t=10, b=30),
            paper_bgcolor="white", plot_bgcolor="white",
            annotations=[dict(text="보정 없음 (모든 링크 정상)", showarrow=False,
                              font=dict(size=13, color="#27ae60"),
                              xref="paper", yref="paper", x=0.5, y=0.5)],
        )

    return card, relay_fig, lat_fig, corr_fig


if __name__ == "__main__":
    print("대시보드 시작: http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
