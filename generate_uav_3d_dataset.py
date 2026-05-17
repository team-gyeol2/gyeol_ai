#!/usr/bin/env python3
"""
generate_uav_3d_dataset.py
──────────────────────────
3D (고도 포함) UAV 데이터셋 생성

출력: ns-3.47/datasets/uav_3d/
  obstacles_3d.csv     — building_id, x_min_m, x_max_m, y_min_m, y_max_m, height_m, attenuation_db
  uav_positions_3d.csv — scenario_id, time_s, uav_id, x_m, y_m, z_m, speed_mps, role
  link_metrics_3d.csv  — scenario_id, time_s, src_uav, dst_uav, dist3d_m, dist2d_m,
                          rssi_dbm_est, snr_db_est, plr_pct_est, throughput_mbps_est,
                          link_state, link_state_int, hop_count,
                          blocked_building_count, optimal_relay_uav

3D 시나리오 (10개):
  altitude_step        — 3단계 고도층 비행 (10/30/50 m)
  altitude_climb       — 점진적 상승 중 코리도 통과
  vertical_relay       — 수직 방향 중계 체인
  formation_3d         — 3D 피라미드 편대
  altitude_oscillation — 정현파 고도 변화 + 수평 이동
  dive_and_rise        — 급강하 후 상승 (2페이즈)
  layered_corridor     — 2층 고도 교행 (건물 차폐 차이)
  spiral_ascent        — 나선형 상승 궤적
  hover_relay          — 중계 UAV 고도 고정, 나머지 수평 분산
  altitude_avoid       — 초기 저고도 → 건물 회피 상승

실행:
    python3 generate_uav_3d_dataset.py
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "ns-3.47" / "datasets" / "uav_3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_UAVS = 5

# AR(1) 채널 노이즈 파라미터
RSSI_NOISE_STD = 8.0
NOISE_CORR     = 0.88
PLR_NOISE_STD  = 7.0
PLR_NOISE_CORR = 0.85

THRESHOLDS = {
    "healthy_rssi_dbm_min":       -78.0,
    "degraded_rssi_dbm_min":      -85.0,
    "healthy_plr_pct_max":          5.0,
    "degraded_plr_pct_max":        20.0,
    "healthy_throughput_mbps_min":  8.0,
    "degraded_throughput_mbps_min": 3.0,
}

# 3D 건물 (height_m 추가)
BUILDINGS_3D = [
    {
        "building_id": "B0",
        "x_min_m": 88.0, "x_max_m": 100.0,
        "y_min_m": 52.0, "y_max_m": 58.0,
        "height_m": 30.0, "attenuation_db": 6.0,
        "note": "Mid-map lower corridor obstacle; blocks UAVs below 30m altitude.",
    },
    {
        "building_id": "B1",
        "x_min_m": 118.0, "x_max_m": 132.0,
        "y_min_m": 62.0, "y_max_m": 68.0,
        "height_m": 40.0, "attenuation_db": 6.0,
        "note": "Upper corridor obstacle; tallest building, blocks UAVs below 40m.",
    },
    {
        "building_id": "B2",
        "x_min_m": 146.0, "x_max_m": 160.0,
        "y_min_m": 50.0, "y_max_m": 56.0,
        "height_m": 25.0, "attenuation_db": 6.0,
        "note": "Late-stage lower corridor; blocks UAVs below 25m altitude.",
    },
]


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def _ar1_noise(n: int, std: float, rho: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    innov_std = std * math.sqrt(1.0 - rho ** 2)
    noise = [rng.gauss(0, std)]
    for _ in range(n - 1):
        noise.append(rho * noise[-1] + rng.gauss(0, innov_std))
    return noise


def _seg_box_3d(x0, y0, z0, x1, y1, z1,
                bx0, bx1, by0, by1, bz0, bz1) -> bool:
    """Check if segment P0→P1 intersects axis-aligned 3D box using slab method."""
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    t_min, t_max = 0.0, 1.0

    for d, p, lo, hi in [
        (dx, x0, bx0, bx1),
        (dy, y0, by0, by1),
        (dz, z0, bz0, bz1),
    ]:
        if abs(d) < 1e-9:
            if p < lo or p > hi:
                return False
        else:
            t1 = (lo - p) / d
            t2 = (hi - p) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False
    return True


def _compute_link(x0, y0, z0, x1, y1, z1, buildings,
                  noise_rssi: float, noise_plr: float) -> dict:
    """Compute all link metrics for a UAV pair in 3D space."""
    d3d = math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
    d2d = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    d   = max(d3d, 1.0)

    rssi_base = -46.0 - 20.0 * math.log10(d)
    rssi = rssi_base + noise_rssi

    # Building attenuation via 3D box intersection
    blocked = 0
    atten   = 0.0
    for b in buildings:
        if _seg_box_3d(x0, y0, z0, x1, y1, z1,
                       b["x_min_m"], b["x_max_m"],
                       b["y_min_m"], b["y_max_m"],
                       0.0, b["height_m"]):
            blocked += 1
            atten   += b["attenuation_db"]
    rssi -= atten

    snr = rssi + 105.0  # assumed noise floor -105 dBm

    # PLR model
    if rssi >= THRESHOLDS["healthy_rssi_dbm_min"]:
        plr_base = 1.0
    elif rssi >= THRESHOLDS["degraded_rssi_dbm_min"]:
        plr_base = 1.0 + (THRESHOLDS["healthy_rssi_dbm_min"] - rssi) / 7.0 * 14.0
    else:
        plr_base = 20.0 + (-THRESHOLDS["degraded_rssi_dbm_min"] + rssi) * (-5.0)
    plr = min(100.0, max(0.0, plr_base + noise_plr))

    # Throughput model
    if rssi >= THRESHOLDS["healthy_rssi_dbm_min"]:
        tp = max(THRESHOLDS["healthy_throughput_mbps_min"],
                 20.0 + 0.1 * (rssi + 78.0))
    elif rssi >= THRESHOLDS["degraded_rssi_dbm_min"]:
        tp = max(THRESHOLDS["degraded_throughput_mbps_min"],
                 5.0 + 0.1 * (rssi + 85.0))
    else:
        tp = max(0.0, 1.0 + 0.05 * (rssi + 90.0))

    # Link state
    if (rssi >= THRESHOLDS["healthy_rssi_dbm_min"]
            and plr <= THRESHOLDS["healthy_plr_pct_max"]):
        state, state_int = "healthy", 0
    elif (rssi >= THRESHOLDS["degraded_rssi_dbm_min"]
          and plr <= THRESHOLDS["degraded_plr_pct_max"]):
        state, state_int = "degraded", 1
    else:
        state, state_int = "disconnected", 2

    return {
        "dist3d_m":           round(d3d, 2),
        "dist2d_m":           round(d2d, 2),
        "rssi_dbm_est":       round(rssi, 2),
        "snr_db_est":         round(snr,  2),
        "plr_pct_est":        round(plr,  2),
        "throughput_mbps_est":round(tp,   2),
        "link_state":         state,
        "link_state_int":     state_int,
        "blocked_building_count": blocked,
    }


def _optimal_relay(uid_pos: dict[int, tuple], buildings,
                   noises_r: dict[int, float], noises_p: dict[int, float],
                   src: int, dst: int) -> int:
    """Find relay UAV that maximises min-RSSI of the two relay legs."""
    best_relay, best_score = src, -9999.0
    for r in uid_pos:
        if r == src or r == dst:
            continue
        x0, y0, z0, _ = uid_pos[src]
        xr, yr, zr, _ = uid_pos[r]
        x1, y1, z1, _ = uid_pos[dst]
        m1 = _compute_link(x0, y0, z0, xr, yr, zr, buildings,
                           noises_r.get(r, 0.0), noises_p.get(r, 0.0))
        m2 = _compute_link(xr, yr, zr, x1, y1, z1, buildings,
                           noises_r.get(dst, 0.0), noises_p.get(dst, 0.0))
        score = min(m1["rssi_dbm_est"], m2["rssi_dbm_est"])
        if score > best_score:
            best_score, best_relay = score, r
    return best_relay


# ── 시나리오 정의 ─────────────────────────────────────────────────────────────

SCENARIOS_3D: list[dict] = [
    # ── 1. 고도층 분리 ─────────────────────────────────────────────────────────
    {
        "scenario_id": "altitude_step",
        "description": "5 UAVs fly east at 3 altitude layers: 10m(low)/30m(relay)/50m(high). "
                       "Buildings block low-altitude links; relay at 30m bridges layers.",
        "relay_uav": 2,
        "duration_s": 60.0,
        "time_step_s": 0.25,
        "mobility": "linear",
        "initial_positions": {  # (x, y, z)
            0: (45.0, 75.0, 50.0),
            1: (45.0, 45.0, 10.0),
            2: (60.0, 60.0, 30.0),
            3: (75.0, 45.0, 10.0),
            4: (75.0, 75.0, 50.0),
        },
        "velocities": {  # (vx, vy, vz)
            0: (5.0, -0.5, 0.0),
            1: (5.0,  0.5, 0.0),
            2: (5.0,  0.0, 0.0),
            3: (5.0, -0.5, 0.0),
            4: (5.0,  0.5, 0.0),
        },
    },
    # ── 2. 점진적 상승 ─────────────────────────────────────────────────────────
    {
        "scenario_id": "altitude_climb",
        "description": "All UAVs fly east and gradually climb from 10m to 80m. "
                       "Buildings (30/40/25m) block early, then UAVs clear them by altitude.",
        "relay_uav": 2,
        "duration_s": 60.0,
        "time_step_s": 0.25,
        "mobility": "linear",
        "initial_positions": {
            0: (45.0, 75.0, 10.0),
            1: (45.0, 45.0, 10.0),
            2: (60.0, 60.0, 10.0),
            3: (75.0, 45.0, 10.0),
            4: (75.0, 75.0, 10.0),
        },
        "velocities": {
            0: (5.0, -0.5, 1.17),  # climb to 80m in 60s
            1: (5.0,  0.5, 1.17),
            2: (5.0,  0.0, 1.17),
            3: (5.0, -0.5, 1.17),
            4: (5.0,  0.5, 1.17),
        },
    },
    # ── 3. 수직 중계 체인 ───────────────────────────────────────────────────────
    {
        "scenario_id": "vertical_relay",
        "description": "Source at z=10m, relay at z=40m, sink at z=80m. UAVs are "
                       "horizontally spread so direct link is lost; relay bridges altitude gap.",
        "relay_uav": 2,
        "duration_s": 40.0,
        "time_step_s": 0.25,
        "mobility": "linear",
        "initial_positions": {
            0: (80.0,  60.0, 10.0),   # source (low)
            1: (90.0,  70.0, 25.0),   # mid-low
            2: (100.0, 60.0, 40.0),   # relay (mid)
            3: (110.0, 50.0, 60.0),   # mid-high
            4: (120.0, 60.0, 80.0),   # sink (high)
        },
        "velocities": {
            0: (-1.0,  0.5, 0.0),
            1: ( 0.5, -0.5, 0.5),
            2: ( 0.0,  0.0, 0.0),
            3: ( 0.5,  0.5, -0.5),
            4: ( 1.0, -0.5, 0.0),
        },
    },
    # ── 4. 3D 피라미드 편대 ────────────────────────────────────────────────────
    {
        "scenario_id": "formation_3d",
        "description": "3D pyramid formation: UAV2 at apex (z=60m), UAV0/1/3/4 at base (z=20m). "
                       "Flying east together; altitude links tested.",
        "relay_uav": 2,
        "duration_s": 60.0,
        "time_step_s": 0.25,
        "mobility": "linear",
        "initial_positions": {
            0: (45.0, 50.0, 20.0),
            1: (45.0, 70.0, 20.0),
            2: (55.0, 60.0, 60.0),   # apex relay
            3: (65.0, 50.0, 20.0),
            4: (65.0, 70.0, 20.0),
        },
        "velocities": {
            0: (4.5,  0.0, 0.0),
            1: (4.5,  0.0, 0.0),
            2: (4.5,  0.0, 0.0),
            3: (4.5,  0.0, 0.0),
            4: (4.5,  0.0, 0.0),
        },
    },
    # ── 5. 정현파 고도 변화 ────────────────────────────────────────────────────
    {
        "scenario_id": "altitude_oscillation",
        "description": "UAVs fly east with phase-shifted sinusoidal altitude (period 20s, amp 20m). "
                       "Dynamic altitude differences create transient link variations.",
        "relay_uav": 2,
        "duration_s": 60.0,
        "time_step_s": 0.25,
        "mobility": "altitude_osc",
        "base_z": 35.0,
        "amp_z": 20.0,
        "period_s": 20.0,
        "initial_positions": {
            0: (45.0, 75.0, 35.0),
            1: (45.0, 45.0, 35.0),
            2: (60.0, 60.0, 35.0),
            3: (75.0, 45.0, 35.0),
            4: (75.0, 75.0, 35.0),
        },
        "velocities": {
            0: (5.0, -0.5, 0.0),
            1: (5.0,  0.5, 0.0),
            2: (5.0,  0.0, 0.0),
            3: (5.0, -0.5, 0.0),
            4: (5.0,  0.5, 0.0),
        },
    },
    # ── 6. 급강하 후 상승 (2페이즈) ────────────────────────────────────────────
    {
        "scenario_id": "dive_and_rise",
        "description": "Phase 1 (0~20s): UAVs dive from z=50m to z=5m (enter building-blocked zone). "
                       "Phase 2 (20~60s): UAVs rise to z=70m while spreading horizontally.",
        "relay_uav": 2,
        "duration_s": 60.0,
        "time_step_s": 0.25,
        "mobility": "phases",
        "phases": [
            {
                "duration_s": 20.0,
                "initial_positions": {
                    0: (95.0, 80.0, 50.0),
                    1: (95.0, 40.0, 50.0),
                    2: (95.0, 60.0, 50.0),
                    3: (95.0, 65.0, 50.0),
                    4: (95.0, 55.0, 50.0),
                },
                "velocities": {
                    0: ( 2.0,  0.0, -2.25),
                    1: ( 2.0,  0.0, -2.25),
                    2: ( 0.0,  0.0, -2.25),
                    3: (-2.0,  0.0, -2.25),
                    4: (-2.0,  0.0, -2.25),
                },
            },
            {
                "duration_s": 40.0,
                "velocities": {
                    0: ( 4.0,  0.5, 1.625),
                    1: ( 4.0, -0.5, 1.625),
                    2: ( 2.0,  0.0, 1.625),
                    3: (-4.0,  0.5, 1.625),
                    4: (-4.0, -0.5, 1.625),
                },
            },
        ],
    },
    # ── 7. 2층 고도 교행 ───────────────────────────────────────────────────────
    {
        "scenario_id": "layered_corridor",
        "description": "UAV0/1/3 fly at z=15m (below B0/B1 height → blocked). "
                       "UAV2 (relay) at z=45m (above all buildings → clear LOS). "
                       "UAV4 at z=15m. Relay bridges both layers.",
        "relay_uav": 2,
        "duration_s": 60.0,
        "time_step_s": 0.25,
        "mobility": "linear",
        "initial_positions": {
            0: (45.0, 75.0, 15.0),
            1: (45.0, 45.0, 15.0),
            2: (60.0, 60.0, 45.0),   # high-altitude relay
            3: (75.0, 45.0, 15.0),
            4: (75.0, 75.0, 15.0),
        },
        "velocities": {
            0: (5.0, -0.5, 0.0),
            1: (5.0,  0.5, 0.0),
            2: (5.0,  0.0, 0.0),
            3: (5.0, -0.5, 0.0),
            4: (5.0,  0.5, 0.0),
        },
    },
    # ── 8. 나선형 상승 ─────────────────────────────────────────────────────────
    {
        "scenario_id": "spiral_ascent",
        "description": "UAV0/1/3/4 orbit center (100,60) at radius 50m while ascending. "
                       "UAV2 is stationary relay. Helical paths create varied link distances.",
        "relay_uav": 2,
        "duration_s": 40.0,
        "time_step_s": 0.25,
        "mobility": "spiral",
        "center": (100.0, 60.0),
        "z_start": 15.0,
        "z_end": 65.0,
        "orbits": {
            0: {"radius": 50.0, "initial_angle_deg":  90.0, "angular_velocity_rad_s": 0.2},
            1: {"radius": 50.0, "initial_angle_deg": 270.0, "angular_velocity_rad_s": 0.2},
            2: {"radius":  0.0, "initial_angle_deg":   0.0, "angular_velocity_rad_s": 0.0},
            3: {"radius": 50.0, "initial_angle_deg":   0.0, "angular_velocity_rad_s": 0.2},
            4: {"radius": 50.0, "initial_angle_deg": 180.0, "angular_velocity_rad_s": 0.2},
        },
    },
    # ── 9. 중계 고도 고정, 나머지 분산 ─────────────────────────────────────────
    {
        "scenario_id": "hover_relay",
        "description": "UAV2 hovers at (100,60,z=45). Others start nearby and "
                       "spread outward, testing altitude-advantage relay at mid-height.",
        "relay_uav": 2,
        "duration_s": 40.0,
        "time_step_s": 0.25,
        "mobility": "linear",
        "initial_positions": {
            0: (100.0, 60.0, 20.0),
            1: (100.0, 60.0, 20.0),
            2: (100.0, 60.0, 45.0),
            3: (100.0, 60.0, 20.0),
            4: (100.0, 60.0, 20.0),
        },
        "velocities": {
            0: ( 0.0,  6.0, 0.5),
            1: ( 0.0, -6.0, 0.5),
            2: ( 0.0,  0.0, 0.0),
            3: (-6.0,  0.0, 0.5),
            4: ( 6.0,  0.0, 0.5),
        },
    },
    # ── 10. 건물 고도 회피 ─────────────────────────────────────────────────────
    {
        "scenario_id": "altitude_avoid",
        "description": "UAVs start at z=15m (blocked by buildings). "
                       "They climb to z=45m mid-flight to clear all buildings (B0=30m, B1=40m, B2=25m). "
                       "Tests autonomous altitude-based building avoidance.",
        "relay_uav": 2,
        "duration_s": 60.0,
        "time_step_s": 0.25,
        "mobility": "phases",
        "phases": [
            {
                "duration_s": 20.0,
                "initial_positions": {
                    0: (45.0, 75.0, 15.0),
                    1: (45.0, 45.0, 15.0),
                    2: (60.0, 60.0, 15.0),
                    3: (75.0, 45.0, 15.0),
                    4: (75.0, 75.0, 15.0),
                },
                "velocities": {
                    0: (5.0, -0.5, 1.5),   # climb 30m in 20s
                    1: (5.0,  0.5, 1.5),
                    2: (5.0,  0.0, 1.5),
                    3: (5.0, -0.5, 1.5),
                    4: (5.0,  0.5, 1.5),
                },
            },
            {
                "duration_s": 40.0,
                "velocities": {             # maintain z=45m, continue east
                    0: (5.0, -0.5, 0.0),
                    1: (5.0,  0.5, 0.0),
                    2: (5.0,  0.0, 0.0),
                    3: (5.0, -0.5, 0.0),
                    4: (5.0,  0.5, 0.0),
                },
            },
        ],
    },
]


# ── 궤적 계산 ─────────────────────────────────────────────────────────────────

def _simulate_trajectory(scen: dict) -> list[dict]:
    """Return list of {time_s, uav_id, x, y, z, speed_mps, role} dicts."""
    mob = scen.get("mobility", "linear")
    dt  = scen["time_step_s"]
    relay_uav = scen["relay_uav"]
    sid = scen["scenario_id"]

    rows: list[dict] = []

    def _role(uid: int) -> str:
        return "relay_anchor" if uid == relay_uav else "peripheral"

    def _append(t, pos_dict):
        for uid, (x, y, z) in pos_dict.items():
            vx = scen.get("velocities", {}).get(uid, (0, 0, 0))
            speed = math.sqrt(vx[0]**2 + vx[1]**2 + vx[2]**2) if isinstance(vx, tuple) else 0.0
            rows.append({
                "scenario_id": sid,
                "time_s": round(t, 4),
                "uav_id": uid,
                "x_m": round(x, 4),
                "y_m": round(y, 4),
                "z_m": round(z, 4),
                "speed_mps": round(speed, 4),
                "role": _role(uid),
            })

    if mob == "linear":
        init_pos = scen["initial_positions"]
        velocities = scen.get("velocities", {u: (0, 0, 0) for u in init_pos})
        n_steps = int(scen["duration_s"] / dt) + 1
        pos = {u: list(init_pos[u]) for u in init_pos}
        for step in range(n_steps):
            t = step * dt
            _append(t, {u: tuple(pos[u]) for u in pos})
            for u in pos:
                vx, vy, vz = velocities.get(u, (0, 0, 0))
                pos[u][0] += vx * dt
                pos[u][1] += vy * dt
                pos[u][2] = max(0.0, pos[u][2] + vz * dt)

    elif mob == "altitude_osc":
        init_pos   = scen["initial_positions"]
        velocities = scen.get("velocities", {})
        base_z  = scen["base_z"]
        amp_z   = scen["amp_z"]
        period  = scen["period_s"]
        n_steps = int(scen["duration_s"] / dt) + 1
        pos = {u: list(init_pos[u]) for u in init_pos}
        phase_offsets = {u: (2 * math.pi * u / NUM_UAVS) for u in pos}
        for step in range(n_steps):
            t = step * dt
            snap = {}
            for u in pos:
                z_osc = base_z + amp_z * math.sin(2 * math.pi * t / period + phase_offsets[u])
                snap[u] = (pos[u][0], pos[u][1], max(0.0, z_osc))
            _append(t, snap)
            for u in pos:
                vx, vy, _ = velocities.get(u, (0, 0, 0))
                pos[u][0] += vx * dt
                pos[u][1] += vy * dt

    elif mob == "phases":
        phases = scen["phases"]
        phase_pos = None
        t_global  = 0.0
        for ph_idx, phase in enumerate(phases):
            if ph_idx == 0:
                phase_pos = {u: list(phase["initial_positions"][u])
                             for u in phase["initial_positions"]}
            n_steps = int(phase["duration_s"] / dt) + 1
            vels    = phase["velocities"]
            for step in range(n_steps):
                t = t_global + step * dt
                _append(t, {u: tuple(phase_pos[u]) for u in phase_pos})
                for u in phase_pos:
                    vx, vy, vz = vels.get(u, (0, 0, 0))
                    phase_pos[u][0] += vx * dt
                    phase_pos[u][1] += vy * dt
                    phase_pos[u][2] = max(0.0, phase_pos[u][2] + vz * dt)
            t_global += phase["duration_s"]

    elif mob == "spiral":
        cx, cy    = scen["center"]
        z_start   = scen["z_start"]
        z_end     = scen["z_end"]
        orb_cfg   = scen["orbits"]
        duration  = scen["duration_s"]
        n_steps   = int(duration / dt) + 1
        for step in range(n_steps):
            t = step * dt
            frac = t / duration
            snap = {}
            for u, cfg in orb_cfg.items():
                r   = cfg["radius"]
                ang = math.radians(cfg["initial_angle_deg"]) + cfg["angular_velocity_rad_s"] * t
                x   = cx + r * math.cos(ang)
                y   = cy + r * math.sin(ang)
                z   = z_start + (z_end - z_start) * frac
                snap[u] = (x, y, max(0.0, z))
            _append(t, snap)

    return rows


# ── CSV 출력 ─────────────────────────────────────────────────────────────────

def write_obstacles():
    path = OUT_DIR / "obstacles_3d.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["building_id", "x_min_m", "x_max_m", "y_min_m", "y_max_m",
                    "height_m", "attenuation_db", "note"])
        for b in BUILDINGS_3D:
            w.writerow([b["building_id"],
                        b["x_min_m"], b["x_max_m"],
                        b["y_min_m"], b["y_max_m"],
                        b["height_m"], b["attenuation_db"],
                        b["note"]])
    print(f"  저장: {path}")


def write_positions_and_links():
    pos_path  = OUT_DIR / "uav_positions_3d.csv"
    link_path = OUT_DIR / "link_metrics_3d.csv"

    pos_cols  = ["scenario_id", "time_s", "uav_id",
                 "x_m", "y_m", "z_m", "speed_mps", "role"]
    link_cols = ["scenario_id", "time_s", "src_uav", "dst_uav",
                 "dist3d_m", "dist2d_m",
                 "rssi_dbm_est", "snr_db_est", "plr_pct_est", "throughput_mbps_est",
                 "link_state", "link_state_int", "hop_count",
                 "blocked_building_count", "optimal_relay_uav"]

    with (open(pos_path,  "w", newline="", encoding="utf-8") as fp,
          open(link_path, "w", newline="", encoding="utf-8") as fl):

        wp = csv.DictWriter(fp, fieldnames=pos_cols)
        wl = csv.DictWriter(fl, fieldnames=link_cols)
        wp.writeheader()
        wl.writeheader()

        for scen in SCENARIOS_3D:
            sid = scen["scenario_id"]
            print(f"  시나리오: {sid} ...", end=" ", flush=True)

            traj_rows = _simulate_trajectory(scen)
            wp.writerows(traj_rows)

            # Group rows by time_s
            ts_dict: dict[float, dict[int, tuple]] = {}
            for r in traj_rows:
                t = r["time_s"]
                if t not in ts_dict:
                    ts_dict[t] = {}
                ts_dict[t][r["uav_id"]] = (r["x_m"], r["y_m"], r["z_m"], r["role"])

            # Build noise sequences per (pair, scenario) for temporal correlation
            n_steps = len(ts_dict)
            pair_noise_r: dict[tuple, list] = {}
            pair_noise_p: dict[tuple, list] = {}
            seed_base = abs(hash(sid)) % (2**31)
            pair_idx  = 0
            for src in range(NUM_UAVS):
                for dst in range(src + 1, NUM_UAVS):
                    pair_noise_r[(src, dst)] = _ar1_noise(
                        n_steps, RSSI_NOISE_STD, NOISE_CORR, seed_base + pair_idx)
                    pair_noise_p[(src, dst)] = _ar1_noise(
                        n_steps, PLR_NOISE_STD, PLR_NOISE_CORR, seed_base + pair_idx + 10000)
                    pair_idx += 1

            for step_i, t_s in enumerate(sorted(ts_dict.keys())):
                uid_pos = ts_dict[t_s]

                for src in range(NUM_UAVS):
                    for dst in range(src + 1, NUM_UAVS):
                        if src not in uid_pos or dst not in uid_pos:
                            continue

                        x0, y0, z0, _ = uid_pos[src]
                        x1, y1, z1, _ = uid_pos[dst]
                        nr = pair_noise_r[(src, dst)][step_i]
                        np_ = pair_noise_p[(src, dst)][step_i]

                        m = _compute_link(x0, y0, z0, x1, y1, z1,
                                          BUILDINGS_3D, nr, np_)

                        relay = _optimal_relay(uid_pos, BUILDINGS_3D,
                                               {u: 0.0 for u in uid_pos},
                                               {u: 0.0 for u in uid_pos},
                                               src, dst)

                        hop = 1 if m["link_state"] != "disconnected" else 2

                        wl.writerow({
                            "scenario_id":          sid,
                            "time_s":               t_s,
                            "src_uav":              src,
                            "dst_uav":              dst,
                            "dist3d_m":             m["dist3d_m"],
                            "dist2d_m":             m["dist2d_m"],
                            "rssi_dbm_est":         m["rssi_dbm_est"],
                            "snr_db_est":           m["snr_db_est"],
                            "plr_pct_est":          m["plr_pct_est"],
                            "throughput_mbps_est":  m["throughput_mbps_est"],
                            "link_state":           m["link_state"],
                            "link_state_int":       m["link_state_int"],
                            "hop_count":            hop,
                            "blocked_building_count": m["blocked_building_count"],
                            "optimal_relay_uav":    relay,
                        })

            print("완료")

    print(f"  저장: {pos_path}")
    print(f"  저장: {link_path}")


def main():
    print(f"출력 디렉토리: {OUT_DIR}")
    print(f"건물 수: {len(BUILDINGS_3D)}")
    print(f"시나리오 수: {len(SCENARIOS_3D)}\n")

    print("[1/2] obstacles_3d.csv 생성 중...")
    write_obstacles()

    print("\n[2/2] uav_positions_3d.csv / link_metrics_3d.csv 생성 중...")
    write_positions_and_links()

    print("\n완료.")


if __name__ == "__main__":
    main()
