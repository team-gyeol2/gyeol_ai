#!/usr/bin/env python3
"""Generate a 2D UAV dataset for the ns-3 capstone scenario.

Fixes applied
─────────────
1. hop_count is determined by link *state* (healthy/degraded → 1),
   not by a hard-coded 30 m distance threshold.
2. blocked_building_count / blocked_building_ids / blocked_attenuation_db
   for 2-hop routes now reflect the actual relay segments
   (src→relay ∪ relay→dst), not the direct src→dst path.
3. Three scenarios are generated:
     • corridor_baseline – building degradation only (no disconnection)
     • relay_stretch     – east/west divergence → relay and direct links
                           eventually disconnect (~t=15 s)
     • cluster_spread    – radial divergence → most pairs disconnect by t≈12 s
   This ensures the dataset contains all three link_state values
   (healthy / degraded / disconnected) needed for AI failure-prediction.
4. Time resolution raised to 0.5 s with longer durations → ~2 000 rows
   instead of 110.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

# ── Channel noise parameters (AR(1) temporally-correlated) ───────────────────
# Real wireless channels exhibit slow fading: noise persists across timesteps.
# AR(1): noise[t] = rho * noise[t-1] + sqrt(1-rho^2) * sigma * eps
#   RSSI_NOISE_STD  : shadowing amplitude (dBm)
#   NOISE_CORR      : temporal correlation (0=white, 1=constant) — 0.9 = slow fading
#   PLR_NOISE_STD   : packet loss fluctuation (%)
RSSI_NOISE_STD = 8.0
NOISE_CORR     = 0.88
PLR_NOISE_STD  = 7.0
PLR_NOISE_CORR = 0.85
GPS_NOISE_STD  = 5.0   # ±5m GPS position error (typical UAV GPS accuracy)


def _gen_ar1_noise(n_steps: int, std: float, rho: float, seed: int) -> list[float]:
    """Generate AR(1) correlated noise sequence."""
    rng = random.Random(seed)
    innov_std = std * math.sqrt(1.0 - rho ** 2)
    noise = [rng.gauss(0, std)]
    for _ in range(n_steps - 1):
        noise.append(rho * noise[-1] + rng.gauss(0, innov_std))
    return noise

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"

NUM_UAVS = 5

THRESHOLDS: dict[str, float] = {
    "healthy_rssi_dbm_min": -78.0,
    "degraded_rssi_dbm_min": -85.0,
    "healthy_plr_pct_max": 5.0,
    "degraded_plr_pct_max": 20.0,
    "healthy_rtt_ms_max": 30.0,
    "degraded_rtt_ms_max": 60.0,
    "healthy_throughput_mbps_min": 8.0,
    "degraded_throughput_mbps_min": 3.0,
    "reconfig_trigger_plr_pct": 15.0,
    "reconfig_trigger_rssi_dbm": -80.0,
    "reconfig_trigger_rtt_ms": 40.0,
    "reconfig_trigger_throughput_mbps": 4.0,
}

BUILDINGS: list[dict] = [
    {
        "building_id": "B0",
        "kind": "fixed_building",
        "x_min_m": 88.0,
        "x_max_m": 100.0,
        "y_min_m": 52.0,
        "y_max_m": 58.0,
        "attenuation_db": 6.0,
        "note": "Mid-map lower corridor obstacle; often affects UAV1-UAV2 LOS.",
    },
    {
        "building_id": "B1",
        "kind": "fixed_building",
        "x_min_m": 118.0,
        "x_max_m": 132.0,
        "y_min_m": 62.0,
        "y_max_m": 68.0,
        "attenuation_db": 6.0,
        "note": "Upper corridor obstacle; can degrade UAV0/UAV4 links to the relay.",
    },
    {
        "building_id": "B2",
        "kind": "fixed_building",
        "x_min_m": 146.0,
        "x_max_m": 160.0,
        "y_min_m": 50.0,
        "y_max_m": 56.0,
        "attenuation_db": 6.0,
        "note": "Late-stage lower corridor obstacle; affects relay access near map exit.",
    },
]

# ── Scenario definitions ──────────────────────────────────────────────────────
# relay_uav: index of the UAV used as the single relay anchor when a direct
#            link is in 'disconnected' state.
SCENARIOS: list[dict] = [
    {
        "scenario_id": "corridor_baseline",
        "description": (
            "All 5 UAVs fly east in loose formation; fixed buildings cause "
            "progressive link degradation as drones transit the obstacle corridor."
        ),
        "initial_positions": {
            0: (45.0, 75.0),
            1: (45.0, 45.0),
            2: (60.0, 60.0),
            3: (75.0, 45.0),
            4: (75.0, 75.0),
        },
        "velocities": {
            0: (5.0, -0.5),
            1: (5.0,  0.5),
            2: (5.0,  0.0),
            3: (5.0, -0.5),
            4: (5.0,  0.5),
        },
        "duration_s": 60.0,
        "time_step_s": 0.25,
        "relay_uav": 2,
    },
    {
        "scenario_id": "relay_stretch",
        "description": (
            "UAV0/1 fly east and UAV3/4 fly west from a common centre; UAV2 "
            "stays stationary as relay anchor. The east and west groups lose "
            "relay connectivity near t=15 s, producing disconnected states."
        ),
        "initial_positions": {
            0: (95.0, 70.0),
            1: (95.0, 50.0),
            2: (95.0, 60.0),
            3: (95.0, 65.0),
            4: (95.0, 55.0),
        },
        "velocities": {
            0: ( 6.0,  0.5),
            1: ( 6.0, -0.5),
            2: ( 0.0,  0.0),
            3: (-6.0,  0.5),
            4: (-6.0, -0.5),
        },
        "duration_s": 20.0,
        "time_step_s": 0.25,
        "relay_uav": 2,
    },
    {
        "scenario_id": "cluster_spread",
        "description": (
            "All UAVs start at the same point and spread outward in different "
            "directions. Most pairs disconnect by t≈12 s; UAV2 drifts slowly "
            "east and briefly acts as relay before also losing connectivity."
        ),
        "initial_positions": {
            0: (95.0, 60.0),
            1: (95.0, 60.0),
            2: (95.0, 60.0),
            3: (95.0, 60.0),
            4: (95.0, 60.0),
        },
        "velocities": {
            0: ( 0.0,  6.0),
            1: ( 0.0, -6.0),
            2: ( 0.5,  0.0),
            3: (-6.0,  0.0),
            4: ( 6.0,  0.0),
        },
        "duration_s": 18.0,
        "time_step_s": 0.25,
        "relay_uav": 2,
    },
    # ── 추가 시나리오 3개 ───────────────────────────────────────────────────
    {
        "scenario_id": "converge_diverge",
        "description": (
            "5 UAVs start far apart (disconnected), converge to a common centre "
            "at t=10 s (healthy), then diverge again. Provides the recovery "
            "pattern disconnected→degraded→healthy→degraded→disconnected that "
            "the first 3 scenarios completely lack."
        ),
        "mobility": "phases",
        "relay_uav": 2,
        "duration_s": 20.0,
        "time_step_s": 0.25,
        "phases": [
            {
                "duration_s": 10.0,
                "initial_positions": {
                    0: ( 95.0, 110.0),
                    1: ( 95.0,  10.0),
                    2: ( 45.0,  60.0),
                    3: (145.0,  80.0),
                    4: (145.0,  40.0),
                },
                "velocities": {   # 수렴 (모두 중심 (95,60) 방향)
                    0: ( 0.0, -5.0),
                    1: ( 0.0,  5.0),
                    2: ( 5.0,  0.0),
                    3: (-5.0, -2.0),
                    4: (-5.0,  2.0),
                },
            },
            {
                "duration_s": 10.0,
                "velocities": {   # 발산 (중심에서 반대 방향)
                    0: ( 0.0,  5.0),
                    1: ( 0.0, -5.0),
                    2: (-5.0,  0.0),
                    3: ( 5.0,  2.0),
                    4: ( 5.0, -2.0),
                },
            },
        ],
    },
    {
        "scenario_id": "slow_separation",
        "description": (
            "Same east/west divergence as relay_stretch but at 3 m/s (half speed). "
            "First disconnect occurs at t≈29 s, giving a much longer degraded "
            "window so the model can learn gradual decline patterns."
        ),
        "mobility": "linear",
        "relay_uav": 2,
        "duration_s": 30.0,
        "time_step_s": 0.25,
        "initial_positions": {
            0: (95.0, 70.0),
            1: (95.0, 50.0),
            2: (95.0, 60.0),
            3: (95.0, 65.0),
            4: (95.0, 55.0),
        },
        "velocities": {
            0: ( 3.0,  0.25),
            1: ( 3.0, -0.25),
            2: ( 0.0,  0.0),
            3: (-3.0,  0.25),
            4: (-3.0, -0.25),
        },
    },
    {
        "scenario_id": "orbit",
        "description": (
            "4 UAVs orbit a stationary relay (UAV2) at radius 50 m, angular "
            "speed 0.2 rad/s (one orbit ≈ 31 s). Opposite pairs are ~100 m "
            "apart → disconnected; adjacent pairs ~71 m → degraded; each UAV "
            "is 50 m from relay → degraded. Simulation covers ≈1.3 orbits."
        ),
        "mobility": "circular",
        "relay_uav": 2,
        "duration_s": 40.0,
        "time_step_s": 0.25,
        "center": (100.0, 60.0),
        "orbits": {
            0: {"radius": 50.0, "initial_angle_deg":  90.0, "angular_velocity_rad_s": 0.2},
            1: {"radius": 50.0, "initial_angle_deg": 270.0, "angular_velocity_rad_s": 0.2},
            2: {"radius":  0.0, "initial_angle_deg":   0.0, "angular_velocity_rad_s": 0.0},
            3: {"radius": 50.0, "initial_angle_deg":   0.0, "angular_velocity_rad_s": 0.2},
            4: {"radius": 50.0, "initial_angle_deg": 180.0, "angular_velocity_rad_s": 0.2},
        },
    },
    # ── 랜덤 이동 시나리오 3개 (Random Waypoint) ────────────────────────────
    {
        "scenario_id": "random_waypoint_1",
        "description": (
            "5 UAVs move independently via Random Waypoint model (seed=42). "
            "Each UAV picks a random destination within the map area and "
            "travels at a random speed (2~8 m/s). On arrival it picks a new "
            "waypoint. Speeds differ per UAV, producing realistic mixed "
            "link-state transitions."
        ),
        "mobility": "random_waypoint",
        "relay_uav": 2,
        "duration_s": 40.0,
        "time_step_s": 0.25,
        "seed": 42,
        "x_range": (50.0, 180.0),
        "y_range": (40.0, 90.0),
        "speed_range": (2.0, 8.0),
    },
    {
        "scenario_id": "random_waypoint_2",
        "description": (
            "Random Waypoint scenario with seed=123. Higher average speed "
            "setting results in more frequent disconnected events."
        ),
        "mobility": "random_waypoint",
        "relay_uav": 2,
        "duration_s": 40.0,
        "time_step_s": 0.25,
        "seed": 123,
        "x_range": (50.0, 180.0),
        "y_range": (40.0, 90.0),
        "speed_range": (3.0, 9.0),
    },
    {
        "scenario_id": "random_waypoint_3",
        "description": (
            "Random Waypoint scenario with seed=999. Lower speed range "
            "produces longer healthy/degraded periods with gradual transitions."
        ),
        "mobility": "random_waypoint",
        "relay_uav": 2,
        "duration_s": 40.0,
        "time_step_s": 0.25,
        "seed": 999,
        "x_range": (50.0, 180.0),
        "y_range": (40.0, 90.0),
        "speed_range": (1.0, 5.0),
    },
    # ── 동적 relay 전환 시나리오 3개 ─────────────────────────────────────────
    {
        "scenario_id": "relay_handover",
        "description": (
            "UAV1 and UAV2 swap roles: UAV2 starts as relay at center and moves "
            "east while UAV1 approaches from the east. Relay switches UAV2→UAV1 "
            "around t=5s, then UAV1→UAV3 around t=20s as UAV1 drifts past center."
        ),
        "mobility": "linear",
        "relay_uav": 2,
        "duration_s": 30.0,
        "time_step_s": 0.25,
        "initial_positions": {
            0: ( 60.0, 90.0),
            1: (150.0, 60.0),
            2: (100.0, 60.0),
            3: ( 60.0, 30.0),
            4: (105.0, 45.0),
        },
        "velocities": {
            0: ( 0.5,  0.0),
            1: (-5.0,  0.0),
            2: ( 5.0,  0.0),
            3: ( 0.5,  0.0),
            4: ( 0.0,  0.0),
        },
    },
    {
        "scenario_id": "relay_rotation",
        "description": (
            "Three-phase relay baton pass: UAV2 holds relay in phase1, "
            "swaps position with UAV0 in phase2 (relay→UAV0), then UAV0 "
            "swaps with UAV4 in phase3 (relay→UAV4). Two clear relay "
            "switches at t≈24s and t≈39s."
        ),
        "mobility": "phases",
        "relay_uav": 2,
        "duration_s": 45.0,
        "time_step_s": 0.25,
        "phases": [
            {
                "duration_s": 15.0,
                "initial_positions": {
                    0: ( 50.0, 110.0),
                    1: (150.0, 110.0),
                    2: (100.0,  60.0),
                    3: ( 50.0,  10.0),
                    4: (150.0,  10.0),
                },
                "velocities": {
                    0: (0.0, 0.0),
                    1: (0.0, 0.0),
                    2: (0.0, 0.0),
                    3: (0.0, 0.0),
                    4: (0.0, 0.0),
                },
            },
            {
                "duration_s": 15.0,
                "velocities": {
                    0: ( 50/15, -50/15),
                    1: (0.0, 0.0),
                    2: (-50/15,  50/15),
                    3: (0.0, 0.0),
                    4: (0.0, 0.0),
                },
            },
            {
                "duration_s": 15.0,
                "velocities": {
                    0: ( 50/15, -50/15),
                    1: (0.0, 0.0),
                    2: (0.0, 0.0),
                    3: (0.0, 0.0),
                    4: (-50/15,  50/15),
                },
            },
        ],
    },
    {
        "scenario_id": "relay_competition",
        "description": (
            "Four UAVs take turns converging to the center while the current "
            "relay disperses. Four-phase structure produces three relay switches: "
            "UAV2→UAV0→UAV3→UAV1, giving the model diverse relay-transition "
            "patterns not seen in other scenarios."
        ),
        "mobility": "phases",
        "relay_uav": 2,
        "duration_s": 40.0,
        "time_step_s": 0.25,
        "phases": [
            {
                "duration_s": 10.0,
                "initial_positions": {
                    0: (160.0,  10.0),
                    1: (160.0, 110.0),
                    2: (100.0,  60.0),
                    3: ( 40.0,  10.0),
                    4: ( 40.0, 110.0),
                },
                "velocities": {
                    0: (0.0, 0.0),
                    1: (0.0, 0.0),
                    2: (0.0, 0.0),
                    3: (0.0, 0.0),
                    4: (0.0, 0.0),
                },
            },
            {
                "duration_s": 10.0,
                "velocities": {
                    0: (-60/10,  50/10),
                    1: (0.0, 0.0),
                    2: ( 60/10, -50/10),
                    3: (0.0, 0.0),
                    4: (0.0, 0.0),
                },
            },
            {
                "duration_s": 10.0,
                "velocities": {
                    0: ( 60/10, -50/10),
                    1: (0.0, 0.0),
                    2: (0.0, 0.0),
                    3: ( 60/10,  50/10),
                    4: (0.0, 0.0),
                },
            },
            {
                "duration_s": 10.0,
                "velocities": {
                    0: (0.0, 0.0),
                    1: (-60/10, -50/10),
                    2: (0.0, 0.0),
                    3: (-60/10,  50/10),
                    4: (0.0, 0.0),
                },
            },
        ],
    },
    # ── [신규] 다중 동시 단절 시나리오 5개 ──────────────────────────────────────
    {
        "scenario_id": "multi_disconnect_fast",
        "description": (
            "UAV0/1/3 scatter simultaneously at high speed (8 m/s) in three "
            "directions while UAV2/4 stay near center. Three links disconnect "
            "within t=8s, producing multi-link simultaneous failure patterns."
        ),
        "mobility": "linear",
        "relay_uav": 2,
        "duration_s": 20.0,
        "time_step_s": 0.25,
        "initial_positions": {
            0: (100.0, 60.0),
            1: (100.0, 60.0),
            2: (100.0, 60.0),
            3: (100.0, 60.0),
            4: (100.0, 60.0),
        },
        "velocities": {
            0: ( 8.0,  5.0),
            1: (-8.0,  5.0),
            2: ( 0.0,  0.5),
            3: ( 0.0, -8.0),
            4: ( 0.5,  0.0),
        },
    },
    {
        "scenario_id": "multi_disconnect_slow",
        "description": (
            "All 5 UAVs slowly diverge from a common center at 2 m/s in different "
            "directions. Provides gradual multi-link degradation → disconnection "
            "transition with long degraded windows for each pair."
        ),
        "mobility": "linear",
        "relay_uav": 2,
        "duration_s": 35.0,
        "time_step_s": 0.25,
        "initial_positions": {i: (100.0, 60.0) for i in range(5)},
        "velocities": {
            0: ( 2.0,  2.0),
            1: (-2.0,  2.0),
            2: ( 0.2,  0.0),
            3: (-2.0, -2.0),
            4: ( 2.0, -2.0),
        },
    },
    {
        "scenario_id": "full_scatter",
        "description": (
            "All 5 UAVs scatter radially from center at 5 m/s. No relay stays "
            "near center, causing complete network fragmentation by t≈15s. "
            "Tests model behavior when no relay path exists for any pair."
        ),
        "mobility": "linear",
        "relay_uav": 2,
        "duration_s": 22.0,
        "time_step_s": 0.25,
        "initial_positions": {i: (100.0, 60.0) for i in range(5)},
        "velocities": {
            0: ( 5.0,  0.0),
            1: (-5.0,  0.0),
            2: ( 0.0,  5.0),
            3: ( 0.0, -5.0),
            4: ( 3.5,  3.5),
        },
    },
    {
        "scenario_id": "partial_disconnect",
        "description": (
            "UAV0/1 maintain close formation (healthy) while UAV3/4 drift far "
            "apart (disconnected). UAV2 acts as relay in the middle but can only "
            "bridge one side at a time. Tests partial network connectivity."
        ),
        "mobility": "phases",
        "relay_uav": 2,
        "duration_s": 30.0,
        "time_step_s": 0.25,
        "phases": [
            {
                "duration_s": 5.0,
                "initial_positions": {
                    0: ( 80.0, 65.0),
                    1: ( 80.0, 55.0),
                    2: (100.0, 60.0),
                    3: (120.0, 65.0),
                    4: (120.0, 55.0),
                },
                "velocities": {i: (0.0, 0.0) for i in range(5)},
            },
            {
                "duration_s": 25.0,
                "velocities": {
                    0: ( 0.5,  0.3),
                    1: ( 0.5, -0.3),
                    2: ( 0.0,  0.0),
                    3: ( 6.0,  0.5),
                    4: ( 6.0, -0.5),
                },
            },
        ],
    },
    {
        "scenario_id": "wave_disconnect",
        "description": (
            "UAVs disconnect in sequence (cascade): UAV4 first at t≈5s, "
            "UAV3 at t≈10s, UAV1 at t≈15s, UAV0 last at t≈20s. "
            "Models cascading link failure scenario where farthest nodes "
            "disconnect first."
        ),
        "mobility": "phases",
        "relay_uav": 2,
        "duration_s": 30.0,
        "time_step_s": 0.25,
        "phases": [
            {
                "duration_s": 3.0,
                "initial_positions": {
                    0: ( 80.0, 60.0),
                    1: ( 90.0, 60.0),
                    2: (100.0, 60.0),
                    3: (110.0, 60.0),
                    4: (120.0, 60.0),
                },
                "velocities": {i: (0.0, 0.0) for i in range(5)},
            },
            {
                "duration_s": 27.0,
                "velocities": {
                    0: ( 1.5,  0.0),
                    1: ( 2.0,  0.0),
                    2: ( 0.0,  0.0),
                    3: (-3.5,  0.0),
                    4: (-7.0,  0.0),
                },
            },
        ],
    },
    # ── [신규] 군집 분리 시나리오 3개 ────────────────────────────────────────────
    {
        "scenario_id": "cluster_split_2",
        "description": (
            "Swarm splits into two distinct clusters: UAV0/1 fly northeast, "
            "UAV3/4 fly southwest, and UAV2 (relay) stays at center but "
            "eventually loses connectivity to both clusters simultaneously."
        ),
        "mobility": "linear",
        "relay_uav": 2,
        "duration_s": 25.0,
        "time_step_s": 0.25,
        "initial_positions": {
            0: (100.0, 62.0),
            1: (100.0, 58.0),
            2: (100.0, 60.0),
            3: (100.0, 62.0),
            4: (100.0, 58.0),
        },
        "velocities": {
            0: ( 5.0,  3.0),
            1: ( 5.0,  1.0),
            2: ( 0.0,  0.0),
            3: (-5.0, -1.0),
            4: (-5.0, -3.0),
        },
    },
    {
        "scenario_id": "cluster_split_3",
        "description": (
            "Swarm splits into three clusters: UAV0/1 fly north, UAV3 flies "
            "southeast, UAV4 flies southwest. UAV2 (relay) tries to bridge "
            "but loses all three clusters by t≈18s. Three-way fragmentation."
        ),
        "mobility": "linear",
        "relay_uav": 2,
        "duration_s": 25.0,
        "time_step_s": 0.25,
        "initial_positions": {i: (100.0, 60.0) for i in range(5)},
        "velocities": {
            0: ( 1.0,  6.0),
            1: (-1.0,  6.0),
            2: ( 0.0,  0.5),
            3: ( 6.0, -4.0),
            4: (-6.0, -4.0),
        },
    },
    {
        "scenario_id": "split_and_rejoin",
        "description": (
            "Two-phase scenario: UAV0/1 and UAV3/4 split into two clusters "
            "(t=0~15s), then reverse direction to rejoin (t=15~30s). "
            "Tests disconnected→degraded→healthy recovery transition."
        ),
        "mobility": "phases",
        "relay_uav": 2,
        "duration_s": 30.0,
        "time_step_s": 0.25,
        "phases": [
            {
                "duration_s": 15.0,
                "initial_positions": {
                    0: (100.0, 62.0),
                    1: (100.0, 58.0),
                    2: (100.0, 60.0),
                    3: (100.0, 62.0),
                    4: (100.0, 58.0),
                },
                "velocities": {
                    0: ( 5.5,  1.0),
                    1: ( 5.5, -1.0),
                    2: ( 0.0,  0.0),
                    3: (-5.5,  1.0),
                    4: (-5.5, -1.0),
                },
            },
            {
                "duration_s": 15.0,
                "velocities": {
                    0: (-5.5, -1.0),
                    1: (-5.5,  1.0),
                    2: ( 0.0,  0.0),
                    3: ( 5.5, -1.0),
                    4: ( 5.5,  1.0),
                },
            },
        ],
    },
    # ── [신규] 복잡 relay 시나리오 4개 ──────────────────────────────────────────
    {
        "scenario_id": "relay_failure",
        "description": (
            "Relay UAV2 itself rapidly exits the network eastward (8 m/s) "
            "while all other UAVs remain stationary. Relay path collapses "
            "at t≈8s, all previously relay-dependent pairs disconnect "
            "simultaneously."
        ),
        "mobility": "linear",
        "relay_uav": 2,
        "duration_s": 20.0,
        "time_step_s": 0.25,
        "initial_positions": {
            0: ( 70.0, 80.0),
            1: ( 70.0, 40.0),
            2: (100.0, 60.0),
            3: (130.0, 80.0),
            4: (130.0, 40.0),
        },
        "velocities": {
            0: (0.0, 0.0),
            1: (0.0, 0.0),
            2: (8.0, 0.0),
            3: (0.0, 0.0),
            4: (0.0, 0.0),
        },
    },
    {
        "scenario_id": "relay_oscillation",
        "description": (
            "Relay UAV2 oscillates east-west (±30 m, period≈20s) while "
            "UAV0/3 are on the east side and UAV1/4 on the west side. "
            "Link quality alternates between degraded/disconnected "
            "rhythmically, testing model on periodic patterns."
        ),
        "mobility": "phases",
        "relay_uav": 2,
        "duration_s": 40.0,
        "time_step_s": 0.25,
        "phases": [
            {
                "duration_s": 10.0,
                "initial_positions": {
                    0: (140.0, 70.0),
                    1: ( 60.0, 70.0),
                    2: (100.0, 60.0),
                    3: (140.0, 50.0),
                    4: ( 60.0, 50.0),
                },
                "velocities": {
                    0: (0.0, 0.0),
                    1: (0.0, 0.0),
                    2: (6.0, 0.0),
                    3: (0.0, 0.0),
                    4: (0.0, 0.0),
                },
            },
            {
                "duration_s": 10.0,
                "velocities": {
                    0: (0.0, 0.0),
                    1: (0.0, 0.0),
                    2: (-6.0, 0.0),
                    3: (0.0, 0.0),
                    4: (0.0, 0.0),
                },
            },
            {
                "duration_s": 10.0,
                "velocities": {
                    0: (0.0, 0.0),
                    1: (0.0, 0.0),
                    2: (6.0, 0.0),
                    3: (0.0, 0.0),
                    4: (0.0, 0.0),
                },
            },
            {
                "duration_s": 10.0,
                "velocities": {
                    0: (0.0, 0.0),
                    1: (0.0, 0.0),
                    2: (-6.0, 0.0),
                    3: (0.0, 0.0),
                    4: (0.0, 0.0),
                },
            },
        ],
    },
    {
        "scenario_id": "relay_chain_3hop",
        "description": (
            "Linear chain: UAV0 far west, UAV4 far east, with UAV1/UAV3 "
            "as intermediate nodes and UAV2 as central bridge. Tests "
            "multi-hop relay chain where link failure at any node cascades."
        ),
        "mobility": "linear",
        "relay_uav": 2,
        "duration_s": 30.0,
        "time_step_s": 0.25,
        "initial_positions": {
            0: ( 50.0, 60.0),
            1: ( 75.0, 60.0),
            2: (100.0, 60.0),
            3: (125.0, 60.0),
            4: (150.0, 60.0),
        },
        "velocities": {
            0: (-3.0,  0.5),
            1: ( 0.5,  0.0),
            2: ( 0.0,  0.0),
            3: ( 0.5,  0.0),
            4: ( 3.0, -0.5),
        },
    },
    {
        "scenario_id": "double_relay_compete",
        "description": (
            "Two candidate relays (UAV2, UAV3) both near center initially. "
            "UAV2 drifts north and UAV3 drifts south, creating two competing "
            "relay zones. UAV0/1 follow UAV2 cluster, UAV4 follows UAV3. "
            "Tests relay selection under split relay topology."
        ),
        "mobility": "linear",
        "relay_uav": 2,
        "duration_s": 30.0,
        "time_step_s": 0.25,
        "initial_positions": {
            0: ( 80.0, 72.0),
            1: ( 80.0, 68.0),
            2: (100.0, 70.0),
            3: (100.0, 50.0),
            4: (120.0, 48.0),
        },
        "velocities": {
            0: ( 1.5,  1.0),
            1: ( 1.5,  0.5),
            2: ( 1.0,  3.0),
            3: ( 1.0, -3.0),
            4: ( 1.5, -1.0),
        },
    },
    # ── [신규] 고속/불규칙 시나리오 4개 ─────────────────────────────────────────
    {
        "scenario_id": "high_speed_scatter",
        "description": (
            "All UAVs scatter at very high speed (10~12 m/s). Complete "
            "disconnection achieved within t=6s. Tests model performance "
            "on rapid state transitions with minimal degraded window."
        ),
        "mobility": "linear",
        "relay_uav": 2,
        "duration_s": 15.0,
        "time_step_s": 0.25,
        "initial_positions": {i: (100.0, 60.0) for i in range(5)},
        "velocities": {
            0: (10.0,  4.0),
            1: (-10.0,  4.0),
            2: ( 1.0,  0.0),
            3: ( 4.0, -10.0),
            4: (-4.0, -10.0),
        },
    },
    {
        "scenario_id": "asymmetric_scatter",
        "description": (
            "Asymmetric speeds: UAV0 moves very fast (9 m/s), UAV1 moderate "
            "(5 m/s), UAV3 slow (2 m/s), UAV4 stationary. UAV2 is relay. "
            "Creates staggered disconnection timing — UAV0 link fails first, "
            "others degrade gradually."
        ),
        "mobility": "linear",
        "relay_uav": 2,
        "duration_s": 25.0,
        "time_step_s": 0.25,
        "initial_positions": {
            0: (100.0, 60.0),
            1: (100.0, 60.0),
            2: (100.0, 60.0),
            3: (100.0, 60.0),
            4: (100.0, 60.0),
        },
        "velocities": {
            0: ( 9.0,  0.0),
            1: ( 0.0,  5.0),
            2: ( 0.5,  0.0),
            3: (-2.0,  0.0),
            4: ( 0.0,  0.0),
        },
    },
    {
        "scenario_id": "chase_pattern",
        "description": (
            "UAV0 chases UAV4 which is always moving away; UAV1 chases UAV3 "
            "similarly. UAV2 (relay) stays near center. Link quality between "
            "pursuers and targets fluctuates rhythmically around degraded "
            "threshold as spacing changes."
        ),
        "mobility": "phases",
        "relay_uav": 2,
        "duration_s": 40.0,
        "time_step_s": 0.25,
        "phases": [
            {
                "duration_s": 10.0,
                "initial_positions": {
                    0: ( 80.0, 75.0),
                    1: ( 80.0, 45.0),
                    2: (100.0, 60.0),
                    3: (120.0, 45.0),
                    4: (120.0, 75.0),
                },
                "velocities": {
                    0: ( 5.0,  0.0),
                    1: ( 5.0,  0.0),
                    2: ( 0.0,  0.0),
                    3: ( 5.0,  0.0),
                    4: ( 5.0,  0.0),
                },
            },
            {
                "duration_s": 10.0,
                "velocities": {
                    0: ( 5.0,  0.0),
                    1: ( 5.0,  0.0),
                    2: ( 0.0,  0.0),
                    3: (-5.0,  0.0),
                    4: (-5.0,  0.0),
                },
            },
            {
                "duration_s": 10.0,
                "velocities": {
                    0: (-5.0,  0.0),
                    1: (-5.0,  0.0),
                    2: ( 0.0,  0.0),
                    3: (-5.0,  0.0),
                    4: (-5.0,  0.0),
                },
            },
            {
                "duration_s": 10.0,
                "velocities": {
                    0: (-5.0,  0.0),
                    1: (-5.0,  0.0),
                    2: ( 0.0,  0.0),
                    3: ( 5.0,  0.0),
                    4: ( 5.0,  0.0),
                },
            },
        ],
    },
    {
        "scenario_id": "random_waypoint_4",
        "description": (
            "Random Waypoint with seed=777 and wider map range. UAVs can "
            "roam across a larger area (40-200m x 20-110m), producing more "
            "extreme disconnection events than the original 3 RWP scenarios."
        ),
        "mobility": "random_waypoint",
        "relay_uav": 2,
        "duration_s": 40.0,
        "time_step_s": 0.25,
        "seed": 777,
        "x_range": (40.0, 200.0),
        "y_range": (20.0, 110.0),
        "speed_range": (2.0, 10.0),
    },
    # ── UAV4 중심 relay 전환 시나리오 2개 ────────────────────────────────────
    {
        "scenario_id": "relay_uav4_handover",
        "description": (
            "UAV4 starts at center as relay. As UAV4 drifts west and exits "
            "the network, UAV1 moves in from the east to take over as relay "
            "(UAV4→UAV1 at t≈20s). Provides UAV4-as-relay patterns rare in "
            "other scenarios."
        ),
        "mobility": "linear",
        "relay_uav": 4,
        "duration_s": 35.0,
        "time_step_s": 0.25,
        "initial_positions": {
            0: ( 60.0, 90.0),
            1: (160.0, 90.0),
            2: ( 60.0, 30.0),
            3: (160.0, 30.0),
            4: (110.0, 60.0),
        },
        "velocities": {
            0: (0.0,   0.0),
            1: (-5.0, -3.0),
            2: (0.0,   0.0),
            3: (0.0,   0.0),
            4: (-5.0,  0.0),
        },
    },
    {
        "scenario_id": "relay_uav4_rotation",
        "description": (
            "Three-phase baton pass starting with UAV4 at center. "
            "Phase1: UAV4 relay (0~15s). "
            "Phase2: UAV4↔UAV0 swap → UAV0 relay (15~30s). "
            "Phase3: UAV0↔UAV3 swap → UAV3 relay (30~45s). "
            "Provides UAV4→UAV0→UAV3 transition sequence."
        ),
        "mobility": "phases",
        "relay_uav": 4,
        "duration_s": 45.0,
        "time_step_s": 0.25,
        "phases": [
            {
                "duration_s": 15.0,
                "initial_positions": {
                    0: ( 50.0, 110.0),
                    1: (170.0, 110.0),
                    2: ( 50.0,  10.0),
                    3: (170.0,  10.0),
                    4: (110.0,  60.0),
                },
                "velocities": {i: (0.0, 0.0) for i in range(5)},
            },
            {
                "duration_s": 15.0,
                "velocities": {
                    0: ( 60/15, -50/15),
                    1: (0.0,    0.0),
                    2: (0.0,    0.0),
                    3: (0.0,    0.0),
                    4: (-60/15,  50/15),
                },
            },
            {
                "duration_s": 15.0,
                "velocities": {
                    0: ( 60/15, -50/15),
                    1: (0.0,    0.0),
                    2: (0.0,    0.0),
                    3: (-120/15, 50/15),
                    4: (0.0,    0.0),
                },
            },
        ],
    },
]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _build_rwp_trajectory(scenario: dict) -> None:
    """Random Waypoint 이동 궤적을 미리 계산해 scenario['_rwp_traj'], ['_rwp_speed']에 저장."""
    if "_rwp_traj" in scenario:
        return  # 이미 계산됨

    rng = random.Random(scenario["seed"])
    dur = scenario["duration_s"]
    step = scenario["time_step_s"]
    times = [round(i * step, 6) for i in range(int(round(dur / step)) + 1)]
    x_min, x_max = scenario["x_range"]
    y_min, y_max = scenario["y_range"]
    sp_min, sp_max = scenario["speed_range"]

    traj: dict[int, dict[float, tuple]] = {}
    spd_map: dict[int, dict[float, float]] = {}

    for uid in range(NUM_UAVS):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        speed = rng.uniform(sp_min, sp_max)
        wx = rng.uniform(x_min, x_max)
        wy = rng.uniform(y_min, y_max)

        pos_by_t: dict[float, tuple] = {}
        spd_by_t: dict[float, float] = {}

        for t in times:
            dist_to_wp = math.hypot(wx - x, wy - y)
            if dist_to_wp < 0.5:
                wx = rng.uniform(x_min, x_max)
                wy = rng.uniform(y_min, y_max)
                speed = rng.uniform(sp_min, sp_max)
                dist_to_wp = math.hypot(wx - x, wy - y)

            move = speed * step
            if move >= dist_to_wp:
                x, y = wx, wy
            else:
                ratio = move / dist_to_wp
                x += (wx - x) * ratio
                y += (wy - y) * ratio

            pos_by_t[t] = (round(x, 3), round(y, 3))
            spd_by_t[t] = round(speed, 3)

        traj[uid] = pos_by_t
        spd_map[uid] = spd_by_t

    scenario["_rwp_traj"] = traj
    scenario["_rwp_speed"] = spd_map


def _pos(scenario: dict, uav_id: int, time_s: float) -> tuple[float, float]:
    mobility = scenario.get("mobility", "linear")

    if mobility == "random_waypoint":
        return scenario["_rwp_traj"][uav_id].get(time_s, (0.0, 0.0))

    if mobility == "circular":
        cx, cy = scenario["center"]
        orb = scenario["orbits"][uav_id]
        r = orb["radius"]
        if r == 0.0:
            return (cx, cy)
        angle = math.radians(orb["initial_angle_deg"]) + orb["angular_velocity_rad_s"] * time_s
        return (round(cx + r * math.cos(angle), 3), round(cy + r * math.sin(angle), 3))

    if mobility == "phases":
        phases = scenario["phases"]
        x, y = phases[0]["initial_positions"][uav_id]
        t_rem = time_s
        for phase in phases:
            dur = min(t_rem, phase["duration_s"])
            vx, vy = phase["velocities"][uav_id]
            x += vx * dur
            y += vy * dur
            t_rem -= dur
            if t_rem <= 0.0:
                break
        return (round(x, 3), round(y, 3))

    # linear (default)
    x0, y0 = scenario["initial_positions"][uav_id]
    vx, vy = scenario["velocities"][uav_id]
    return (round(x0 + vx * time_s, 3), round(y0 + vy * time_s, 3))


def _speed_at(scenario: dict, uav_id: int, time_s: float) -> float:
    """현재 시간의 UAV 속도 크기 (m/s)."""
    mobility = scenario.get("mobility", "linear")
    if mobility == "random_waypoint":
        return scenario["_rwp_speed"][uav_id].get(time_s, 0.0)
    if mobility == "circular":
        orb = scenario["orbits"][uav_id]
        return round(orb["radius"] * abs(orb["angular_velocity_rad_s"]), 3)
    if mobility == "phases":
        elapsed = 0.0
        for phase in scenario["phases"]:
            elapsed += phase["duration_s"]
            if time_s <= elapsed or phase is scenario["phases"][-1]:
                vx, vy = phase["velocities"][uav_id]
                return round(math.hypot(vx, vy), 3)
        return 0.0
    vx, vy = scenario["velocities"][uav_id]
    return round(math.hypot(vx, vy), 3)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _orientation(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> float:
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])


def _on_segment(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> bool:
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def _segments_intersect(
    p1: tuple[float, float],
    q1: tuple[float, float],
    p2: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)
    if (o1 > 0 > o2 or o1 < 0 < o2) and (o3 > 0 > o4 or o3 < 0 < o4):
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True
    return False


def _point_in_rect(pt: tuple[float, float], r: dict) -> bool:
    return r["x_min_m"] <= pt[0] <= r["x_max_m"] and r["y_min_m"] <= pt[1] <= r["y_max_m"]


def _segment_intersects_rect(
    a: tuple[float, float], b: tuple[float, float], r: dict
) -> bool:
    if _point_in_rect(a, r) or _point_in_rect(b, r):
        return True
    corners = [
        (r["x_min_m"], r["y_min_m"]),
        (r["x_max_m"], r["y_min_m"]),
        (r["x_max_m"], r["y_max_m"]),
        (r["x_min_m"], r["y_max_m"]),
    ]
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return any(_segments_intersect(a, b, e0, e1) for e0, e1 in edges)


def _blocked_buildings(a: tuple[float, float], b: tuple[float, float]) -> list[dict]:
    return [bg for bg in BUILDINGS if _segment_intersects_rect(a, b, bg)]


# ── Radio model ───────────────────────────────────────────────────────────────
# Coarse planning model (not a substitute for ns-3 PHY traces).
#   RSSI  : free-space log-distance with per-building attenuation
#   PLR   : linear in distance + step per building
#   RTT   : linear in distance + step per building
#   Tput  : linear decay with floor

def _link_state(rssi: float, plr: float) -> str:
    """Map (rssi_dBm, plr_%) to a 3-class link state label."""
    if rssi >= THRESHOLDS["healthy_rssi_dbm_min"] and plr <= THRESHOLDS["healthy_plr_pct_max"]:
        return "healthy"
    if rssi >= THRESHOLDS["degraded_rssi_dbm_min"] and plr <= THRESHOLDS["degraded_plr_pct_max"]:
        return "degraded"
    return "disconnected"


def _direct_link(a: tuple[float, float], b: tuple[float, float],
                 rssi_noise: float = 0.0, plr_noise: float = 0.0,
                 dist_noise: float = 0.0) -> dict:
    dist = max(1.0, _dist(a, b) + dist_noise)  # GPS-measured distance (noisy)
    blockers = _blocked_buildings(a, b)
    total_atten = sum(bg["attenuation_db"] for bg in blockers)

    # Ground-truth (noiseless) values — used for link_state label
    rssi_true = -46.0 - 20.0 * math.log10(max(dist, 1.0)) - total_atten
    plr_true  = min(0.8 + max(0.0, dist - 15.0) * 0.24 + len(blockers) * 8.0, 95.0)

    # Measured (noisy) values — used as ML features
    rssi = rssi_true + rssi_noise
    plr  = max(0.0, plr_true + plr_noise)
    rtt = 8.0 + dist * 0.22 + len(blockers) * 12.0
    throughput = max(1.2, 16.0 - dist * 0.23 - len(blockers) * 2.5)
    snr = round(rssi - (-95.0), 3)  # noise floor -95 dBm (IEEE 802.11g)

    return {
        "distance_m": round(dist, 3),
        "blocked_building_count": len(blockers),
        "blocked_building_ids": "|".join(bg["building_id"] for bg in blockers) or "none",
        "blocked_attenuation_db": round(total_atten, 1),
        "rssi_dbm": round(rssi, 3),
        "snr_db": snr,
        "plr_pct": round(plr, 3),
        "rtt_ms": round(rtt, 3),
        "throughput_mbps": round(throughput, 3),
        "state": _link_state(rssi_true, plr_true),  # label = ground truth, not noisy
    }


def _compose_two_hop(first: dict, second: dict) -> dict:
    """Merge two direct-link dicts into end-to-end 2-hop estimates.

    Fix #2: building blockage is derived from *both relay segments*
    (first: src→relay, second: relay→dst), not from the direct src→dst path.
    """
    plr1 = first["plr_pct"] / 100.0
    plr2 = second["plr_pct"] / 100.0
    end_plr = (1.0 - (1.0 - plr1) * (1.0 - plr2)) * 100.0

    rssi = min(first["rssi_dbm"], second["rssi_dbm"]) - 1.5
    snr  = round(rssi - (-95.0), 3)
    rtt = first["rtt_ms"] + second["rtt_ms"] + 4.0
    throughput = max(1.0, min(first["throughput_mbps"], second["throughput_mbps"]) - 0.8)

    # Deduplicated union of buildings that block either relay hop
    ids_1 = set(first["blocked_building_ids"].split("|")) if first["blocked_building_ids"] != "none" else set()
    ids_2 = set(second["blocked_building_ids"].split("|")) if second["blocked_building_ids"] != "none" else set()
    combined = ids_1 | ids_2
    blocked_atten = round(first["blocked_attenuation_db"] + second["blocked_attenuation_db"], 1)

    return {
        "blocked_building_count": len(combined),
        "blocked_building_ids": "|".join(sorted(combined)) or "none",
        "blocked_attenuation_db": blocked_atten,
        "rssi_dbm": round(rssi, 3),
        "snr_db": snr,
        "plr_pct": round(end_plr, 3),
        "rtt_ms": round(rtt, 3),
        "throughput_mbps": round(throughput, 3),
        "state": _link_state(rssi, end_plr),
    }


def _needs_reconfig(m: dict, hop_count: int) -> bool:
    return (
        m["plr_pct"] >= THRESHOLDS["reconfig_trigger_plr_pct"]
        or m["rssi_dbm"] <= THRESHOLDS["reconfig_trigger_rssi_dbm"]
        or m["rtt_ms"] >= THRESHOLDS["reconfig_trigger_rtt_ms"]
        or m["throughput_mbps"] <= THRESHOLDS["reconfig_trigger_throughput_mbps"]
        or hop_count > 2
        or m["state"] == "disconnected"
    )


def _recommended_action(triggered: int) -> str:
    if triggered == 0:
        return "keep_topology"
    if triggered <= 2:
        return "tighten_spacing_near_relay"
    return "promote_autonomous_relay_repositioning"


# ── Per-scenario data generation ─────────────────────────────────────────────

def _time_steps(scenario: dict) -> list[float]:
    dur = scenario["duration_s"]
    step = scenario["time_step_s"]
    n = int(round(dur / step)) + 1
    return [round(i * step, 6) for i in range(n)]


def _generate_scenario(scenario: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (position_rows, link_rows, summary_rows) for one scenario."""
    if scenario.get("mobility") == "random_waypoint":
        _build_rwp_trajectory(scenario)

    sid = scenario["scenario_id"]
    relay = scenario["relay_uav"]
    pos_rows: list[dict] = []
    link_rows: list[dict] = []
    summary_rows: list[dict] = []

    # Pre-generate AR(1) correlated noise per (src, dst) pair
    times = _time_steps(scenario)
    n_steps = len(times)
    pair_seed_base = abs(hash(sid)) % (2**31)
    link_rssi_noise: dict[tuple, list] = {}
    link_plr_noise:  dict[tuple, list] = {}
    link_dist_noise: dict[tuple, list] = {}
    for src in range(NUM_UAVS):
        for dst in range(src + 1, NUM_UAVS):
            seed_r = (pair_seed_base + src * 100 + dst * 10) % (2**31)
            seed_p = (pair_seed_base + src * 100 + dst * 10 + 1) % (2**31)
            seed_d = (pair_seed_base + src * 100 + dst * 10 + 2) % (2**31)
            link_rssi_noise[(src, dst)] = _gen_ar1_noise(n_steps, RSSI_NOISE_STD, NOISE_CORR, seed_r)
            link_plr_noise[(src, dst)]  = _gen_ar1_noise(n_steps, PLR_NOISE_STD,  PLR_NOISE_CORR, seed_p)
            link_dist_noise[(src, dst)] = _gen_ar1_noise(n_steps, GPS_NOISE_STD,  0.85, seed_d)

    for t_idx, t in enumerate(times):
        positions = {uid: _pos(scenario, uid, t) for uid in range(NUM_UAVS)}

        # UAV position records
        for uid in range(NUM_UAVS):
            x, y = positions[uid]
            pos_rows.append({
                "scenario_id": sid,
                "time_s": t,
                "uav_id": uid,
                "x_m": x,
                "y_m": y,
                "speed_mps": _speed_at(scenario, uid, t),
                "role": "relay_anchor" if uid == relay else "peripheral",
            })

        # 최적 relay_uav 계산 (평균 RSSI 기준, 노이즈 없는 기준값 사용)
        def _avg_rssi(uid: int) -> float:
            others = [j for j in range(NUM_UAVS) if j != uid]
            return sum(
                -46.0 - 20.0 * math.log10(max(_dist(positions[uid], positions[j]), 1.0))
                for j in others
            ) / len(others)
        optimal_relay = max(range(NUM_UAVS), key=_avg_rssi)

        # Link metrics for every ordered pair (src < dst)
        pair_rows: list[dict] = []
        for src in range(NUM_UAVS):
            for dst in range(src + 1, NUM_UAVS):
                rn = link_rssi_noise[(src, dst)][t_idx]
                pn = link_plr_noise[(src, dst)][t_idx]
                dn = link_dist_noise[(src, dst)][t_idx]
                direct = _direct_link(positions[src], positions[dst], rn, pn, dn)

                # Start with direct link info; may be overwritten by relay below.
                m = dict(direct)
                route = "direct"

                # Fix #1: hop_count based on link state, not a distance threshold.
                if direct["state"] != "disconnected":
                    hop_count = 1
                else:
                    hop_count = 0  # no usable route yet
                    # Try single relay via relay_uav (only when neither endpoint IS the relay)
                    if src != relay and dst != relay:
                        k_sr = (min(src,relay), max(src,relay))
                        k_rd = (min(relay,dst), max(relay,dst))
                        first  = _direct_link(positions[src],   positions[relay],
                                              link_rssi_noise.get(k_sr,[0]*n_steps)[t_idx],
                                              link_plr_noise.get(k_sr,[0]*n_steps)[t_idx],
                                              link_dist_noise.get(k_sr,[0]*n_steps)[t_idx])
                        second = _direct_link(positions[relay], positions[dst],
                                              link_rssi_noise.get(k_rd,[0]*n_steps)[t_idx],
                                              link_plr_noise.get(k_rd,[0]*n_steps)[t_idx],
                                              link_dist_noise.get(k_rd,[0]*n_steps)[t_idx])
                        if first["state"] != "disconnected" and second["state"] != "disconnected":
                            hop_count = 2
                            route = f"via_uav{relay}"
                            two_hop = _compose_two_hop(first, second)
                            # Fix #2: overwrite blocking info with relay-segment data.
                            m.update(two_hop)
                            # distance_m stays as Euclidean src→dst (intentional).

                trigger = _needs_reconfig(m, hop_count)
                row = {
                    "scenario_id": sid,
                    "time_s": t,
                    "src_uav": src,
                    "dst_uav": dst,
                    "distance_m": m["distance_m"],
                    "hop_count": hop_count,
                    "route_type": route,
                    "blocked_building_count": m["blocked_building_count"],
                    "blocked_building_ids": m["blocked_building_ids"],
                    "blocked_attenuation_db": m["blocked_attenuation_db"],
                    "rssi_dbm_est": m["rssi_dbm"],
                    "snr_db_est": m["snr_db"],
                    "plr_pct_est": m["plr_pct"],
                    "rtt_ms_est": m["rtt_ms"],
                    "throughput_mbps_est": m["throughput_mbps"],
                    "link_state": m["state"],
                    "reconfig_trigger": "yes" if trigger else "no",
                    "optimal_relay_uav": optimal_relay,
                }
                link_rows.append(row)
                pair_rows.append(row)

        # Topology summary for this time step
        connected = all(r["hop_count"] in (1, 2) for r in pair_rows)
        max_hops = max(r["hop_count"] for r in pair_rows)
        avg_rssi = sum(r["rssi_dbm_est"] for r in pair_rows) / len(pair_rows)
        avg_plr = sum(r["plr_pct_est"] for r in pair_rows) / len(pair_rows)
        triggered = sum(1 for r in pair_rows if r["reconfig_trigger"] == "yes")
        summary_rows.append({
            "scenario_id": sid,
            "time_s": t,
            "connected": "yes" if connected else "no",
            "max_pair_hops": max_hops,
            "triggered_link_count": triggered,
            "avg_rssi_dbm_est": round(avg_rssi, 3),
            "avg_plr_pct_est": round(avg_plr, 3),
            "topology_goal_met": "yes" if connected and max_hops <= 2 else "no",
            "recommended_action": _recommended_action(triggered),
        })

    return pos_rows, link_rows, summary_rows


# ── File writers ──────────────────────────────────────────────────────────────

def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_manifest() -> None:
    payload = {
        "generated_by": "generate_uav_2d_dataset.py (fixed v2)",
        "fixes": [
            "hop_count based on link state, not 30 m distance threshold (Fix #1)",
            "blocked_building fields for 2-hop routes use relay-segment data (Fix #2)",
            "three scenarios include disconnected links for AI training (Fix #3)",
            "time resolution 0.5 s, duration 18-60 s → ~2 000 rows (Fix #4)",
        ],
        "scenarios": [
            {k: v for k, v in sc.items() if k not in ("initial_positions", "velocities")}
            for sc in SCENARIOS
        ],
        "thresholds": THRESHOLDS,
        "heuristics": [
            "Prefer relay_uav as the default relay anchor in the 5-UAV baseline.",
            "If a link crosses a trigger threshold, move the farthest peripheral UAV 5-10 m toward relay.",
            "If the same relay-adjacent link is triggered for ≥2 consecutive steps, prepare relay reassignment.",
            "Treat blocked LOS plus PLR/RSSI degradation as a stronger trigger than distance alone.",
        ],
    }
    with (OUT_DIR / "scenario_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=str)
        fp.write("\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pos: list[dict] = []
    all_links: list[dict] = []
    all_summary: list[dict] = []

    print("Generating scenarios…")
    for sc in SCENARIOS:
        pos_rows, link_rows, summary_rows = _generate_scenario(sc)
        all_pos.extend(pos_rows)
        all_links.extend(link_rows)
        all_summary.extend(summary_rows)
        states = [r["link_state"] for r in link_rows]
        print(
            f"  {sc['scenario_id']:25s}  "
            f"{len(link_rows):5d} link rows  "
            f"disconnected: {states.count('disconnected'):4d}  "
            f"degraded: {states.count('degraded'):4d}  "
            f"healthy: {states.count('healthy'):4d}"
        )

    write_manifest()

    _write_csv(
        OUT_DIR / "obstacles.csv",
        ["building_id", "kind", "x_min_m", "x_max_m", "y_min_m", "y_max_m", "attenuation_db", "note"],
        BUILDINGS,
    )
    _write_csv(
        OUT_DIR / "uav_positions.csv",
        ["scenario_id", "time_s", "uav_id", "x_m", "y_m", "speed_mps", "role"],
        all_pos,
    )
    _write_csv(
        OUT_DIR / "link_metrics.csv",
        [
            "scenario_id", "time_s", "src_uav", "dst_uav", "distance_m",
            "hop_count", "route_type",
            "blocked_building_count", "blocked_building_ids", "blocked_attenuation_db",
            "rssi_dbm_est", "snr_db_est", "plr_pct_est", "rtt_ms_est", "throughput_mbps_est",
            "link_state", "reconfig_trigger", "optimal_relay_uav",
        ],
        all_links,
    )
    _write_csv(
        OUT_DIR / "topology_summary.csv",
        [
            "scenario_id", "time_s", "connected", "max_pair_hops",
            "triggered_link_count", "avg_rssi_dbm_est", "avg_plr_pct_est",
            "topology_goal_met", "recommended_action",
        ],
        all_summary,
    )

    total = len(all_links)
    disc = sum(1 for r in all_links if r["link_state"] == "disconnected")
    deg = sum(1 for r in all_links if r["link_state"] == "degraded")
    hlth = sum(1 for r in all_links if r["link_state"] == "healthy")
    print(f"\nTotal link rows : {total}")
    print(f"  healthy       : {hlth:5d}  ({100*hlth/total:.1f}%)")
    print(f"  degraded      : {deg:5d}  ({100*deg/total:.1f}%)")
    print(f"  disconnected  : {disc:5d}  ({100*disc/total:.1f}%)")
    print(f"\nDataset written to {OUT_DIR}")


if __name__ == "__main__":
    main()
