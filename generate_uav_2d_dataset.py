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
from pathlib import Path

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
        "time_step_s": 0.5,
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
        "time_step_s": 0.5,
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
        "time_step_s": 0.5,
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
        "time_step_s": 0.5,
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
        "time_step_s": 0.5,
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
        "time_step_s": 0.5,
        "center": (100.0, 60.0),
        "orbits": {
            0: {"radius": 50.0, "initial_angle_deg":  90.0, "angular_velocity_rad_s": 0.2},
            1: {"radius": 50.0, "initial_angle_deg": 270.0, "angular_velocity_rad_s": 0.2},
            2: {"radius":  0.0, "initial_angle_deg":   0.0, "angular_velocity_rad_s": 0.0},
            3: {"radius": 50.0, "initial_angle_deg":   0.0, "angular_velocity_rad_s": 0.2},
            4: {"radius": 50.0, "initial_angle_deg": 180.0, "angular_velocity_rad_s": 0.2},
        },
    },
]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _pos(scenario: dict, uav_id: int, time_s: float) -> tuple[float, float]:
    mobility = scenario.get("mobility", "linear")

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


def _direct_link(a: tuple[float, float], b: tuple[float, float]) -> dict:
    dist = _dist(a, b)
    blockers = _blocked_buildings(a, b)
    total_atten = sum(bg["attenuation_db"] for bg in blockers)

    rssi = -46.0 - 20.0 * math.log10(max(dist, 1.0)) - total_atten
    plr = min(0.8 + max(0.0, dist - 15.0) * 0.24 + len(blockers) * 8.0, 95.0)
    rtt = 8.0 + dist * 0.22 + len(blockers) * 12.0
    throughput = max(1.2, 16.0 - dist * 0.23 - len(blockers) * 2.5)

    return {
        "distance_m": round(dist, 3),
        "blocked_building_count": len(blockers),
        "blocked_building_ids": "|".join(bg["building_id"] for bg in blockers) or "none",
        "blocked_attenuation_db": round(total_atten, 1),
        "rssi_dbm": round(rssi, 3),
        "plr_pct": round(plr, 3),
        "rtt_ms": round(rtt, 3),
        "throughput_mbps": round(throughput, 3),
        "state": _link_state(rssi, plr),
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
    sid = scenario["scenario_id"]
    relay = scenario["relay_uav"]
    pos_rows: list[dict] = []
    link_rows: list[dict] = []
    summary_rows: list[dict] = []

    for t in _time_steps(scenario):
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

        # Link metrics for every ordered pair (src < dst)
        pair_rows: list[dict] = []
        for src in range(NUM_UAVS):
            for dst in range(src + 1, NUM_UAVS):
                direct = _direct_link(positions[src], positions[dst])

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
                        first = _direct_link(positions[src], positions[relay])
                        second = _direct_link(positions[relay], positions[dst])
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
                    "plr_pct_est": m["plr_pct"],
                    "rtt_ms_est": m["rtt_ms"],
                    "throughput_mbps_est": m["throughput_mbps"],
                    "link_state": m["state"],
                    "reconfig_trigger": "yes" if trigger else "no",
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
            "rssi_dbm_est", "plr_pct_est", "rtt_ms_est", "throughput_mbps_est",
            "link_state", "reconfig_trigger",
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
