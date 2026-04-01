#!/usr/bin/env python3
"""
visualize_all_scenarios.py
───────────────────────────
Python 데이터셋(uav_positions.csv + link_metrics.csv)을 읽어
6개 전체 시나리오를 각각 GIF로 저장합니다.

실행:
    python3 visualize_all_scenarios.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "ns-3.47" / "scenario_gifs"

BUILDINGS = [
    {"id": "B0", "x_min": 88,  "x_max": 100, "y_min": 52, "y_max": 58},
    {"id": "B1", "x_min": 118, "x_max": 132, "y_min": 62, "y_max": 68},
    {"id": "B2", "x_min": 146, "x_max": 160, "y_min": 50, "y_max": 56},
]

NODE_COLORS = {
    0: "#2196F3", 1: "#03A9F4", 2: "#4CAF50", 3: "#F44336", 4: "#FF5722"
}
NODE_LABELS = {
    0: "UAV0", 1: "UAV1", 2: "UAV2\n(Relay)", 3: "UAV3", 4: "UAV4"
}
LINK_COLOR  = {"healthy": "#00C853", "degraded": "#FFD600", "disconnected": "#546E7A"}
LINK_STYLE  = {"healthy": "-",       "degraded": "--",      "disconnected": ":"}
LINK_ALPHA  = {"healthy": 0.9,       "degraded": 0.7,       "disconnected": 0.2}
LINK_WIDTH  = {"healthy": 2.0,       "degraded": 1.4,       "disconnected": 0.8}


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_positions(path: Path) -> dict[str, dict[float, dict[int, tuple]]]:
    """scenario_id → time → uav_id → (x, y)"""
    result: dict = defaultdict(lambda: defaultdict(dict))
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["scenario_id"]
            t   = round(float(row["time_s"]), 6)
            uid = int(row["uav_id"])
            result[sid][t][uid] = (float(row["x_m"]), float(row["y_m"]))
    return {k: dict(v) for k, v in result.items()}


def load_links(path: Path) -> dict[str, dict[float, dict[tuple, str]]]:
    """scenario_id → time → (src,dst) → link_state"""
    result: dict = defaultdict(lambda: defaultdict(dict))
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid  = row["scenario_id"]
            t    = round(float(row["time_s"]), 6)
            pair = (int(row["src_uav"]), int(row["dst_uav"]))
            result[sid][t][pair] = row["link_state"]
    return {k: dict(v) for k, v in result.items()}


# ── 단일 시나리오 GIF 생성 ───────────────────────────────────────────────────
def make_gif(sid: str,
             pos_data: dict[float, dict[int, tuple]],
             link_data: dict[float, dict[tuple, str]],
             out_path: Path) -> None:

    times     = sorted(pos_data)
    num_uavs  = 5
    pairs     = [(s, d) for s in range(num_uavs) for d in range(s + 1, num_uavs)]

    # 맵 범위 자동 계산
    all_x = [p[0] for t in pos_data.values() for p in t.values()]
    all_y = [p[1] for t in pos_data.values() for p in t.values()]
    pad = 20
    xmin, xmax = min(all_x) - pad, max(all_x) + pad
    ymin, ymax = min(all_y) - pad, max(all_y) + pad
    # 건물 범위도 포함
    for b in BUILDINGS:
        xmin = min(xmin, b["x_min"] - pad)
        xmax = max(xmax, b["x_max"] + pad)
        ymin = min(ymin, b["y_min"] - pad)
        ymax = max(ymax, b["y_max"] + pad)

    # 가로 세로 비율 유지
    width_m  = xmax - xmin
    height_m = ymax - ymin
    fig_w = 12
    fig_h = max(4.5, fig_w * height_m / width_m)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_facecolor("#1A1A2E")
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_xlabel("X (m)", color="white", fontsize=9)
    ax.set_ylabel("Y (m)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(color="#333", linewidth=0.4, linestyle="--", alpha=0.5)

    # 건물
    for b in BUILDINGS:
        bx_min, bx_max = b["x_min"], b["x_max"]
        by_min, by_max = b["y_min"], b["y_max"]
        if bx_max < xmin or bx_min > xmax or by_max < ymin or by_min > ymax:
            continue
        ax.add_patch(Rectangle(
            (bx_min, by_min), bx_max - bx_min, by_max - by_min,
            lw=1.5, edgecolor="#FF9800", facecolor="#3E2723", alpha=0.85, zorder=2))
        ax.text((bx_min + bx_max) / 2, (by_min + by_max) / 2, b["id"],
                color="#FF9800", fontsize=7, ha="center", va="center",
                fontweight="bold", zorder=3)

    # 초기 위치
    init_pos = pos_data[times[0]]
    xs0 = [init_pos.get(i, (0, 0))[0] for i in range(num_uavs)]
    ys0 = [init_pos.get(i, (0, 0))[1] for i in range(num_uavs)]
    scat = ax.scatter(xs0, ys0, s=160, c=[NODE_COLORS[i] for i in range(num_uavs)],
                      zorder=5, edgecolors="white", linewidths=1.2)

    node_texts = []
    for i in range(num_uavs):
        txt = ax.text(xs0[i] + 1.5, ys0[i] + 2.5,
                      NODE_LABELS[i], color=NODE_COLORS[i],
                      fontsize=7, fontweight="bold", zorder=6)
        node_texts.append(txt)

    # 궤적
    trail_x = {i: [xs0[i]] for i in range(num_uavs)}
    trail_y = {i: [ys0[i]] for i in range(num_uavs)}
    trail_lines = {}
    for i in range(num_uavs):
        (ln,) = ax.plot([], [], color=NODE_COLORS[i], lw=0.8, alpha=0.3, zorder=3)
        trail_lines[i] = ln

    # 링크선
    link_lines = {}
    for pair in pairs:
        (ln,) = ax.plot([], [], zorder=4)
        link_lines[pair] = ln

    # 상태 카운트 텍스트
    count_txt = ax.text(0.02, 0.05, "", transform=ax.transAxes,
                        color="white", fontsize=8, va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A1A2E",
                                  edgecolor="#444", alpha=0.8))
    time_txt = ax.text(0.02, 0.96, "t = 0.0 s", transform=ax.transAxes,
                       color="white", fontsize=11, fontweight="bold", va="top")

    # 범례
    legend_handles = [
        mpatches.Patch(color=LINK_COLOR["healthy"],      label="Healthy"),
        mpatches.Patch(color=LINK_COLOR["degraded"],     label="Degraded"),
        mpatches.Patch(color=LINK_COLOR["disconnected"], label="Disconnected"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              facecolor="#1A1A2E", edgecolor="#444",
              labelcolor="white", fontsize=8)
    ax.set_title(f"Scenario: {sid}  —  UAV Link State Animation",
                 color="white", fontsize=10, pad=8)

    def update(frame_idx: int):
        t   = times[frame_idx]
        pos = pos_data.get(t, init_pos)
        ls  = link_data.get(t, {})

        new_xs, new_ys = [], []
        for i in range(num_uavs):
            x, y = pos.get(i, init_pos.get(i, (0, 0)))
            new_xs.append(x)
            new_ys.append(y)
            node_texts[i].set_position((x + 1.5, y + 2.5))
            trail_x[i].append(x)
            trail_y[i].append(y)
            trail_lines[i].set_data(trail_x[i][-30:], trail_y[i][-30:])

        scat.set_offsets(list(zip(new_xs, new_ys)))

        counts = {"healthy": 0, "degraded": 0, "disconnected": 0}
        for pair, ln in link_lines.items():
            x0, y0 = pos.get(pair[0], (0, 0))
            x1, y1 = pos.get(pair[1], (0, 0))
            state = ls.get(pair, "disconnected")
            ln.set_data([x0, x1], [y0, y1])
            ln.set_color(LINK_COLOR[state])
            ln.set_linestyle(LINK_STYLE[state])
            ln.set_alpha(LINK_ALPHA[state])
            ln.set_linewidth(LINK_WIDTH[state])
            counts[state] += 1

        time_txt.set_text(f"t = {t:.1f} s")
        count_txt.set_text(
            f"H:{counts['healthy']}  D:{counts['degraded']}  X:{counts['disconnected']}")
        return (list(trail_lines.values()) + list(link_lines.values()) +
                [scat, time_txt, count_txt] + node_texts)

    anim = FuncAnimation(fig, update, frames=len(times),
                         interval=120, blit=False)
    print(f"  [{sid}] {len(times)} 프레임 GIF 저장 중…")
    anim.save(str(out_path), writer=PillowWriter(fps=8), dpi=110)
    plt.close(fig)
    print(f"  → {out_path.name} 저장 완료")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("데이터 로딩 중…")
    all_pos  = load_positions(DATA_DIR / "uav_positions.csv")
    all_link = load_links(DATA_DIR / "link_metrics.csv")

    scenarios = list(all_pos.keys())
    print(f"시나리오 {len(scenarios)}개: {scenarios}\n")

    for sid in scenarios:
        out_path = OUT_DIR / f"anim_{sid}.gif"
        make_gif(sid, all_pos[sid], all_link[sid], out_path)

    print(f"\n완료! 저장 위치: {OUT_DIR}")
    print("파일 목록:")
    for sid in scenarios:
        f = OUT_DIR / f"anim_{sid}.gif"
        size_kb = f.stat().st_size // 1024
        print(f"  anim_{sid}.gif  ({size_kb} KB)")


if __name__ == "__main__":
    main()
