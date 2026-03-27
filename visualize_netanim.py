#!/usr/bin/env python3
"""
visualize_netanim.py
─────────────────────
NS-3 uav-animation.xml + ns3_link_metrics.csv를 읽어
relay_stretch 시나리오의 UAV 이동 및 링크 상태를 애니메이션으로 시각화합니다.

실행:
    python3 visualize_netanim.py              # GIF 저장
    python3 visualize_netanim.py --show       # 화면에 표시
"""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT      = Path(__file__).resolve().parent
XML_PATH  = ROOT / "ns-3.47" / "uav-animation.xml"
CSV_PATH  = ROOT / "ns-3.47" / "ns3_link_metrics.csv"
GIF_OUT   = ROOT / "ns-3.47" / "uav-animation.gif"

# ── 건물 정보 (Python 데이터셋과 동일) ────────────────────────────────────────
BUILDINGS = [
    {"id": "B0", "x_min": 88, "x_max": 100, "y_min": 52, "y_max": 58},
    {"id": "B1", "x_min": 118, "x_max": 132, "y_min": 62, "y_max": 68},
    {"id": "B2", "x_min": 146, "x_max": 160, "y_min": 50, "y_max": 56},
]

# ── 색상 설정 ─────────────────────────────────────────────────────────────────
NODE_COLORS = {
    0: "#2196F3",   # UAV0 파랑 (동쪽)
    1: "#03A9F4",   # UAV1 하늘 (동쪽)
    2: "#4CAF50",   # UAV2 초록 (중계 앵커)
    3: "#F44336",   # UAV3 빨강 (서쪽)
    4: "#FF5722",   # UAV4 주황 (서쪽)
}
LINK_COLORS = {
    "healthy":      "#00C853",   # 초록
    "degraded":     "#FFD600",   # 노랑
    "disconnected": "#B0BEC5",   # 회색 (점선)
    "unknown":      "#ECEFF1",
}
LINE_STYLES = {
    "healthy":      "-",
    "degraded":     "--",
    "disconnected": ":",
    "unknown":      ":",
}
LINE_ALPHA = {
    "healthy":      0.9,
    "degraded":     0.7,
    "disconnected": 0.25,
    "unknown":      0.1,
}


# ── XML 파싱: 시간별 UAV 위치 ─────────────────────────────────────────────────
def parse_positions(xml_path: Path) -> tuple[dict, dict[float, dict[int, tuple]]]:
    """
    Returns:
        init_pos: {node_id: (x, y)}
        timed_pos: {time: {node_id: (x, y)}}  (snapshot every LOG_STEP_S)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 초기 위치
    init_pos: dict[int, tuple] = {}
    for node in root.findall("node"):
        nid = int(node.attrib["id"])
        x   = float(node.attrib["locX"])
        y   = float(node.attrib["locY"])
        init_pos[nid] = (x, y)

    # 위치 업데이트 (p="p" → position)
    raw_updates: dict[float, dict[int, tuple]] = defaultdict(dict)
    for nu in root.findall("nu"):
        if nu.attrib.get("p") == "p":
            t   = round(float(nu.attrib["t"]), 3)
            nid = int(nu.attrib["id"])
            x   = float(nu.attrib["x"])
            y   = float(nu.attrib["y"])
            raw_updates[t][nid] = (x, y)

    # 0.5s 간격 스냅샷 생성 (중간 업데이트는 보간)
    all_times = sorted(raw_updates)
    max_t = max(all_times) if all_times else 20.0

    snapshots: dict[float, dict[int, tuple]] = {}
    # 누적 위치 추적
    current: dict[int, tuple] = dict(init_pos)
    prev_t = 0.0

    for t in sorted(set(all_times)):
        for nid, pos in raw_updates[t].items():
            current[nid] = pos
        t_snap = round(t, 1)
        if t_snap not in snapshots:
            snapshots[t_snap] = dict(current)

    # t=0 초기 스냅샷 추가
    if 0.0 not in snapshots:
        snapshots[0.0] = dict(init_pos)

    return init_pos, snapshots


# ── CSV 파싱: 시간별 링크 상태 ────────────────────────────────────────────────
def parse_link_states(csv_path: Path) -> dict[float, dict[tuple, str]]:
    """Returns {time: {(src, dst): link_state}}"""
    result: dict[float, dict[tuple, str]] = defaultdict(dict)
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t   = round(float(row["time_s"]), 1)
            src = int(row["src_uav"])
            dst = int(row["dst_uav"])
            result[t][(src, dst)] = row["link_state_ns3"]
    return result


# ── 메인 시각화 ───────────────────────────────────────────────────────────────
def main(show: bool = False) -> None:
    if not XML_PATH.exists():
        sys.exit(f"[오류] {XML_PATH} 없음. NS-3 시뮬레이션을 먼저 실행하세요.")
    if not CSV_PATH.exists():
        sys.exit(f"[오류] {CSV_PATH} 없음.")

    print("XML 파싱 중…")
    init_pos, pos_snapshots = parse_positions(XML_PATH)
    print(f"  위치 스냅샷: {len(pos_snapshots)}개 시간점")

    print("링크 상태 CSV 파싱 중…")
    link_states = parse_link_states(CSV_PATH)

    # 공통 시간축 (0.5s 간격)
    snap_times = sorted(pos_snapshots)
    N = len(snap_times)

    # ── 그림 설정 ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(40, 230)
    ax.set_ylim(30, 90)
    ax.set_aspect("equal")
    ax.set_facecolor("#1A1A2E")
    fig.patch.set_facecolor("#0F0F1A")
    ax.set_xlabel("X (m)", color="white", fontsize=10)
    ax.set_ylabel("Y (m)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(color="#333", linewidth=0.5, linestyle="--", alpha=0.5)

    # 건물 그리기
    for bld in BUILDINGS:
        rect = Rectangle(
            (bld["x_min"], bld["y_min"]),
            bld["x_max"] - bld["x_min"],
            bld["y_max"] - bld["y_min"],
            linewidth=1.5,
            edgecolor="#FF9800",
            facecolor="#3E2723",
            alpha=0.85,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            (bld["x_min"] + bld["x_max"]) / 2,
            (bld["y_min"] + bld["y_max"]) / 2,
            bld["id"],
            color="#FF9800",
            fontsize=7,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=3,
        )

    # UAV 노드 (scatter)
    num_uavs = len(init_pos)
    node_ids  = sorted(init_pos)
    xs = [init_pos[i][0] for i in node_ids]
    ys = [init_pos[i][1] for i in node_ids]
    colors = [NODE_COLORS[i] for i in node_ids]
    scatters = ax.scatter(xs, ys, s=180, c=colors, zorder=5, edgecolors="white", linewidths=1.5)

    # UAV 레이블
    labels_txt = ["UAV0\n(East)", "UAV1\n(East)", "UAV2\n(Relay)", "UAV3\n(West)", "UAV4\n(West)"]
    node_texts = []
    for i, nid in enumerate(node_ids):
        txt = ax.text(
            xs[i] + 1.5, ys[i] + 2.5,
            labels_txt[nid],
            color=NODE_COLORS[nid],
            fontsize=7.5,
            fontweight="bold",
            zorder=6,
        )
        node_texts.append(txt)

    # 궤적선 (trail) per node
    trail_lines = {}
    trail_x: dict[int, list] = {nid: [init_pos[nid][0]] for nid in node_ids}
    trail_y: dict[int, list] = {nid: [init_pos[nid][1]] for nid in node_ids}
    for nid in node_ids:
        (line,) = ax.plot([], [], color=NODE_COLORS[nid], lw=1, alpha=0.35, zorder=3)
        trail_lines[nid] = line

    # 링크선 (10쌍)
    pairs = [(s, d) for s in node_ids for d in node_ids if s < d]
    link_lines = {}
    for pair in pairs:
        (line,) = ax.plot([], [], lw=1.5, zorder=4)
        link_lines[pair] = line

    # 시간 텍스트
    time_txt = ax.text(
        0.02, 0.96, "t = 0.0 s",
        transform=ax.transAxes,
        color="white",
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    # 범례
    legend_handles = [
        mpatches.Patch(color=LINK_COLORS["healthy"],      label="Healthy"),
        mpatches.Patch(color=LINK_COLORS["degraded"],     label="Degraded"),
        mpatches.Patch(color=LINK_COLORS["disconnected"], label="Disconnected"),
        mpatches.Patch(color="#3E2723", edgecolor="#FF9800", label="Building"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              facecolor="#1A1A2E", edgecolor="#444", labelcolor="white", fontsize=8)

    ax.set_title("relay_stretch Scenario — NS-3 UAV Movement & Link State",
                 color="white", fontsize=11, pad=10)

    # ── 애니메이션 업데이트 함수 ────────────────────────────────────────────
    def update(frame_idx: int):
        t = snap_times[frame_idx]
        pos = pos_snapshots.get(t, {})

        # 노드 위치 업데이트
        new_xs = []
        new_ys = []
        for i, nid in enumerate(node_ids):
            x, y = pos.get(nid, init_pos[nid])
            new_xs.append(x)
            new_ys.append(y)
            node_texts[i].set_position((x + 1.5, y + 2.5))

            # 궤적 업데이트
            trail_x[nid].append(x)
            trail_y[nid].append(y)
            max_trail = 20  # 최근 20 프레임까지만 표시
            trail_lines[nid].set_data(trail_x[nid][-max_trail:], trail_y[nid][-max_trail:])

        scatters.set_offsets(list(zip(new_xs, new_ys)))

        # 링크 상태 업데이트
        ls = link_states.get(t, {})
        for (src, dst), line in link_lines.items():
            x0, y0 = pos.get(src, init_pos[src])
            x1, y1 = pos.get(dst, init_pos[dst])
            state = ls.get((src, dst), "unknown")
            line.set_data([x0, x1], [y0, y1])
            line.set_color(LINK_COLORS[state])
            line.set_linestyle(LINE_STYLES[state])
            line.set_alpha(LINE_ALPHA[state])
            line.set_linewidth(1.8 if state == "healthy" else 1.2)

        time_txt.set_text(f"t = {t:.1f} s")
        return list(trail_lines.values()) + list(link_lines.values()) + [scatters, time_txt] + node_texts

    anim = FuncAnimation(
        fig, update,
        frames=N,
        interval=100,   # 100ms per frame → 10fps (20s 시뮬 = 2s 영상)
        blit=False,
    )

    if show:
        plt.tight_layout()
        plt.show()
    else:
        print(f"GIF 저장 중… ({N} 프레임, 약 10fps)")
        writer = PillowWriter(fps=10)
        anim.save(str(GIF_OUT), writer=writer, dpi=120)
        print(f"저장 완료: {GIF_OUT}")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="화면에 표시 (GIF 저장 대신)")
    args = parser.parse_args()
    main(show=args.show)
