#!/usr/bin/env python3
"""
animate_scenario.py
───────────────────
시나리오별 타임스텝 애니메이션 생성

출력:
  ① models/animation_<scenario>.html  — Plotly 인터랙티브 애니메이션 (Play 버튼)
  ② models/animation_<scenario>.gif   — matplotlib GIF

포함 요소:
  - UAV 위치 (역할별 색상)
  - 링크 상태 (healthy=초록, degraded=노랑, disconnected=빨강)
  - 건물 장애물 (회색 사각형, 감쇠 dB 표시)

실행:
    python3 animate_scenario.py                     # 기본: relay_failure
    python3 animate_scenario.py --scenario orbit    # 시나리오 지정
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import plotly.graph_objects as go

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"
OUT_DIR  = ROOT / "models"

STATE_COLOR_PLOTLY = {
    "healthy":      "#2ecc71",
    "degraded":     "#f39c12",
    "disconnected": "#e74c3c",
}
STATE_COLOR_MPL = {
    "healthy":      "green",
    "degraded":     "orange",
    "disconnected": "red",
}
ROLE_COLOR = {
    "relay":  "#e74c3c",
    "source": "#3498db",
    "sink":   "#9b59b6",
    "":       "#3498db",
}


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_obstacles() -> list[dict]:
    rows = []
    with open(DATA_DIR / "obstacles.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "id":    r["building_id"],
                "x_min": float(r["x_min_m"]), "x_max": float(r["x_max_m"]),
                "y_min": float(r["y_min_m"]), "y_max": float(r["y_max_m"]),
                "atten": float(r["attenuation_db"]),
            })
    return rows


def load_scenario(scenario: str):
    positions: dict[str, dict[int, tuple]] = {}
    with open(DATA_DIR / "uav_positions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scenario_id"] != scenario:
                continue
            t = r["time_s"]
            if t not in positions:
                positions[t] = {}
            positions[t][int(r["uav_id"])] = (
                float(r["x_m"]), float(r["y_m"]), r.get("role", "")
            )

    links: dict[str, dict[tuple, dict]] = {}
    with open(DATA_DIR / "link_metrics.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scenario_id"] != scenario:
                continue
            t    = r["time_s"]
            pair = (int(r["src_uav"]), int(r["dst_uav"]))
            if t not in links:
                links[t] = {}
            links[t][pair] = {
                "state": r["link_state"],
                "rssi":  float(r["rssi_dbm_est"]),
                "plr":   float(r["plr_pct_est"]),
                "relay": int(r["optimal_relay_uav"]),
            }

    timesteps = sorted(positions.keys(), key=float)
    return timesteps, positions, links


# ── Plotly HTML 애니메이션 ────────────────────────────────────────────────────
def make_plotly_animation(scenario: str, timesteps, positions, links, obstacles):
    # ── 건물 트레이스 (정적) ────────────────────────────────────────────────
    bld_traces = []
    for b in obstacles:
        xs = [b["x_min"], b["x_max"], b["x_max"], b["x_min"], b["x_min"]]
        ys = [b["y_min"], b["y_min"], b["y_max"], b["y_max"], b["y_min"]]
        bld_traces.append(go.Scatter(
            x=xs, y=ys, fill="toself",
            fillcolor="rgba(150,150,150,0.35)",
            line=dict(color="#555", width=1.2),
            mode="lines",
            name=f"{b['id']} ({b['atten']}dB)",
            hovertemplate=f"<b>{b['id']}</b><br>감쇠: {b['atten']}dB<extra></extra>",
            legendgroup="buildings",
        ))

    n_bld = len(bld_traces)

    def make_frame_traces(t_s):
        pos  = positions.get(t_s, {})
        lnks = links.get(t_s, {})
        traces = []

        # 링크
        for (src, dst), info in lnks.items():
            if src not in pos or dst not in pos:
                continue
            x0, y0, _ = pos[src]
            x1, y1, _ = pos[dst]
            color = STATE_COLOR_PLOTLY[info["state"]]
            traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="skip",
                showlegend=False,
            ))

        # UAV
        uav_ids = sorted(pos)
        traces.append(go.Scatter(
            x=[pos[u][0] for u in uav_ids],
            y=[pos[u][1] for u in uav_ids],
            mode="markers+text",
            marker=dict(
                size=20,
                color=[ROLE_COLOR.get(pos[u][2], "#3498db") for u in uav_ids],
                line=dict(color="white", width=2),
            ),
            text=[f"UAV{u}" for u in uav_ids],
            textposition="top center",
            textfont=dict(size=11, color="#2c3e50"),
            hovertemplate=[
                f"<b>UAV{u}</b><br>role: {pos[u][2]}<br>({pos[u][0]:.1f}, {pos[u][1]:.1f})<extra></extra>"
                for u in uav_ids
            ],
            name="UAV",
            showlegend=False,
        ))

        return traces

    # 초기 프레임
    init_traces = make_frame_traces(timesteps[0])
    all_traces  = bld_traces + init_traces

    # 프레임 생성
    frames = []
    for t_s in timesteps:
        ft = make_frame_traces(t_s)
        frames.append(go.Frame(
            data=ft,
            traces=list(range(n_bld, n_bld + len(ft))),
            name=t_s,
            layout=go.Layout(title_text=f"{scenario}  |  t = {float(t_s):.2f}s"),
        ))

    # 슬라이더 눈금: 최대 20개
    step_size = max(1, len(timesteps) // 20)
    slider_steps = [
        dict(args=[[t], {"frame": {"duration": 0, "redraw": True},
                          "mode": "immediate", "transition": {"duration": 0}}],
             label=f"{float(t):.1f}s",
             method="animate")
        for t in timesteps[::step_size]
    ]

    fig = go.Figure(
        data=all_traces,
        frames=frames,
        layout=go.Layout(
            title=dict(text=f"{scenario}  |  t = {float(timesteps[0]):.2f}s",
                       font=dict(size=15)),
            xaxis=dict(title="X (m)", showgrid=True, gridcolor="#eee"),
            yaxis=dict(title="Y (m)", showgrid=True, gridcolor="#eee",
                       scaleanchor="x"),
            plot_bgcolor="#f8f9fa",
            paper_bgcolor="white",
            height=580,
            legend=dict(x=1.01, y=1, bgcolor="rgba(255,255,255,0.8)"),
            updatemenus=[dict(
                type="buttons", showactive=False,
                x=0.0, xanchor="left", y=-0.12, yanchor="top",
                buttons=[
                    dict(label="▶ Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 300, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸ Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate", "transition": {"duration": 0}}]),
                ],
            )],
            sliders=[dict(
                active=0, steps=slider_steps,
                x=0.08, xanchor="left",
                y=-0.06, yanchor="top",
                len=0.9,
                currentvalue=dict(prefix="t = ", suffix="s", visible=True,
                                   font=dict(size=12)),
            )],
            # 상태 범례 annotation
            annotations=[
                dict(x=1.01, y=0.55, xref="paper", yref="paper", showarrow=False,
                     text="<b>링크 상태</b><br>"
                          "<span style='color:#2ecc71'>━ healthy</span><br>"
                          "<span style='color:#f39c12'>━ degraded</span><br>"
                          "<span style='color:#e74c3c'>━ disconnected</span>",
                     align="left", bgcolor="white",
                     bordercolor="#ccc", borderwidth=1,
                     font=dict(size=12)),
            ],
        ),
    )

    out = OUT_DIR / f"animation_{scenario}.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Plotly HTML 저장: {out}")


# ── Matplotlib GIF ────────────────────────────────────────────────────────────
def make_gif(scenario: str, timesteps, positions, links, obstacles,
             fps: int = 8, max_frames: int = 80):
    # 너무 많은 프레임이면 균등 샘플링
    if len(timesteps) > max_frames:
        idx = np.linspace(0, len(timesteps) - 1, max_frames, dtype=int)
        timesteps = [timesteps[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f9fa")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)

    def draw_frame(t_s):
        ax.cla()
        ax.set_facecolor("#f8f9fa")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        ax.set_title(f"{scenario}   t = {float(t_s):.2f}s", fontsize=12, fontweight="bold")

        # 건물
        for b in obstacles:
            rect = mpatches.FancyBboxPatch(
                (b["x_min"], b["y_min"]),
                b["x_max"] - b["x_min"], b["y_max"] - b["y_min"],
                boxstyle="square,pad=0",
                facecolor="lightgray", edgecolor="#555", linewidth=1.2, zorder=2,
            )
            ax.add_patch(rect)
            cx = (b["x_min"] + b["x_max"]) / 2
            cy = (b["y_min"] + b["y_max"]) / 2
            ax.text(cx, cy, f"{b['id']}\n{b['atten']}dB",
                    ha="center", va="center", fontsize=7, color="#444", zorder=3)

        pos  = positions.get(t_s, {})
        lnks = links.get(t_s, {})

        # 링크
        for (src, dst), info in lnks.items():
            if src not in pos or dst not in pos:
                continue
            x0, y0, _ = pos[src]
            x1, y1, _ = pos[dst]
            ax.plot([x0, x1], [y0, y1],
                    color=STATE_COLOR_MPL[info["state"]],
                    linewidth=2, alpha=0.75, zorder=3)

        # UAV
        for uid, (x, y, role) in pos.items():
            c = ROLE_COLOR.get(role, "#3498db")
            ax.scatter(x, y, s=180, color=c, edgecolors="white",
                       linewidths=2, zorder=5)
            ax.text(x, y + 2.5, f"UAV{uid}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#2c3e50", zorder=6)

        # 범례
        handles = [
            mpatches.Patch(color="green",  label="healthy"),
            mpatches.Patch(color="orange", label="degraded"),
            mpatches.Patch(color="red",    label="disconnected"),
            mpatches.Patch(color="lightgray", edgecolor="#555", label="building"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.85)

        # 축 범위 고정 (전체 위치 기반)
        all_x = [v[0] for d in positions.values() for v in d.values()]
        all_y = [v[1] for d in positions.values() for v in d.values()]
        pad = 15
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=timesteps,
        interval=1000 // fps, repeat=False,
    )

    out = OUT_DIR / f"animation_{scenario}.gif"
    writer = animation.PillowWriter(fps=fps)
    anim.save(str(out), writer=writer, dpi=110)
    plt.close(fig)
    print(f"GIF 저장: {out}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="relay_failure",
                        help="시나리오 이름 (기본: relay_failure)")
    parser.add_argument("--gif-only",  action="store_true")
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    scenario  = args.scenario
    obstacles = load_obstacles()
    print(f"건물 수: {len(obstacles)}")
    print(f"시나리오 로딩: {scenario}")
    timesteps, positions, links = load_scenario(scenario)
    print(f"타임스텝 수: {len(timesteps)}")

    if not args.gif_only:
        print("Plotly HTML 생성 중...")
        make_plotly_animation(scenario, timesteps, positions, links, obstacles)

    if not args.html_only:
        print("GIF 생성 중...")
        make_gif(scenario, timesteps, positions, links, obstacles)

    print("완료.")


if __name__ == "__main__":
    main()
