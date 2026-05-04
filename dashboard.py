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
import math
from collections import defaultdict
from pathlib import Path

import dash
from dash import Input, Output, State, callback, dcc, html, no_update
import plotly.graph_objects as go

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "ns-3.47" / "datasets" / "uav_2d_initial"

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
print(f"시나리오 {len(SCENARIOS)}개 로딩 완료")


def get_ts(scenario: str) -> list[str]:
    return sorted(POSITIONS.get(scenario, {}).keys(), key=float)


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
                prev_relay: int | None, bad_streak: int) -> go.Figure:
    pos  = POSITIONS.get(scenario, {}).get(t_s, {})
    lnks = LINKS.get(scenario, {}).get(t_s, {})

    traces = []

    # ── 건물 ────────────────────────────────────────────────────────────────
    for b in OBSTACLES:
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
        main_cx, main_cy = _centroid(pos, main_comp)
        for comp in comps:
            if comp == main_comp: continue
            for uid in comp:
                isolated_uavs.add(uid)
                x, y, _ = pos[uid]
                dx, dy = main_cx - x, main_cy - y
                d = math.sqrt(dx**2 + dy**2)
                if d > 0.01:
                    scale = STEP_ARROW / d
                    correction_arrows.append((x, y, dx*scale, dy*scale))

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


# ── 앱 ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="UAV 통신 대시보드")

app.layout = html.Div([
    dcc.Store(id="frame-store", data=0),
    dcc.Store(id="prev-relay-store", data=None),
    dcc.Store(id="bad-streak-store", data=0),
    dcc.Interval(id="interval", interval=600, n_intervals=0, disabled=True),

    # ── 헤더 ─────────────────────────────────────────────────────────────────
    html.Div([
        html.H2("UAV 군집 통신 장애 예측 대시보드",
                style={"margin": 0, "color": "white", "fontSize": 20}),
        html.Span("중계 전환 & 위치 보정 시각화",
                  style={"color": "#bdc3c7", "fontSize": 13}),
    ], style={"background": "#2c3e50", "padding": "14px 24px",
              "display": "flex", "flexDirection": "column"}),

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
                         style={"maxHeight": 160, "overflowY": "auto",
                                "fontSize": 12}),
            ], style={"background": "white", "borderRadius": 8, "padding": 12,
                      "marginBottom": 12, "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),

            # 위치 보정 이력
            html.Div([
                html.H4("위치 보정 이력", style={"margin": "0 0 8px", "fontSize": 13,
                                               "color": "#8e44ad"}),
                html.Div(id="correction-log",
                         style={"maxHeight": 140, "overflowY": "auto",
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
                      "boxShadow": "0 1px 4px rgba(0,0,0,.1)"}),
        ], style={"flex": "1.2", "display": "flex", "flexDirection": "column"}),

    ], style={"display": "flex", "padding": "14px 24px",
              "background": "#f5f6fa", "minHeight": "calc(100vh - 150px)"}),

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
    State("prev-relay-store", "data"),
    State("bad-streak-store", "data"),
)
def update_view(frame_idx, scenario, prev_relay, bad_streak):
    ts  = get_ts(scenario)
    idx = min(frame_idx, len(ts) - 1)
    t_s = ts[idx]

    fig = make_figure(scenario, t_s, prev_relay, bad_streak)

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


if __name__ == "__main__":
    print("대시보드 시작: http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
