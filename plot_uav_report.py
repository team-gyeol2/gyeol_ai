#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import statistics

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> float:
    return float(value) if value not in ("", None) else 0.0


def build_report(data_dir: Path, output_path: Path):
    rssi_rows = read_csv_rows(data_dir / "uav-rssi.csv")
    rtt_rows = read_csv_rows(data_dir / "uav-rtt.csv")
    pos_rows = read_csv_rows(data_dir / "uav-pos.csv")

    if not rssi_rows or not pos_rows:
        raise SystemExit("Missing CSV data. Run the ns-3 scenario first.")

    rssi_time = [to_float(row["time_s"]) for row in rssi_rows]
    rssi_signal = [to_float(row["signal_dbm"]) for row in rssi_rows]
    rssi_noise = [to_float(row["noise_dbm"]) for row in rssi_rows]

    rtt_time = [to_float(row["time_s"]) for row in rtt_rows]
    rtt_ms = [to_float(row["rtt_ms"]) for row in rtt_rows]

    uav_series = {}
    for row in pos_rows:
        uav_id = int(row["uav_id"])
        uav_series.setdefault(uav_id, {"time": [], "x": [], "y": [], "z": []})
        uav_series[uav_id]["time"].append(to_float(row["time_s"]))
        uav_series[uav_id]["x"].append(to_float(row["x"]))
        uav_series[uav_id]["y"].append(to_float(row["y"]))
        uav_series[uav_id]["z"].append(to_float(row["z"]))

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "RSSI / Noise Time Series",
            "RTT Time Series",
            "UAV X Position Over Time",
            "UAV 2D Trajectory",
        ),
        specs=[[{}, {}], [{}, {}]],
    )

    fig.add_trace(
        go.Scatter(x=rssi_time, y=rssi_signal, mode="lines+markers", name="RSSI (dBm)"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=rssi_time, y=rssi_noise, mode="lines", name="Noise (dBm)"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=rtt_time, y=rtt_ms, mode="lines+markers", name="RTT (ms)"),
        row=1,
        col=2,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for index, (uav_id, series) in enumerate(sorted(uav_series.items())):
        color = colors[index % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=series["time"],
                y=series["x"],
                mode="lines+markers",
                name=f"UAV{uav_id} X",
                line={"color": color},
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=series["x"],
                y=series["y"],
                mode="lines+markers",
                name=f"UAV{uav_id} path",
                line={"color": color},
            ),
            row=2,
            col=2,
        )

    rssi_avg = statistics.mean(rssi_signal) if rssi_signal else 0.0
    rtt_avg = statistics.mean(rtt_ms) if rtt_ms else 0.0
    ping_received = len(rtt_ms)
    ping_sent = (max(int(row["seq_no"]) for row in rtt_rows) + 1) if rtt_rows else ping_received
    plr = ((ping_sent - ping_received) / ping_sent * 100.0) if ping_sent else 0.0

    fig.update_layout(
        title=(
            f"ns-3 UAV Ad-hoc Report"
            f"<br><sup>avg RSSI={rssi_avg:.2f} dBm, avg RTT={rtt_avg:.2f} ms, "
            f"approx PLR={plr:.2f}%</sup>"
        ),
        height=900,
        width=1400,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.15},
    )

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="dBm", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="RTT (ms)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="X Position (m)", row=2, col=1)
    fig.update_xaxes(title_text="X Position (m)", row=2, col=2)
    fig.update_yaxes(title_text="Y Position (m)", row=2, col=2)

    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create Plotly report from ns-3 UAV CSV logs.")
    parser.add_argument(
        "--data-dir",
        default="/Users/inyoung/Desktop/캡디1/ns3/ns-3.47",
        help="Directory containing uav-rssi.csv, uav-rtt.csv, and uav-pos.csv",
    )
    parser.add_argument(
        "--output",
        default="/Users/inyoung/Desktop/캡디1/ns3/ns-3.47/uav-report.html",
        help="Output HTML report path",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = build_report(data_dir, output_path)
    print(f"Plotly report written to {result}")


if __name__ == "__main__":
    main()
