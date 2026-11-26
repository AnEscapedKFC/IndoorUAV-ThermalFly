"""Compare actual closed-loop paths with recorded demonstrations at their own timestamps."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def trace(path):
    with path.open() as handle:
        frames = [json.loads(line) for line in handle]
    return np.array([f["t_s"] for f in frames]), np.array([f["position_w_m"] for f in frames])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs" / "model")
    parser.add_argument("--dataset", type=Path, default=ROOT.parent / "gs_gradnav_simulator" / "outputs" / "expert_rgbt_20261123")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "flight_paths.png")
    args = parser.parse_args()
    runs = sorted(args.runs.glob("*/summary.json"))
    if not runs:
        parser.error("No completed episode summaries were found")
    fig, axes = plt.subplots(len(runs), 2, figsize=(10, 2.75 * len(runs)), squeeze=False, constrained_layout=True)
    for panels, result in zip(axes, runs):
        summary = json.loads(result.read_text())
        episode = summary["episode"]
        xy, height = panels
        for path, label, color, style in [
            (args.dataset / "episodes" / episode / "frames.jsonl", "Expert recording", "#657484", "--"),
            (result.parent / "frames.jsonl", "IndoorUAV-Agent", "#087f77", "-"),
        ]:
            t, xyz = trace(path)
            xy.plot(xyz[:, 0], xyz[:, 1], ls=style, color=color, lw=1.9, label=label)
            height.plot(t, xyz[:, 2], ls=style, color=color, lw=1.9, label=label)
        start = summary["spawn_w_m"]
        goal = summary["goal_w_m"]
        end = summary["end_w_m"]
        xy.scatter(*start[:2], marker="o", facecolor="white", edgecolor="#222", zorder=4, label="Start")
        xy.scatter(*goal[:2], marker="*", color="#a14c11", s=90, zorder=5, label="Goal")
        xy.scatter(*end[:2], marker="x", color="#087f77", s=50, zorder=6, label="Policy end")
        xy.set_title(episode.replace("_", " "), loc="left", fontsize=11, weight="bold")
        xy.set(xlabel="World X (m)", ylabel="World Y (m)")
        xy.set_aspect("equal", adjustable="datalim")
        height.set_title("Altitude through flight", loc="left", fontsize=11)
        height.set(xlabel="Simulation time (s)", ylabel="World Z (m)")
        for panel in panels:
            panel.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.56), frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
