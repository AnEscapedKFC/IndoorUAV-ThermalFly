"""Run preregistered, distance-free RGB-D inspection tasks using physical dynamics."""

import argparse
import fcntl
import os
from pathlib import Path
import tempfile
from scipy.spatial.transform import Rotation

from metric_flight.experiment import OUT, execute
from metric_flight.furniture import deny_expert_reads
from detector import Detector

TASKS = {
    "sofa": "Fly toward the black sofa and show its whole front. Keep both ends, the backrest and the seat in the picture while hovering.",
    "laptop": "Fly toward the open laptop and inspect its screen and keyboard together. Hover with both details clearly in view.",
    "ac": "Fly toward the wall-mounted air conditioner. Inspect its front panel and lower air outlet together, then hold that view.",
}
# Starts chosen using initial visibility and spawn clearance only. No goal poses.
STARTS = {
    "sofa_east": ("sofa", (2.1, -0.85, 1.70, 128.0, -16.0)),
    "sofa_west": ("sofa", (-1.15, -0.65, 1.50, 80.0, -18.0)),
    "laptop_east": ("laptop", (1.30, 2.10, 1.70, -123.0, -20.0)),
    "laptop_west": ("laptop", (-1.40, 2.0, 1.63, -58.0, -22.0)),
    "ac_east": ("ac", (2.10, -0.85, 1.60, 117.0, 13.0)),
    "ac_west": ("ac", (-1.15, -0.65, 1.55, 69.0, 13.0)),
    "sofa_diagonal": ("sofa", (1.80, 0.30, 1.59, 134.0, -22.0)),
    "laptop_diagonal": ("laptop", (0.47, 2.04, 1.72, -107.0, -30.0)),
    "ac_diagonal": ("ac", (0.73, -0.82, 1.46, 98.0, 13.0)),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(STARTS))
    args = parser.parse_args(argv)
    deny_expert_reads()
    detector = Detector(use_cache=False)
    root = OUT / "history_inspection"
    root.mkdir(parents=True, exist_ok=True)
    for name in args.cases or STARTS:
        noun, (x, y, z, yaw, pitch) = STARTS[name]
        instruction = TASKS[noun]
        task = dict(
            target=noun,
            instruction=instruction,
            initial=dict(
                position_w_m=[x, y, z],
                velocity_w_mps=[0.0, 0.0, 0.0],
                rig_pitch_deg=pitch,
                q_w_from_b_wxyz=Rotation.from_euler("z", yaw, degrees=True)
                .as_quat()[[3, 0, 1, 2]]
                .tolist(),
            ),
        )
        with (root / (name + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (root / name / "episode.json").exists():
                continue
            with tempfile.TemporaryDirectory(dir=root, prefix="." + name + "-") as work:
                folder = Path(work) / name
                execute(
                    name,
                    "auto",
                    True,
                    detector,
                    room_goal="history",
                    record_rgbt="history",
                    furniture_task=task,
                    furniture_mode="history",
                    history_output=folder,
                )
                os.rename(folder, root / name)
