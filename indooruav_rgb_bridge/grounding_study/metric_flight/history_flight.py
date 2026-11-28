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
    "sofa_long": ("sofa", (0.7, -1.04, 1.67, 109.0, -15.0)),
    "laptop_long": ("laptop", (2.16, 2.05, 1.76, -137.0, -21.0)),
    "ac_long": ("ac", (0.7, -1.04, 1.58, 99.0, 12.0)),
}


COORDINATED_TASKS = {
    "level": "Approach the air conditioner and look horizontally at its front panel. Hover with the unit centered and fully in view.",
    "outlet": "Approach the air conditioner and look up at its lower air outlet from below. Keep the unit centered and the outlet visible while hovering.",
    "desk": "Approach the open laptop and look down from above at its screen and keyboard together. Hover with both clearly visible.",
    "overview": TASKS["sofa"],
}
# Identical launch states for both instructions on the same physical AC.
COORDINATED_CASES = {
    **{f"{mode}_{start}": (mode, "ac_"+start) for start in ("east","diagonal") for mode in ("level","outlet")},
    **{"desk_"+start: ("desk","laptop_"+start) for start in ("east","long","west")},
    "overview_east": ("overview","sofa_east"),
}


# Additional launch states selected before scoring, separate from the historical starts.
NEW_STARTS = {
    "ac_new": ("ac", (.28,-.52,1.64,92.,10.)),
    "laptop_new": ("laptop", (.78,1.86,1.70,-114.,-22.)),
}
COORDINATED_CASES.update(level_new=("level","ac_new"), outlet_new=("outlet","ac_new"),
                         desk_new=("desk","laptop_new"))
# Diagnostic pair added after horizontal-view detection losses. Report separately.
COORDINATED_CASES.update(level_west=("level","ac_west"), outlet_west=("outlet","ac_west"))


WINDOW_CASES = dict(COORDINATED_CASES)
WINDOW_CASES.update(overview_west=("overview","sofa_west"),overview_diagonal=("overview","sofa_diagonal"),
                    overview_long=("overview","sofa_long"))
# Chosen from door/chair visibility in launch images; no final pose is supplied.
NEW_STARTS.update(chair_north=("armchair",(.47,2.04,1.72,-47.,-32.)),
                  chair_east=("armchair",(1.30,2.10,1.70,-64.,-34.)),
                  chair_far=("armchair",(-1.40,2.0,1.63,-32.,-25.)))
WINDOW_CASES.update({n:("chair",n) for n in ("chair_north","chair_east","chair_far")})
# New positions frozen before any holdout outcomes, shared between both AC instructions.
NEW_STARTS.update(ac_fresh=("ac",(-.55,-.84,1.59,82.,10.)),
                  sofa_fresh=("sofa",(-.55,-.84,1.63,89.,-19.)),
                  laptop_fresh=("laptop",(1.02,2.15,1.66,-115.,-27.)),
                  chair_fresh=("armchair",(.08,1.87,1.62,-42.,-31.)))
WINDOW_CASES.update(level_fresh=("level","ac_fresh"),outlet_fresh=("outlet","ac_fresh"),
                    overview_fresh=("overview","sofa_fresh"),desk_fresh=("desk","laptop_fresh"),
                    chair_fresh=("chair","chair_fresh"))
WINDOW_TASKS = dict(COORDINATED_TASKS, chair="Approach the freestanding black armchair by the door. Keep the whole chair centered, with its backrest and seat visible, and hover.")


def main(argv=None, *, coordinated=False, window=False):
    parser = argparse.ArgumentParser(description=__doc__)
    cases = WINDOW_CASES if window else COORDINATED_CASES if coordinated else STARTS
    parser.add_argument("--cases", nargs="+", choices=list(cases))
    args = parser.parse_args(argv)
    deny_expert_reads()
    detector = Detector(
        use_cache=False, device=os.environ.get("INDOORUAV_BOX_DEVICE", "cuda")
    )
    root = OUT / ("coordinated_inspection" if coordinated else "history_inspection_final")
    if window:
        root = OUT / "observation_window"
    root.mkdir(parents=True, exist_ok=True)
    for name in args.cases or cases:
        mode, start = cases[name] if coordinated else (STARTS[name][0], name)
        noun, (x, y, z, yaw, pitch) = (NEW_STARTS if start in NEW_STARTS else STARTS)[start]
        instruction = (WINDOW_TASKS if window else COORDINATED_TASKS)[mode] if coordinated else TASKS[noun]
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
                    furniture_mode="window" if window else "coordinated" if coordinated else "history",
                    history_output=folder,
                )
                os.rename(folder, root / name)
