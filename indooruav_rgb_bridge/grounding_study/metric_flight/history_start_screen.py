"""Select one distant visible start per target using only observations at launch."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import SIM, RoomRig, RoomSpace, rgb_factory, PolicyFlightSimulator
from detector import Detector
from metric_flight.furniture import deny_expert_reads
from metric_flight.depth_guard import DepthGuard, preflight, survey_rig
from metric_flight.instance_mask import instance_mask

CANDIDATES = {
    "sofa": [
        (2.65, -1.02, 1.78, 128.0, -17.0),
        (1.55, -1.00, 1.70, 120.0, -16.0),
        (0.70, -1.04, 1.67, 109.0, -15.0),
        (-0.74, -0.98, 1.62, 87.0, -16.0),
    ],
    "laptop": [
        (2.56, 1.86, 1.72, -145.0, -19.0),
        (2.16, 2.05, 1.76, -137.0, -21.0),
        (1.48, 2.20, 1.73, -125.0, -22.0),
        (-1.34, 1.82, 1.57, -57.0, -20.0),
    ],
    "ac": [
        (2.65, -1.02, 1.75, 118.0, 10.0),
        (1.55, -1.00, 1.60, 110.0, 12.0),
        (0.70, -1.04, 1.58, 99.0, 12.0),
        (-0.74, -0.98, 1.63, 77.0, 12.0),
    ],
}


def measure(noun, start, known, detector, masker, tmp):
    x, y, z, yaw, pitch = start
    p = np.array([x, y, z])
    scene = SIM / "scenes/room65_selected_anchored"
    rig = survey_rig(rgb_factory(RoomRig, pitch))
    row = dict(pose=start, eligible=False)
    with PolicyFlightSimulator(
        scene / "models/rgb.ply",
        point_cloud=scene / "collision/rgb_surface_m.ply",
        config={"limits": {"max_steps": 1500}},
        seed=65,
        spawn=p.tolist(),
        yaw=yaw,
        resolution=1.0,
        spawn_velocity_w_mps=[0.0, 0.0, 0.0],
        scene_factory=rig,
    ) as sim:
        q = sim.last_receipt["state"]["quaternion_wxyz"]
        rot = Rotation.from_quat(np.array(q)[[1, 2, 3, 0]]).as_matrix()
        if not known.contains_box(p, rot, sim.obstacles.safety_half_size, extra=0.08):
            return dict(row, status="spawn_clearance")
        rgb = np.rint(sim.last_rgb * 255).astype(np.uint8)
        path = Path(tmp) / "initial.png"
        Image.fromarray(rgb).save(path)
        boxes = detector(path, noun)["boxes"]
        if not boxes:
            return dict(row, status="target_not_detected")
        fg = masker(rgb, boxes[0]["xyxy"])
        depths = sim.last_depth[
            fg & np.isfinite(sim.last_depth) & (sim.last_depth > 0.08)
        ]
        if len(depths) < 100:
            return dict(row, status="insufficient_depth")
        row["initial_median_optical_depth_m"] = float(np.median(depths))
        guard = DepthGuard(p, sim.obstacles.safety_half_size)
        preflight(sim, guard)
        guard.observe(sim.last_depth, sim.scene)
        goal = np.r_[p + 0.12 * rot[:, 0], yaw]
        check = guard.check(
            sim.last_receipt["state"], goal, 0.5, None, 0.24 * rot[:, 0]
        )
        return dict(row, status=check, eligible=check == "free")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    deny_expert_reads()
    mask = instance_mask()
    detector = Detector(use_cache=False, device="cuda")
    known = RoomSpace()
    runs = {}
    chosen = {}
    with TemporaryDirectory() as tmp:
        for noun, starts in CANDIDATES.items():
            rows = [measure(noun, p, known, detector, mask, tmp) for p in starts]
            runs[noun] = rows
            options = [r for r in rows if r["eligible"]]
            chosen[noun] = (
                max(options, key=lambda r: r["initial_median_optical_depth_m"])["pose"]
                if options
                else None
            )
            print(noun, chosen[noun], flush=True)
    args.output.write_text(
        json.dumps(dict(candidates=runs, selected=chosen), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
