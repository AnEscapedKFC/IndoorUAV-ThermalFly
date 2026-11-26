"""Check the model's requested body poses against the original known-volume test."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim_bridge import CorridorSpace, RoomSpace
from sim_coordinates import EpisodeCoordinates
from pointcloud import DEFAULT_DRONE_SIZE


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def main():
    half = np.asarray(DEFAULT_DRONE_SIZE) / 2 + .05
    output = {}
    for group in ["model", "bounded"]:
        for folder in sorted((ROOT / "runs" / group).iterdir()):
            summary = json.loads((folder / "summary.json").read_text())
            frames = rows(folder / "frames.jsonl")
            cur = {f["frame_id"]:f for f in frames}
            space = RoomSpace() if folder.name.startswith("room_") else CorridorSpace()
            coord = EpisodeCoordinates(frames[0]["position_w_m"], frames[0]["yaw_deg"], summary["native_start_yaw_deg"])

            def in_space(wp):
                return bool(space.contains_box(wp[:3], Rotation.from_euler("z", wp[3], degrees=True).as_matrix(), half))

            records = []
            for d in rows(folder / "decisions.jsonl"):
                native_world = [coord.to_world(v) for v in d["native_chunk"]]
                expected = d.get("raw_target_pose_w_m_deg",d["target_pose_w_m_deg"])
                np.testing.assert_allclose(native_world[-1], expected, atol=1.e-6, rtol=0)
                ok = [in_space(w) for w in native_world]
                records.append(dict(frame=d["frame_id"], request_s=cur[d["frame_id"]]["t_s"],
                    first_native_outside_index_1based=next((j for j,inside in enumerate(ok, 1) if not inside),None),
                    native_endpoint_inside=ok[-1], issued_endpoint_inside_assuming_upright=in_space(d["target_pose_w_m_deg"]),
                    hold_replaces_target=d["stop_heuristic"]))
            # Re-test known volume with measured attitudes and positions, matching saved flags.
            for f in frames:
                r = Rotation.from_quat(np.asarray(f["q_w_from_b_wxyz"])[[1,2,3,0]]).as_matrix()
                assert bool(space.contains_box(np.asarray(f["position_w_m"]),r,half)) == f["in_known_space"]
            output[str(folder.relative_to(ROOT))] = records
    (ROOT/"diagnoses/planned_space.json").write_text(json.dumps(output,indent=2)+"\n")
    for name, records in output.items():
        print(name, [(r["frame"],r["native_endpoint_inside"],r["issued_endpoint_inside_assuming_upright"]) for r in records])


if __name__ == "__main__":
    main()
