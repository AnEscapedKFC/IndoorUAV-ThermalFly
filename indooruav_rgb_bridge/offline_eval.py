"""Teacher-forced RGB evaluation of IndoorUAV's native 10-pose output."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess

import numpy as np

from sim_coordinates import EpisodeCoordinates


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "gs_gradnav_simulator/outputs/expert_rgbt_20261123"
# One distinct validation route family and one test route family per scene.
EPISODES = ["room_inspect_ac", "corridor_high_return", "room_search_laptop_sofa", "corridor_climb_accelerate"]
WRONG_ROOM = {"room_inspect_ac": "room_inspect_laptop", "room_search_laptop_sofa": "room_inspect_ac"}


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def yaw(frame):
    w, x, y, z = frame["q_w_from_b_wxyz"]
    return math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def yaw_error(a, b):
    return abs(math.degrees((a - b + math.pi) % math.tau - math.pi))


def projection(frame, calibration, landmarks):
    """Geometric visibility of documented landmark centres; occlusion is unknown."""
    tf = calibration["source_to_sim"]
    view = np.asarray(frame["rgb_w2c_native"])
    k = np.asarray(calibration["rgb_K"])
    width, height = calibration["wh"]
    result = {}
    for name, detail in landmarks.items():
        point = np.asarray(tf["rotation"]).T @ detail["center_w_m"] / tf["meters_per_source_unit"] + tf["origin"]
        xyz = view[:3, :3] @ point + view[:3, 3]
        uv = (k @ xyz)[:2] / xyz[2] if xyz[2] > 1e-8 else None
        in_frame = uv is not None and (0 <= uv[0] < width and 0 <= uv[1] < height)
        result[name] = {"center_in_frame": bool(in_frame), "pixel": uv.tolist() if uv is not None else None}
    return result


def prepare(root, samples, seed):
    splits = read_json(root / "splits.json")["splits"]
    room = root / "scenes/room65_selected_anchored"
    calibration = read_json(room / "calibration.json")
    landmarks = read_json(room / "landmarks.json")
    hold_path = HERE.parent / "gs_gradnav_simulator/scenes/corridor_native/navigation_task.json"
    chunks = []
    for name in EPISODES:
        folder = root / "episodes" / name
        episode = read_json(folder / "episode.json")
        split = next(s for s in ("test", "val") if name in splits[s])
        scene = "room" if name.startswith("room_") else "corridor"
        if scene == "room":
            other_path = root / "episodes" / WRONG_ROOM[name] / "episode.json"
            other = read_json(other_path)["instruction"]
        else:
            other_path = hold_path
            other = read_json(other_path)["hold_instruction"]
        decisions = rows(folder / "decisions.jsonl")
        frames = rows(folder / "frames.jsonl")
        # The local 20 Hz receipts and native 10 Hz dataset share every second control tick.
        # Sample across MOVE decisions and the recorded STOP. Require the entire future.
        eligible = [d for d in decisions if d["observed_frame_id"] + 20 < len(frames)]
        if len(eligible) < 2 or eligible[-1]["teacher"]["chunk"][0]["kind"] != "stop":
            raise ValueError(f"Missing complete endpoint supervision in {name}")
        coord = EpisodeCoordinates(frames[0]["position_w_m"], yaw(frames[0]))
        for idx in sorted(set(np.linspace(0, len(eligible)-1, min(samples, len(eligible))).round().astype(int))):
            decision = eligible[idx]
            j = decision["observed_frame_id"]
            observed = frames[j]
            future = frames[j+2:j+21:2]
            targets = [coord.to_native(row["position_w_m"], yaw(row)).tolist() for row in future]
            if any(abs(row["t_s"] - observed["t_s"] - i/10) > 1e-5 for i,row in enumerate(future, 1)):
                raise ValueError(f"Invalid future time alignment {name} frame {j}")
            sample = {"episode": name, "split": split, "scene": scene, "frame_id": j,
                      "decision_id": decision["decision_id"], "kind": decision["teacher"]["chunk"][0]["kind"],
                      "expert_subgoal": decision["teacher"]["subgoal_id"], "t_s": observed["t_s"],
                      "target_poses": targets, "target_end_frame": future[-1]["frame_id"],
                      "reference_frame": 0, "counterfactual_source": str(other_path)}
            if scene == "room":
                sample["rgb_center_projections"] = projection(observed, calibration, landmarks)
            common = {"rgb_path": str(folder / observed["rgb_path"]), "ref_rgb_path": str(folder / frames[0]["rgb_path"]),
                      "state": coord.to_native(observed["position_w_m"], yaw(observed)).tolist(),
                      "seed": seed + len(chunks)}
            pair = {}
            for mode, instruction in [("correct", episode["instruction"]), ("swapped", other)]:
                pair[mode] = {**common, "id": f"offline-{name}-{j}-{mode}", "instruction": instruction}
            sample["requests"] = pair
            chunks.append(sample)
    return chunks


def metrics(poses, targets):
    errors = [math.dist(p[:3], q[:3]) for p, q in zip(poses, targets)]
    return {"endpoint_m": errors[-1], "mean_path_error_m": statistics.mean(errors),
            "endpoint_yaw_deg": yaw_error(poses[-1][3], targets[-1][3])}


def extract_poses(reply, request):
    if "error" in reply or reply.get("id") != request["id"]:
        raise ValueError(f"{request['id']}: {reply}")
    path = np.asarray(reply.get("waypoints"), dtype=float)
    if path.shape != (10,4) or not np.isfinite(path).all() or reply.get("seed") != request["seed"]:
        raise ValueError(f"{request['id']}: shape/seed/finite check failed")
    return path.tolist()


def evaluate(sample, worker):
    modes = {}
    poses = {}
    for mode, req in sample["requests"].items():
        worker.stdin.write(json.dumps(req, allow_nan=False) + "\n")
        worker.stdin.flush()
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError(f"Worker disconnected at {req['id']}")
        reply = json.loads(line)
        poses[mode] = extract_poses(reply, req)
        modes[mode] = {"response": reply, **metrics(poses[mode], sample["target_poses"])}
    modes["stationary"] = metrics([sample["requests"]["correct"]["state"]]*10, sample["target_poses"])
    return {**sample, "result": modes,
            "instruction_shift_m": math.dist(poses["correct"][-1][:3], poses["swapped"][-1][:3]),
            "instruction_shift_yaw_deg": yaw_error(poses["correct"][-1][3], poses["swapped"][-1][3])}


def summarize(records):
    by_group = defaultdict(list)
    for r in records:
        for key in ("all", r["kind"], r["scene"], f"{r['scene']}/{r['kind']}", r["episode"]):
            by_group[key].append(r)
    report = {}
    for key, group in by_group.items():
        entry = {"n":len(group)}
        for mode in ("correct","swapped","stationary"):
            entry[mode] = {field: statistics.mean(r["result"][mode][field] for r in group)
                           for field in ("endpoint_m", "endpoint_yaw_deg","mean_path_error_m")}
        entry["correct_closer_than_stationary_count"] = sum(r["result"]["correct"]["endpoint_m"] < r["result"]["stationary"]["endpoint_m"] for r in group)
        entry["correct_closer_than_swapped_count"] = sum(r["result"]["correct"]["endpoint_m"] < r["result"]["swapped"]["endpoint_m"] for r in group)
        entry["mean_instruction_shift_m"] = statistics.mean(r["instruction_shift_m"] for r in group)
        entry["mean_instruction_shift_yaw_deg"] = statistics.mean(r["instruction_shift_yaw_deg"] for r in group)
        report[key] = entry
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--samples-per-episode", type=int, default=4)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--worker", help="JSONL worker command; omit to generate input manifest only")
    args = ap.parse_args()
    if args.samples_per_episode < 2:
        ap.error("Provide at least two decisions per episode")
    samples = prepare(args.data.resolve(), args.samples_per_episode, args.seed)
    out = args.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    manifest = {"target": "Recorded expert body pose at +0.1..+1.0 s, pose 10 scored at +1.0 s",
                "reference": "Episode-start RGB and full original unsegmented instruction",
                "counterfactual": "Same current RGB, reference, state and random key. Swapped instruction is scored against original expert future",
                "sampling": "Equally spaced indices across eligible expert decision IDs per episode, including terminal STOP",
                "projections": "Ground truth room landmark center projection for interpretation only; no occlusion test",
                "samples": samples}
    mpath = out / "manifest.json"
    if mpath.is_file() and read_json(mpath) != manifest:
        raise ValueError("Input manifest differs from existing one. Choose a new destination.")
    if not mpath.exists():
        mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    if not args.worker:
        print(f"Prepared {len(samples)} decision pairs at {mpath}")
        return
    rpath = out / "predictions.jsonl"
    completed = rows(rpath) if rpath.exists() else []
    for index, result in enumerate(completed):
        if any(result[k] != v for k,v in samples[index].items()):
            raise ValueError("Partial results do not match manifest")
    if len(completed) < len(samples):
        with ExitStack() as stack:
            worker = stack.enter_context(subprocess.Popen(shlex.split(args.worker), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True))
            stream = stack.enter_context(rpath.open("a"))
            stack.callback(worker.stdin.close)
            for sample in samples[len(completed):]:
                record = evaluate(sample, worker)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
                stream.flush()
                completed.append(record)
                print(f"Completed {len(completed)}/{len(samples)} pairs", flush=True)
    report = {"total_pairs":len(completed), "groups":summarize(completed)}
    (out / "summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
