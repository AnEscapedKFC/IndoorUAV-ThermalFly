"""Check recorded flights, action/observation linkage and simulator source integrity."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
SIM = ROOT.parent / "gs_gradnav_simulator"
DATA = SIM / "outputs" / "expert_rgbt_20261123"


def rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_inputs():
    before = json.loads((ROOT / "results/source_integrity_before.json").read_text())
    sources = {str(f.relative_to(SIM)): sha(f) for f in sorted(SIM.rglob("*.py")) if "outputs" not in f.relative_to(SIM).parts}
    inputs = sorted([DATA / "splits.json", *DATA.rglob("episode.json"), *DATA.rglob("frames.jsonl"),
                     *DATA.rglob("decisions.jsonl"), *(DATA / "episodes").glob("*/rgb/*.png")])
    digest = hashlib.sha256()
    for path in inputs:
        digest.update((str(path.relative_to(SIM)) + "\0").encode())
        digest.update(bytes.fromhex(sha(path)))
    commit = subprocess.check_output(["git", "-C", str(SIM), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(SIM), "status", "--porcelain=v1"], text=True)
    assert before["source_sha256"] == sources
    assert (before["git_commit"], before["git_status"]) == (commit, status)
    assert before["expert_rgb_input_index_sha256"] == digest.hexdigest()
    return dict(commit=commit, git_status=status, source_files=len(sources), expert_input_files=len(inputs),
                expert_input_sha256=digest.hexdigest())


def audit_episode(result):
    folder = result.parent
    summary = json.loads(result.read_text())
    frames = rows(folder / "frames.jsonl")
    episode = DATA / "episodes" / summary["episode"]
    expert = rows(episode / "frames.jsonl")
    meta = json.loads((episode / "episode.json").read_text())
    assert len(frames) == summary["frames"] and [f["frame_id"] for f in frames] == list(range(len(frames)))
    assert all(a["t_s"] < b["t_s"] for a, b in zip(frames, frames[1:]))
    assert math.dist(frames[0]["position_w_m"], expert[0]["position_w_m"]) < 1.e-5
    assert math.isclose(math.dist(frames[-1]["position_w_m"], meta["task_spec"]["goal_w_m"]),
                        summary["terminal_position_error_m"], abs_tol=1.e-6)
    record = {"path": str(folder.relative_to(ROOT)), "mode": summary["mode"], "frames": len(frames)}
    if summary["mode"] == "replay":
        assert len(frames) == len(expert)
        record["max_replay_position_error_m"] = max(math.dist(a["position_w_m"], b["position_w_m"]) for a, b in zip(frames, expert))
    if summary["mode"] == "model":
        decisions = rows(folder / "decisions.jsonl")
        assert len(decisions) == summary["inferences"] > 0
        ids = [row["frame_id"] for row in decisions]
        assert ids[0] == 0 and ids == sorted(set(ids))
        for row in decisions:
            req = row["request"]
            current, reference = Path(req["rgb_path"]), Path(req["ref_rgb_path"])
            assert current.parent == reference.parent == folder / "rgb"
            assert int(current.stem) == row["frame_id"] and int(reference.stem) <= row["frame_id"]
            assert current.is_file() and reference.is_file()
            assert len(row["native_chunk"]) == 10
            assert all(len(slot) == 4 and all(math.isfinite(x) for x in slot) for slot in row["native_chunk"])
        record["decisions"] = len(decisions)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", default=[])
    args = parser.parse_args()
    results = []
    for group in args.group:
        matched = sorted((ROOT / "runs" / group).glob("*/summary.json"))
        if not matched:
            raise FileNotFoundError(f"No completed runs in {group}")
        results.extend(audit_episode(result) for result in matched)
    print(json.dumps({"protected_inputs": audit_inputs(), "flights": results}, indent=2))
