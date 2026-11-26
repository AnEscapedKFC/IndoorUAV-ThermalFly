"""Measure physical action changes under equivalent native horizontal reference frames."""
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge_client import DEFAULT_SOCKET, exchange
from sim_coordinates import EpisodeCoordinates, signed_degrees


def main():
    out = ROOT / "diagnoses/pose"
    out.mkdir(exist_ok=True)
    seeds = [37, 38, 39, 40, 41, 42, 43, 44]
    gauges = [0, 90, 180, 270]
    results = {}
    with (out / "predictions.jsonl").open("x") as log:
        for path in sorted((ROOT / "runs/model").iterdir()):
            with (path / "decisions.jsonl").open() as f:
                decision = json.loads(next(f))
            with (path / "frames.jsonl").open() as f:
                spawn = json.loads(next(f))
            state = spawn["position_w_m"]
            angle = spawn["yaw_deg"]
            values = {}
            for gauge, seed in itertools.product(gauges, seeds):
                mapper = EpisodeCoordinates(state, angle, gauge)
                req = dict(decision["request"], seed=seed, state=mapper.to_native(state, angle).tolist(),
                           id=f"pose-{path.name}-{gauge}-{seed}")
                reply = exchange(DEFAULT_SOCKET, req)
                assert reply["id"] == req["id"] and reply["seed"] == seed
                chunk = np.asarray(reply["waypoints"])
                assert chunk.shape == (10, 4) and np.isfinite(chunk).all()
                if (gauge, seed) == (0, 42):
                    np.testing.assert_allclose(chunk, decision["native_chunk"], atol=1e-5, rtol=0)
                end = mapper.to_world(chunk[-1])
                world_delta = (end[:3] - state).tolist()
                values[(gauge, seed)] = world_delta
                log.write(json.dumps(dict(episode=path.name, gauge_deg=gauge, seed=seed, request=req,
                    native_chunk=chunk.tolist(), world_delta_m=world_delta,
                    yaw_delta_deg=signed_degrees(end[3] - angle))) + "\n")
                log.flush()
            by_gauge = np.asarray([[values[g, s] for s in seeds] for g in gauges])
            means = by_gauge.mean(axis=1)
            # Paired seeds are represented in native coordinates; summarize separately over the eight draws.
            pair, change = max(((p, math.dist(means[p[0]], means[p[1]]))
                                for p in itertools.combinations(range(len(gauges)), 2)), key=lambda row: row[1])
            within = np.mean(np.linalg.norm(by_gauge - means[:, None, :], axis=2), axis=1)
            results[path.name] = dict(gauges_deg=gauges, seeds=seeds, mean_world_delta_per_gauge_m=means.tolist(),
                mean_seed_distance_from_its_gauge_mean_m=within.tolist(), largest_mean_difference_m=change,
                maximum_pair_deg=[gauges[i] for i in pair],
                difference_between_0_and_90_m=math.dist(means[0],means[1]))
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
