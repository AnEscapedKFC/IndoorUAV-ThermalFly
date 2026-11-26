"""Compare two endpoint-velocity rules without changing the bridge or simulator on disk.

The baseline drives to the last 10 Hz predicted pose in 1 second and uses start-to-goal
displacement / duration as its desired terminal velocity. Here only that velocity is
changed, using the last 0.2 s of the very same predicted sequence. Each new decision
is made from the actual resulting image and body state.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicHermiteSpline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sim_bridge as bridge
from sim_coordinates import EpisodeCoordinates, signed_degrees


def run(episode):
    dataset = bridge.SIM / "outputs" / "expert_rgbt_20261123"
    with (ROOT / "runs/model" / episode / "frames.jsonl").open() as handle:
        initial = json.loads(next(handle))
    coordinates = EpisodeCoordinates(initial["position_w_m"], initial["yaw_deg"])
    output = ROOT / "diagnoses/tracker" / episode
    chunks = []
    velocities = []
    original_worker = bridge.ModelProcess
    original_curve = bridge.guided_curve

    class ObservedWorker(original_worker):
        def predict(self, request):
            path = super().predict(request)
            chunks.append(path)
            return path

    def curve(state, goal, seconds, holding):
        if holding:
            return original_curve(state, goal, seconds, holding)
        start = np.r_[state["position"], state["euler_deg"][2]]
        end = np.asarray(goal, dtype=float).copy()
        previous = coordinates.to_world(chunks[-1][7])
        predicted_end = coordinates.to_world(chunks[-1][9])
        np.testing.assert_allclose(end, predicted_end)
        velocity = (predicted_end - previous) / .2
        velocity[3] = signed_degrees(predicted_end[3] - previous[3]) / .2
        end[3] = start[3] + signed_degrees(end[3] - start[3])
        current_rate = np.r_[state["velocity"], np.rad2deg(state["angular_velocity"][2])]
        velocities.append({"decision":len(chunks)-1, "terminal_rate_from_chunk":velocity.tolist(),
                           "baseline_secant_at_this_state":((end - start)/seconds).tolist()})
        spline = CubicHermiteSpline([0., seconds], [start, end], [current_rate, velocity])
        return lambda _, time: (spline(time), spline(time, 1), spline(time, 2), 0)

    bridge.ModelProcess = ObservedWorker
    bridge.guided_curve = curve
    try:
        args = argparse.Namespace(dataset=dataset,episode=episode,output=output,mode="model",
            worker=f"python -B {ROOT / 'bridge_client.py'}",instruction=None,seed=42,
            segment_seconds=1.,hold_seconds=2.,max_seconds=60.,subinstructions=None,
            limits=None,native_start_yaw_deg=0.)
        bridge.run(args)
    finally:
        bridge.ModelProcess = original_worker
        bridge.guided_curve = original_curve
    baseline = ROOT / "runs/model" / episode / "decisions.jsonl"
    with baseline.open() as f:
        first = json.loads(next(f))
    np.testing.assert_allclose(chunks[0], first["native_chunk"], atol=1.e-6, rtol=0)
    (output/"terminal_rates.json").write_text(json.dumps(velocities,indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    run(parser.parse_args().episode)
