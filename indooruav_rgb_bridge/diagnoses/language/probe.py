"""Paired instruction-only interventions at the original four flight starts."""
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parents[1]
sys.path.insert(0, str(BRIDGE))
from bridge_client import DEFAULT_SOCKET, exchange


PROMPTS = {
    "forward": "Fly straight forward.",
    "backward": "Fly straight backward.",
    "left": "Fly sideways to the left without turning.",
    "right": "Fly sideways to the right without turning.",
    "up": "Fly upward.",
    "down": "Fly downward.",
    "yaw_left": "Turn left in place.",
    "yaw_right": "Turn right in place.",
    "hover": "Hover in place.",
}
FIRST_CLAUSES = {
    "room_inspect_ac": "Approach the wall-mounted air conditioner above the sofa.",
    "room_search_laptop_sofa": "Rise and turn slowly to search for the open laptop.",
    "corridor_high_return": "Fly back along the east section at about 1.82 m at a brisk cruising speed.",
    "corridor_climb_accelerate": "Start low. Fly straight down the corridor, climb and accelerate.",
}
PAIRS = [
    ("forward", "backward", "forward_m", 0.10),
    ("right", "left", "right_m", 0.10),
    ("up", "down", "up_m", 0.10),
    ("yaw_right", "yaw_left", "yaw_right_deg", 10.0),
]
SEEDS = [42, 73, 19]


def body_endpoint(initial, final):
    x, y, z, yaw = initial
    dx, dy, dz = final[0] - x, final[1] - y, final[2] - z
    return {"forward_m": dx * math.sin(yaw) - dy * math.cos(yaw),
            "right_m": dx * math.cos(yaw) + dy * math.sin(yaw), "up_m": dz,
            "yaw_right_deg": math.degrees((final[3] - yaw + math.pi) % math.tau - math.pi),
            "travel_m": math.sqrt(dx * dx + dy * dy + dz * dz)}


def run():
    rows = []
    for episode, clause in FIRST_CLAUSES.items():
        source = BRIDGE / "runs" / "model" / episode / "decisions.jsonl"
        logged = json.loads(source.open().readline())
        first = logged["request"]
        for seed in SEEDS:
            for condition, sentence in {**PROMPTS, "original": first["instruction"], "first_clause": clause}.items():
                req = {**first, "id": f"lang-{episode}-{seed}-{condition}", "seed": seed, "instruction": sentence}
                response = exchange(DEFAULT_SOCKET, req)
                if response.get("id") != req["id"] or "error" in response:
                    raise RuntimeError(str(response))
                waypoints = response["waypoints"]
                assert len(waypoints) == 10 and all(len(p) == 4 and all(math.isfinite(x) for x in p) for p in waypoints)
                if seed == first["seed"] and condition == "original":
                    assert waypoints == logged["native_chunk"], "Recorded first call could not be reproduced"
                rows.append(dict(episode=episode, seed=seed, condition=condition, request=req,
                                 waypoints=waypoints, body=body_endpoint(first["state"], waypoints[-1])))
    (HERE / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def summarize(rows):
    matched = {(r["episode"], r["seed"], r["condition"]): r["body"] for r in rows}
    groups = {}
    for positive, negative, dim, threshold in PAIRS:
        pairs = [(matched[ep, seed, positive][dim], matched[ep, seed, negative][dim])
                 for ep in FIRST_CLAUSES for seed in SEEDS]
        effects = [p - n for p, n in pairs]
        groups[dim] = {
            "n_pairs": len(pairs), "threshold": threshold,
            "positive_endpoint_median": statistics.median([p for p, _ in pairs]),
            "negative_endpoint_median": statistics.median([n for _, n in pairs]),
            "median_paired_effect": statistics.median(effects),
            "paired_effect_trimmed_mean_drop_one_per_tail": statistics.mean(sorted(effects)[1:-1]),
            "paired_effect_positive": sum(c > 0 for c in effects),
            "positive_signed_over_threshold": sum(p > threshold for p, n in pairs),
            "negative_signed_over_threshold": sum(n < -threshold for p, n in pairs),
            "both_signed_over_threshold": sum(p > threshold and n < -threshold for p, n in pairs),
            "paired_values": pairs,
        }
    hover = [r["body"] for r in rows if r["condition"] == "hover"]
    return {
        "input_images": 4, "seeds": SEEDS, "calls": len(rows),
        "pair_ordering": "positive endpoint signed axis minus negative endpoint signed axis; both_signed requires both individual directions with margin",
        "reference": "Original spawn RGB for current and reference, native height at spawn, identical random key per pair",
        "axes": groups,
        "hover": {"accepted_by_native_initial_stop": sum(b["travel_m"] < .2 and abs(b["yaw_right_deg"]) < 15 for b in hover),
                  "median_travel_m": statistics.median(b["travel_m"] for b in hover),
                  "median_abs_yaw_deg": statistics.median(abs(b["yaw_right_deg"]) for b in hover), "samples": hover},
        "original_vs_first_clause": [{"episode": ep, "seed": seed,
                                      "original": matched[ep, seed, "original"],
                                      "first_clause": matched[ep, seed, "first_clause"]}
                                     for ep in FIRST_CLAUSES for seed in SEEDS],
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["--summarize"]:
        rows = [json.loads(line) for line in (HERE / "predictions.jsonl").read_text().splitlines()]
    else:
        rows = run()
    doc = summarize(rows)
    (HERE / "summary.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({k: [v["both_signed_over_threshold"], v["paired_effect_positive"]] for k, v in doc["axes"].items()}))
