"""In the same target-specific instruction insert just one left/right bearing word."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge_client import DEFAULT_SOCKET, exchange


def main():
    folder = ROOT / "diagnoses/image"
    original = [json.loads(s) for s in (folder / "raw.jsonl").read_text().splitlines()]
    records = []
    for row in original:
        if row["mode"] != "fixed_reference" or row["goal"] not in {"ac", "sofa", "laptop"}:
            continue
        for sign, arm in [(row["side"],"correct_hint"), (-row["side"],"contradictory_hint")]:
            req = dict(row["request"])
            word = "left" if sign == -1 else "right"
            assert req["instruction"].startswith("Turn in place")
            req["instruction"] = req["instruction"].replace("Turn in place",f"Turn {word} in place",1)
            req["id"] += "-" + arm
            ans = exchange(DEFAULT_SOCKET, req)
            assert ans["id"] == req["id"] and ans["seed"] == req["seed"]
            delta = math.degrees(math.remainder(ans["waypoints"][-1][3] - req["state"][3], math.tau))
            records.append(dict(view=row["view"], goal=row["goal"], seed=req["seed"], arm=arm, side=row["side"],
                yaw_right_deg=delta, unhinted_yaw_right_deg=row["yaw_right_deg"],
                correct=delta*row["side"]>0, correct_above_2deg=delta*row["side"]>2,
                previously_correct=row["sign_ok"], request=req, response=ans))
    assert len(records)==36
    (folder/"hinted.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    result = {}
    for arm in ["correct_hint","contradictory_hint"]:
        rows = [r for r in records if r["arm"]==arm]
        result[arm] = dict(n=len(rows),correct=sum(r["correct"] for r in rows),
            above_2deg=sum(r["correct_above_2deg"] for r in rows),unhinted=sum(r["previously_correct"] for r in rows))
    (folder/"hinted_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__ == "__main__":
    main()
