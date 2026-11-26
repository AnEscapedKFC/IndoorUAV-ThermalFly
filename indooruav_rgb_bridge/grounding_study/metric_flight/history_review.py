"""Verify synchronized keyframes and produce an RGB evidence sheet per closed-loop flight."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metric_flight.observation_memory import write_json


def read(path):
    return json.loads(path.read_text())


def keyframe_sheet(folder, index):
    """Show the strongest saved observation, its mask, and measured axial depth."""
    if not index["keyframes"]:
        return
    entry = max(
        (index["frames"][i] for i in index["keyframes"]), key=lambda row: row["quality"]
    )
    stem = f"{entry['id']:04d}"
    base = folder / "observations"
    rgb = np.asarray(Image.open(base / (stem + "_rgb.png")))
    label = np.asarray(Image.open(base / (stem + "_mask.png")))
    colors = np.array(
        [[0, 0, 0], [150, 150, 150], [48, 175, 229], [250, 170, 55]], np.uint8
    )
    overlay = rgb.copy()
    foreground = label > 0
    overlay[foreground] = 0.42 * rgb[foreground] + 0.58 * colors[label[foreground]]
    depth = np.load(base / (stem + "_depth.npz"))["depth_m"]
    valid = np.isfinite(depth) & (depth > 0)
    low, high = np.percentile(depth[valid], [2, 98])
    value = np.zeros_like(depth, dtype=np.uint8)
    value[valid] = np.rint(
        255 * np.clip((depth[valid] - low) / max(high - low, 0.01), 0, 1)
    ).astype(np.uint8)
    depth_rgb = cv2.applyColorMap(value, cv2.COLORMAP_VIRIDIS)[:, :, ::-1].copy()
    depth_rgb[~valid] = 0
    sheet = Image.new("RGB", (960, 272), "white")
    draw = ImageDraw.Draw(sheet)
    captions = [
        f"RGB  t={entry['time_s']:.2f}s  pitch={entry['pitch_deg']:.1f}",
        f"blue: {index['labels'][2]} / orange: {index['labels'][3]}",
        f"axial depth, dark to light: {low:.1f} to {high:.1f} m",
    ]
    for n, (im, caption) in enumerate(zip((rgb, overlay, depth_rgb), captions)):
        sheet.paste(
            Image.fromarray(im).resize((320, 240), Image.Resampling.LANCZOS),
            (320 * n, 0),
        )
        draw.text((320 * n + 4, 246), caption, fill="black")
    sheet.save(folder / "keyframe_review.jpg", quality=91)


def audit(folder):
    idx = read(folder / "observations/index.json")
    rec = read(folder / "recording.json")
    ep = read(folder / "episode.json")
    frames = read(folder / "frames.json")
    decisions = read(folder / "decisions.json")
    poses = [
        json.loads(row)
        for row in (folder / "video_frames.jsonl").read_text().splitlines()
    ]
    if (
        len(frames) != len(poses)
        or len(poses) != rec["frames"]
        or rec["rgb_wh"] != [640, 480]
    ):
        raise ValueError("Incomplete sensor stream")
    if not all(v["passed"] for v in rec["streams"].values()):
        raise ValueError("Recording verification failed")
    by_id = {r["receipt_frame"]: (r, f) for r, f in zip(poses, frames)}
    obsdir = folder / "observations"
    catalog = np.load(obsdir / "surfaces.npz")
    n = len(catalog["world_voxel"])
    assert len(idx["frames"]) == len(decisions)
    keys = set(idx["keyframes"])
    assert len(keys) == len(idx["keyframes"])
    occluded = set()
    verified = set()
    for row in idx["frames"]:
        i = row["id"]
        stem = f"{i:04d}"
        decision = decisions[i]
        pose, f = by_id[row["receipt_frame"]]
        assert (
            abs(decision["t"] - row["time_s"]) < 1e-6
            and abs(pose["t"] - row["time_s"]) < 1e-6
        )
        assert np.allclose(pose["rgb_w2c"], row["rgb_w2c_native"])
        assert abs(f["rig_pitch_deg"] - row["pitch_deg"]) < 1e-6
        assert np.allclose(f["xyz"], row["body_xyz"])
        assert (i in keys) == bool(row["reasons"])
        trace = np.load(obsdir / (stem + "_surface.npz"))
        assert (trace["pixels"] == trace["part_pixels"].sum(axis=1)).all()
        assert trace["pixels"].sum() == sum(row["part_pixels"])
        for col in ("visible", "occluded"):
            assert ((trace[col] >= 0) & (trace[col] < n)).all()
        assert not np.intersect1d(trace["visible"], trace["occluded"]).size
        occluded.update(trace["occluded"].tolist())
        verified.update(trace["visible"][trace["matched_edges"] > 0].tolist())
        if i in keys:
            im = np.asarray(Image.open(obsdir / (stem + "_rgb.png")))
            dep = np.load(obsdir / (stem + "_depth.npz"))["depth_m"]
            mask = np.asarray(Image.open(obsdir / (stem + "_mask.png")))
            valid = (mask > 0) & np.isfinite(dep) & (dep > 0.08)
            assert dep.shape == mask.shape == im.shape[:2] == (480, 640)
            assert hashlib.sha256(im.tobytes()).hexdigest() == pose["rgb_sha256"]
            assert np.array_equal(
                np.bincount(mask[valid], minlength=4), row["part_pixels"]
            )
        if row["pair"] is not None:
            pair = row["pair"]
            assert pair in keys and pair < i
            baseline = np.linalg.norm(
                np.subtract(
                    idx["frames"][pair]["world_camera_center"],
                    row["world_camera_center"],
                )
            )
            assert 0.055 - 1e-6 <= baseline <= 0.75 + 1e-6
    assert set(catalog["best_keyframe"].tolist()) <= (keys | {-1})
    assert (
        (catalog["best_observation"] >= 0)
        & (catalog["best_observation"] < len(decisions))
    ).all()
    travel = np.array([f["cumulative_m"] for f in frames])
    sample = [
        0,
        *[int(np.argmin(abs(travel - travel[-1] * s))) for s in (0.3, 0.65)],
        len(frames) - 1,
    ]
    sample += [max(0, len(frames) - 31), max(0, len(frames) - 11)]
    sample = sorted(sample)
    cap = cv2.VideoCapture(str(folder / "rgb_fpv.mp4"))
    sheet = Image.new("RGB", (3 * 384, 2 * 315), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        for slot, frame in enumerate(sample):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {frame}")
            im = Image.fromarray(bgr[:, :, ::-1]).resize(
                (384, 288), Image.Resampling.LANCZOS
            )
            left, top = slot % 3 * 384, slot // 3 * 315
            sheet.paste(im, (left, top))
            draw.text(
                (left + 5, top + 290),
                f"{folder.name}   t={frames[frame]['t']:.2f}s   path={travel[frame]:.2f}m",
                fill="black",
            )
    finally:
        cap.release()
    sheet.save(folder / "rgb_review.jpg", quality=91)
    keyframe_sheet(folder, idx)
    hold = ep["hold"]
    held = bool(
        ep["reason"] == "stop"
        and len(hold) == 41
        and max(np.linalg.norm(np.subtract(x["xyz"], hold[0]["xyz"])) for x in hold)
        <= 0.12
        and max(x["speed"] for x in hold[-10:]) <= 0.10
    )
    result = dict(
        case=folder.name,
        reason=ep["reason"],
        travel_m=round(float(travel[-1]), 3),
        flight_s=round(frames[-1]["t"], 2),
        keyframes=len(keys),
        decisions=len(decisions),
        unique_cells=n,
        cells_with_verified_edges=len(verified),
        cells_ever_occluded=len(occluded),
        stable_two_second_hold=held,
        collisions=sum(bool(f["collision"]) for f in frames),
        outside_samples=sum(not f["inside"] for f in frames),
        first_detection=bool(decisions[0]["detection"]["boxes"]),
    )
    write_json(folder / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    for folder in parser.parse_args().folders:
        print(audit(folder), flush=True)


if __name__ == "__main__":
    main()
