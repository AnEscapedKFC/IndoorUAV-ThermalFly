"""Fit the simulation frame from labelled floor/ceiling SfM observations and the RGB poses."""
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_tracks(rows, planar):
    """Use only the inner detection box, masking every competing object box."""
    clouds, groups, labels = [], [], []
    for index, row in enumerate(rows):
        stem = Path(row["source_name"]).stem
        detect = read_json(planar / "work/detections/f20bb1067579" / (stem + ".json"))
        with np.load(planar / "work/tracks" / (stem + ".npz")) as tracks:
            xy, z = tracks["xy"] + .5, tracks["z"]
        k = row["targets"]["lr"]
        width, height = 640, 480  # registered working resolution of the PlanarGS tracks
        uv = xy / [width, height]
        focal = np.array([k["fx"]/k["width"], k["fy"]/k["height"]])
        principal = np.array([k["cx"]/k["width"], k["cy"]/k["height"]])
        q = np.column_stack(((uv-principal)/focal*z[:, None], z))
        w = np.asarray(row["base_w2c"])
        clouds.append((q-w[:3, 3]) @ w[:3, :3])
        groups.append(np.full(len(z), index))
        selected = np.full(len(z), "", dtype="U7")
        for target in ("floor", "ceiling"):
            inner = np.zeros(len(z), bool)
            other = np.zeros(len(z), bool)
            for (cx, cy, bw, bh), name in zip(detect["boxes"], detect["labels"]):
                fraction = .45 if name == target else .5
                hit = (np.abs(uv-[cx, cy]) < np.array([bw, bh])*fraction).all(1)
                if name == target:
                    inner |= hit
                else:
                    other |= hit
            selected[inner & ~other] = target
        labels.append(selected)
    return np.concatenate(clouds), np.concatenate(groups), np.concatenate(labels)


def fit_plane(xyz, groups, approx_up, camera_level, direction, extent, rng):
    # Spread the vote across spatial cells so repeated frames do not dominate.
    threshold = .0075 * extent
    _, representatives = np.unique(np.floor(xyz/(1.5*threshold)), axis=0, return_index=True)
    pts = xyz[representatives]
    triplets = pts[rng.integers(len(pts), size=(5000, 3))]
    normals = np.cross(triplets[:, 1]-triplets[:, 0], triplets[:, 2]-triplets[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    normals *= np.where(normals @ approx_up >= 0, 1, -1)[:, None]
    levels = (triplets[:, 0]*normals).sum(1)
    good = (normals @ approx_up > np.cos(np.deg2rad(16))) & ((levels - normals @ camera_level)*direction > 3*threshold)
    normals, levels = normals[good], levels[good]
    if not len(levels):
        raise ValueError("No supported horizontal plane on the expected side of the cameras")
    score_points = pts[rng.choice(len(pts), min(5000,len(pts)), replace=False)]
    scores = np.empty(len(levels))
    for i in range(0,len(levels),128):
        sl = slice(i,i+128)
        scores[sl] = (np.abs(score_points @ normals[sl].T - levels[sl]) < threshold).sum(0)
    best = np.argmax(scores)
    n, level = normals[best], levels[best]
    for tolerance in (1.3*threshold, threshold, .85*threshold):
        inside = pts[np.abs(pts @ n-level) < tolerance]
        center = inside.mean(0)
        n = np.linalg.svd(inside-center, full_matrices=False)[2][-1]
        if n @ approx_up < 0:
            n = -n
        level = n @ center
    good = np.abs(xyz @ n-level) <= threshold
    bins = len(np.unique(groups[good]))
    if bins < 5 or good.sum() < 40:
        raise ValueError(f"Insufficient horizontal-plane support: {bins} views, {good.sum()} observations")
    return n, float(level), good, {"observations":int(good.sum()), "views":bins,
            "median_abs_residual_src":float(np.median(np.abs(xyz[good] @ n-level)))}


def fit_frame(scene, planar, height_m):
    inputs = scene / "inputs"
    manifest = read_json(inputs/"manifest.json")
    fingerprint = sha256(inputs/"manifest.json")
    if read_json(planar/"work/inputs.json")["source_manifest_sha256"] != fingerprint:
        raise ValueError("PlanarGS tracks do not match the bundle manifest")
    rows = [v for v in manifest["views"] if v["modality"] == "rgb" and v["split"] == "train"]
    poses = np.linalg.inv(np.array([r["base_w2c"] for r in rows]))
    centers = poses[:, :3, 3]
    center = np.median(centers, axis=0)
    with np.load(planar/"walls.npz") as w:
        wall_normals = w["planes"][:, :3]
    approx_up = np.linalg.svd(wall_normals, full_matrices=False)[2][-1]
    if approx_up @ poses[:, :3, 1].mean(0) > 0:
        approx_up = -approx_up
    points, groups, labels = sample_tracks(rows, planar)
    rng = np.random.default_rng(76941)
    planes, supports, meta = {}, {}, {}
    for name, side in (("floor",-1), ("ceiling",1)):
        ids = labels == name
        p = points[ids]
        n,level,inliers,meta[name] = fit_plane(p, groups[ids], approx_up, center, side, manifest["extent"], rng)
        planes[name] = (n,level)
        supports[name] = p[inliers]
        meta[name]["normal"] = n.tolist()
        meta[name]["projection_src"] = level
    up,floor = planes["floor"]
    prior = inputs/"viewer_floor.json"
    if prior.is_file():
        existing = read_json(prior)
        if existing["manifest_sha256"] != fingerprint:
            raise ValueError("Mismatched existing floor calibration")
        normal = np.array(existing["upward_floor_plane"])
        up, floor = normal[:3]/np.linalg.norm(normal[:3]), -normal[3]/np.linalg.norm(normal[:3])
        meta["viewer_floor_median_abs_residual_src"] = float(np.median(abs(supports["floor"] @ up-floor)))
    ceiling = float(np.median(supports["ceiling"] @ up))
    scale = height_m/(ceiling-floor)
    if scale <= 0 or planes["ceiling"][0] @ up < np.cos(np.deg2rad(8)):
        raise ValueError(f"Floor and ceiling disagree: {meta}")
    forward = poses[0,:3,2].copy()
    forward -= up*(forward @ up)
    forward /= np.linalg.norm(forward)
    rotation = np.stack((forward, np.cross(up, forward), up))
    origin = center + (floor-center@up)*up
    metric = (points-origin) @ rotation.T * scale
    # Limit the map to supported indoor geometry, including every camera origin.
    footprint = metric[(metric[:,2] > -.2) & (metric[:,2] < height_m+.2)]
    cameras = (centers-origin) @ rotation.T * scale
    lo,hi = np.quantile(footprint, [.002,.998], axis=0)
    lo,hi = np.minimum(lo,cameras.min(0))-.5, np.maximum(hi,cameras.max(0))+.5
    lo[2],hi[2]=-.20,height_m+.20
    lo,hi=np.floor(lo*10)/10, np.ceil(hi*10)/10
    return dict(rotation=rotation.tolist(), origin=origin.tolist(), meters_per_source_unit=scale,
                bounds_m=[lo.tolist(), hi.tolist()], floor_src=floor, ceiling_src=ceiling,
                assumed_ceiling_height_m=height_m, scale_status="assumed floor-to-ceiling height; no surveyed length",
                plane_fit=meta, source_manifest_sha256=fingerprint,
                camera_height_range_m=np.quantile(cameras[:,2],[0,.5,1]).tolist(),
                transform="p_sim = scale * R @ (p_source - origin); Z up, floor Z=0")
