"""Causal, instance-local RGB-D surface observations in metric flight coordinates."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from metric_flight.depth_guidance import camera


PARTS = {
    "sofa": ("backrest", "seat"),
    "laptop": ("screen", "keyboard_deck"),
    "ac": ("front_panel", "outlet_band"),
}
CELL = 0.06  # metres, indices are in the world frame


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )


def segment(rgb, box, noun):
    """RGB foreground cut within a grounded box, then fit the visible part boundary.

    The named regions are appearance-based estimates, not annotated part masks.
    Unreliable or absent seams leave the foreground labelled 'object only'.
    """
    h, w = rgb.shape[:2]
    labels = np.zeros((h, w), np.uint8)
    if box is None:
        return labels
    l, t, r, b = np.rint(np.array(box) * [w, h, w, h]).astype(int)
    l, t, r, b = max(0, l), max(0, t), min(w, r), min(h, b)
    if r - l < 12 or b - t < 8:
        return labels
    # Add an outer background border even when a detector box touches the image edge.
    pad = 6
    patch = cv2.copyMakeBorder(
        rgb[max(0, t - pad) : min(h, b + pad), max(0, l - pad) : min(w, r + pad)],
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_REPLICATE,
    )
    x = pad + min(l, pad)
    y = pad + min(t, pad)
    mask = np.zeros(patch.shape[:2], np.uint8)
    rect = (x, y, r - l, b - t)
    cv2.setRNGSeed(42)
    cv2.grabCut(patch, mask, rect, None, None, 2, cv2.GC_INIT_WITH_RECT)
    fg = (mask[y : y + b - t, x : x + r - l] & 1).astype(bool)
    if not fg.any():
        return labels
    # Estimate the upholstered seam, display hinge, or outlet lip from row-wise
    # luminance edges. Missing edges stay unclassified, not a fixed box split.
    roi = cv2.cvtColor(rgb[t:b, l:r], cv2.COLOR_RGB2GRAY)
    edge = np.abs(cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3))
    low, high = {"sofa": (0.35, 0.77), "laptop": (0.40, 0.80), "ac": (0.52, 0.90)}[noun]
    hh, ww = fg.shape
    xs = []
    ys = []
    for a in np.array_split(np.arange(ww), min(12, ww)):
        if len(a) == 0:
            continue
        y0, y1 = int(hh * low), max(int(hh * low) + 1, int(hh * high))
        support = fg[y0:y1, a].mean(axis=1)
        strength = (edge[y0:y1, a] * fg[y0:y1, a]).mean(axis=1)
        strength[support < 0.42] = 0
        peak = int(np.argmax(strength))
        if strength[peak] >= 18:
            xs.append(float(a.mean()))
            ys.append(y0 + peak)
    region = labels[t:b, l:r]
    region[fg] = 1
    if len(xs) >= 4:
        xs, ys = np.asarray(xs), np.asarray(ys)
        slope, offset = np.polyfit(xs, ys, 1)
        good = np.abs(ys - (slope * xs + offset)) <= max(3, 0.10 * hh)
        if good.sum() >= 4 and np.ptp(xs[good]) > 0.30 * ww:
            slope, offset = np.polyfit(xs[good], ys[good], 1)
            boundary = np.clip(slope * np.arange(ww) + offset, low * hh, high * hh)
            region[fg & (np.arange(hh)[:, None] < boundary)] = 2
            region[fg & (np.arange(hh)[:, None] >= boundary)] = 3
    return labels


def project(points, center, axes, k, shape):
    optical = (points - center) @ axes
    z = optical[:, 2]
    uv = optical @ k.T
    uv = np.rint(uv[:, :2] / np.maximum(z[:, None], 1e-5)).astype(np.int64)
    h, w = shape
    inside = (
        (z > 0.08) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )
    return uv[:, 0].clip(0, w - 1), uv[:, 1].clip(0, h - 1), z, inside


class ObservationMemory:
    def __init__(self, folder, noun):
        self.root = Path(folder) / "observations"
        self.root.mkdir()
        self.noun = noun
        self.keys = []
        self.views = []
        self.cells = {}
        self.best = {}
        self.parts = set()
        self.last_occluded = set()
        self.records = []
        self.best_detail = 0.0
        self.best_seen = {}

    def observe(self, rgb, depth, rig, boxes, frame, time, body):
        image = np.rint(rgb * 255).astype(np.uint8)
        depth = np.asarray(depth, np.float32)
        box = boxes[0]["xyxy"] if boxes else None
        labels = segment(image, box, self.noun)
        c, R, K = camera(rig)
        rows, cols = np.nonzero((labels > 0) & np.isfinite(depth) & (depth > 0.08))
        parts = labels[rows, cols]
        rays = np.column_stack((cols, rows, np.ones(len(rows)))) @ np.linalg.inv(K).T
        xyz = c + (rays * depth[rows, cols, None]) @ R.T
        edge = cv2.Canny(image, 45, 110)
        near_edge = cv2.dilate(edge, np.ones((5, 5), np.uint8)) > 0
        is_edge = edge[rows, cols] > 0
        # Use only already observed frames with translational parallax and a
        # comparable optical direction. A repeated hover image cannot verify detail.
        options = []
        for view in self.views:
            delta = np.linalg.norm(c - view["center"])
            if 0.055 <= delta <= 0.75 and R[:, 2] @ view["axes"][:, 2] > 0.90:
                options.append((delta, view))
        consistent = np.zeros(len(rows), bool)
        edge_match = consistent.copy()
        paired = None
        if options and len(rows):
            _, view = min(options, key=lambda a: abs(a[0] - 0.20))
            paired = view["id"]
            u, v, z, inside = project(xyz, view["center"], view["axes"], K, depth.shape)
            prior = view["depth"][v, u]
            consistent = (
                inside
                & (parts >= 2)
                & (view["mask"][v, u] == parts)
                & (np.abs(prior - z) < 0.055 + 0.018 * z)
            )
            edge_match = consistent & is_edge & view["near_edge"][v, u]
        grid, inv, counts = np.unique(
            np.floor(xyz / CELL).astype(np.int32),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        ids = np.array(
            [self.cells.setdefault(tuple(key), len(self.cells)) for key in grid],
            np.int32,
        )

        def counts_for(mask):
            return np.bincount(inv, weights=mask, minlength=len(grid)).astype(np.int32)

        good = counts_for(consistent)
        edges = counts_for(is_edge)
        matched = counts_for(edge_match)
        part_counts = np.column_stack(
            [counts_for(parts == label) for label in (1, 2, 3)]
        )
        quality = (
            np.log1p(counts)
            * (good / np.maximum(counts, 1))
            * (matched / np.maximum(edges, 3))
        )
        # Check current depth at previously mapped cell centers. An out-of-FOV
        # cell is unobserved; a nearer measured surface is positive occlusion evidence.
        prev_keys = np.asarray(list(self.cells.keys()), dtype=float).reshape(-1, 3)
        u, v, z, inside = project((prev_keys + 0.5) * CELL, c, R, K, depth.shape)
        observed = depth[v, u]
        occluded = np.flatnonzero(
            inside & np.isfinite(observed) & (observed > 0.08) & (observed < z - 0.12)
        )
        occluded = np.setdiff1d(occluded, ids, assume_unique=False)
        occ = set(occluded.tolist())
        visible_parts = {i for i in (2, 3) if np.count_nonzero(parts == i) >= 32}
        score = float(quality.mean()) if len(quality) else 0.0
        reasons = []
        if ids.size:
            if not self.views:
                reasons.append("initial")
            if visible_parts - self.parts:
                reasons.append("new_part")
            if self.views:
                last = self.views[-1]
                if np.linalg.norm(c - last["center"]) >= 0.25 or last["axes"][:, 2] @ R[
                    :, 2
                ] < np.cos(np.deg2rad(11)):
                    reasons.append("viewpoint")
                if len(occ ^ self.last_occluded) >= max(5, 0.15 * len(self.cells)):
                    reasons.append("occlusion_change")
            if score > max(0.025, 1.15 * self.best_detail):
                reasons.append("detail_improvement")
            # A part can improve even when the mean for the whole object falls.
            if len(self.views) and sum(
                q > max(0.12, 1.35 * self.best.get(int(i), (0.0, None))[0])
                for i, q in zip(ids, quality)
            ) >= max(4, 0.12 * len(ids)):
                reasons.append("surface_detail_improvement")
        index = len(self.records)
        name = f"{index:04d}"
        meta = dict(
            id=index,
            receipt_frame=int(frame),
            time_s=float(time),
            body_xyz=np.asarray(body).tolist(),
            pitch_deg=float(rig.pitch_deg),
            rgb_w2c_native=rig.W.tolist(),
            world_camera_center=c.tolist(),
            world_from_optical_axes=R.tolist(),
            K=K.tolist(),
            pair=paired,
            reasons=reasons,
            part_pixels=np.bincount(parts, minlength=4).tolist(),
            consistent_pixels=int(consistent.sum()),
            matched_edge_pixels=int(edge_match.sum()),
            visible_cells=len(ids),
            occluded_cells=len(occ),
            quality=score,
        )
        np.savez_compressed(
            self.root / (name + "_surface.npz"),
            visible=ids,
            pixels=counts,
            part_pixels=part_counts,
            depth_agree=good,
            edge_pixels=edges,
            matched_edges=matched,
            occluded=occluded,
        )
        if reasons:
            Image.fromarray(image).save(self.root / (name + "_rgb.png"))
            np.savez_compressed(self.root / (name + "_depth.npz"), depth_m=depth)
            Image.fromarray(labels).save(self.root / (name + "_mask.png"))
            self.views.append(
                dict(
                    id=index,
                    center=c.copy(),
                    axes=R.copy(),
                    depth=depth.copy(),
                    mask=labels,
                    near_edge=near_edge,
                )
            )
            self.keys.append(index)
            self.best_detail = max(self.best_detail, score)
            self.parts.update(visible_parts)
            self.last_occluded = occ
            for i, q in zip(ids, quality):
                i = int(i)
                if i not in self.best or q > self.best[i][0]:
                    self.best[i] = (float(q), index)
        for i, q, count in zip(ids, quality, counts):
            i = int(i)
            rank = (float(q), int(count))
            if i not in self.best_seen or rank > self.best_seen[i][0]:
                self.best_seen[i] = (rank, index)
        self.records.append(meta)
        centroid = np.median(xyz, axis=0) if len(xyz) else None
        return meta, centroid

    def finish(self):
        coords = list(self.cells.keys())
        best = [self.best.get(i, (0.0, -1)) for i in range(len(coords))]
        np.savez_compressed(
            self.root / "surfaces.npz",
            world_voxel=np.asarray(coords, np.int32).reshape(-1, 3),
            best_observation=np.array(
                [self.best_seen[i][1] for i in range(len(coords))], int
            ),
            best_keyframe=np.array([x[1] for x in best], int),
            best_quality=np.array([x[0] for x in best]),
        )
        write_json(
            self.root / "index.json",
            dict(
                target=self.noun,
                cell_m=CELL,
                labels=["background", "unassigned", *PARTS[self.noun]],
                keyframes=self.keys,
                frames=self.records,
            ),
        )
