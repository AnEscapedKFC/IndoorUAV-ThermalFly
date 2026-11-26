"""Select inspection poses from observed framing and repeated part evidence."""

import numpy as np

from metric_flight.control import angle
from metric_flight.depth_guidance import camera
from metric_flight.observation_memory import ObservationMemory
from metric_flight.instance_mask import instance_mask


class HistoryInspection:
    def __init__(self, folder, noun, instruction):
        self.memory = ObservationMemory(folder, noun, instance_mask())
        words = instruction.lower()
        self.intent = (
            "overview"
            if any(term in words for term in ("whole", "overview", "entire"))
            else "detail"
        )
        self.lost = 0
        self.ready = 0
        self.previous_anchor = None
        self.best = None
        self.last_move = None
        self.probe = 1
        self.scan = False

    def step(self, p, yaw, v, raw, det, rig, dt, image, depth, *, receipt):
        obs, point = self.memory.observe(
            image, depth, rig, det["boxes"], receipt["frame_id"], receipt["sim_time"], p
        )
        self.lost = self.lost + 1 if point is None else 0
        if point is None:
            self.ready = 0
            return (
                np.r_[p, yaw],
                False,
                dict(phase="observation_lost", observation_id=obs["id"]),
            )
        # Prefer recent, consistent object centroids; a clipped box can shift
        # from one part to another without the object actually moving.
        l, t, r, b = det["boxes"][0]["xyxy"]
        clipped = min(l, t, 1 - r, 1 - b) < 0.018
        if self.previous_anchor is None:
            self.previous_anchor = point
        elif not clipped:
            self.previous_anchor = 0.55 * self.previous_anchor + 0.45 * point
        d = self.previous_anchor - p
        forward = d[:2] / max(np.linalg.norm(d[:2]), 1.0e-6)
        c, rot, k = camera(rig)
        optical = rot.T @ (self.previous_anchor - c)
        uv = k @ optical
        uv = uv[:2] / max(uv[2], 1e-5) / [rig.width, rig.height]
        target_yaw = yaw + np.clip(
            angle(np.degrees(np.arctan2(d[1], d[0])) - yaw), -9.0, 9.0
        )
        pitch_error = np.degrees(np.arctan((uv[1] - 0.5) * rig.height / k[1, 1]))
        requested_pitch = rig.pitch_deg - pitch_error
        pitch = float(np.clip(requested_pitch, -62.0, 45.0))
        width = r - l
        # Framing is dimensionless; there are no target coordinates or range setpoints.
        desired = 0.64 if self.intent == "overview" else 0.41
        part_pixels = np.array(obs["part_pixels"][2:])
        part_fraction = part_pixels / max(1, sum(obs["part_pixels"]))
        observed_both = bool(np.all(part_fraction > 0.045))
        verified = obs["matched_edge_pixels"] >= 6 and obs["consistent_pixels"] >= 70
        quality = min(width / desired, 1.0) * (
            0.4 + 0.6 * min(1.0, part_fraction.min() / 0.13)
        )
        quality *= 0.6 + 0.4 * min(1.0, obs["quality"] / 0.24)
        if clipped:
            quality *= 0.65
        if verified and (self.best is None or quality > self.best[0]):
            self.best = (quality, p.copy(), obs["id"])
        center_ok = abs(uv[0] - 0.5) < 0.14 and abs(uv[1] - 0.5) < 0.15
        close_enough = width >= desired * 0.88
        if self.intent == "detail" and verified:
            close_enough |= part_pixels.min() >= 0.018 * rig.width * rig.height
        ready = (
            observed_both and verified and center_ok and close_enough and not clipped
        )
        self.ready = self.ready + 1 if ready else 0
        step = np.clip((desired - width) * 0.8, -0.16, 0.20)
        max_step = np.clip(np.dot(raw[:2] - p[:2], forward), 0.105, 0.20)
        step = min(step, max_step)
        # Verify a comparable second view before holding for the final image.
        phase = "approach"
        movement = np.r_[forward * step, 0.0]
        # Share the vertical viewing angle with the body before the gimbal hits
        # its stop. Moving down makes a low target less steeply below the camera.
        tilt_err = requested_pitch - np.clip(requested_pitch, -40.0, 26.0)
        movement[2] = np.clip(0.005 * tilt_err, -0.08, 0.08)
        if not center_ok:
            movement[:2] *= 0.55
        if clipped:
            movement[:2] = -0.09 * forward
            phase = "widen"
        elif ready:
            movement[:] = 0.0
            phase = "terminal_hold"
        elif (
            self.best is not None and width > desired and quality < 0.72 * self.best[0]
        ):
            delta = self.best[1] - p
            movement = delta * min(1.0, 0.14 / max(np.linalg.norm(delta), 1.0e-6))
            phase = "restore_view"
        elif close_enough and abs(tilt_err) < 10.0:
            side = np.array([-forward[1], forward[0]])
            # Low/occluded lower parts merit a higher line of sight. Reverse a
            # height probe when the following RGB-D observations show worse parts.
            if self.last_move is not None and quality < self.last_move - 0.06:
                self.probe *= -1
            if not observed_both or obs["occluded_cells"] > max(
                10, 0.22 * obs["visible_cells"]
            ):
                movement[2] = 0.065 * self.probe
                phase = "change_height"
            else:
                movement[:2] += 0.07 * side * self.probe
                phase = "verify_detail"
            self.last_move = quality
        return (
            np.r_[p + movement, target_yaw],
            bool(self.ready >= 3 and np.linalg.norm(v) < 0.13),
            dict(
                phase=phase,
                observation_id=obs["id"],
                range_est_m=float(np.linalg.norm(d)),
                anchor=dict(
                    world=self.previous_anchor.tolist(), confirmed=True, uv=uv.tolist()
                ),
                history_reference=None if self.best is None else self.best[2],
                quality=float(quality),
                part_fraction=part_fraction.tolist(),
                cross_view_verified=bool(verified),
                pitch_request=pitch,
            ),
        )
