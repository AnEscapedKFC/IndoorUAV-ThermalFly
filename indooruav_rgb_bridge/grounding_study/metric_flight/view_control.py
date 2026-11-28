"""Coordinate camera pitch, body height and RGB framing in the calibrated flight frame."""
import numpy as np

from metric_flight.control import angle
from metric_flight.depth_guidance import camera
from metric_flight.instance_mask import instance_mask
from metric_flight.object_goal import centered
from metric_flight.observation_memory import ObservationMemory, measured_parts, segment

PITCH_LIMITS = (-60., 30.)
# Elevation of the camera-to-observed-surface ray relative to world horizontal.
VIEW_BANDS = {"level": (-8., 8.), "below": (12., 32.), "above": (-48., -18.)}


def view_intent(instruction):
    text = instruction.lower()
    for mode, phrases in (("below", ("from below", "look up")), ("above", ("from above", "look down")),
                          ("level", ("level view", "horizontally", "at eye level"))):
        if any(phrase in text for phrase in phrases):
            return mode
    return "unconstrained"


def framing(box, area_range=(.20, .60)):
    if box is None:
        return dict(area=None, uv=None, centered=False, clipped=False, framing_ok=False)
    left, top, right, bottom = np.clip(box, 0., 1.)
    uv = [(left + right) / 2, (top + bottom) / 2]
    area = float((right - left) * (bottom - top))
    clipped = min(left, top, 1 - right, 1 - bottom) < .009
    return dict(area=area, uv=uv, centered=centered(*uv), clipped=bool(clipped),
                framing_ok=bool(area_range[0] - 1e-12 <= area <= area_range[1] + 1e-12 and centered(*uv) and not clipped))


def elevation(direction):
    return float(np.degrees(np.arctan2(direction[2], np.linalg.norm(direction[:2]))))


def pointing_request(rig, world):
    c, rotation, _ = camera(rig)
    local = rotation.T @ (world - c)
    error = np.degrees(np.arctan2(-local[1], local[2]))
    return float(np.clip(rig.pitch_deg + error, *PITCH_LIMITS))


def body_height_delta(surface, center, horizontal_after, band, mechanical):
    """Keep the chosen sightline as range changes, including camera mounting height."""
    actual = elevation(surface - center)
    # With no language constraint, transfer excess excursion back to the body.
    if band is None:
        lower, upper = np.array(PITCH_LIMITS) * .65
        desired = actual - (mechanical - np.clip(mechanical, lower, upper))
    else:
        desired = np.mean(band)
    return float(surface[2] - horizontal_after * np.tan(np.deg2rad(desired)) - center[2])


class CoordinatedInspection:
    def __init__(self, folder, noun, instruction, *, single_frame=False):
        self.single_frame = single_frame
        self.memory = ObservationMemory(folder, noun, instance_mask())
        self.ready = 0
        self.lost = 0
        self.scan = False
        self.view = view_intent(instruction)
        self.band = VIEW_BANDS.get(self.view)
        self.part = 3 if "outlet" in instruction.lower() and self.view == "below" else None
        self.aim = None
        self.surface = None

    def inspect(self, point, parts, box, rig):
        from metric_flight.observation_window import INSPECTION_AREA
        frame = framing(box, INSPECTION_AREA) if self.single_frame else framing(box)
        counts = np.array([part["pixels"] for part in parts])
        both = bool(counts.min() >= max(32, .04 * counts.sum()))
        landmark = parts[self.part - 2]["world"] if self.part else None
        # A missing seam must not silently switch the acceptance reference to the housing.
        surface = np.asarray(landmark) if landmark is not None else point
        los = elevation(surface - camera(rig)[0]) if surface is not None else None
        angle_ok = los is not None and (self.band is None or self.band[0] <= los <= self.band[1])
        return surface, dict(**frame, los_deg=los, view=self.view, view_ok=bool(angle_ok),
                             parts_ok=both and (self.part is None or landmark is not None))

    def step(self, p, yaw, v, raw, det, rig, dt, image, depth, *, receipt):
        obs, point = self.memory.observe(image, depth, rig, det["boxes"], receipt["frame_id"], receipt["sim_time"], p)
        box = det["boxes"][0]["xyxy"] if det["boxes"] else None
        surface, metrics = self.inspect(point, obs["part_samples"], box, rig)
        info = dict(**metrics, observation_id=obs["id"], phase="observation_lost")
        if point is None:
            self.lost += 1
            self.ready = 0
            self.aim = None
            return np.r_[p, yaw], False, info
        self.lost = 0
        if self.surface is None:
            self.surface = surface.copy()
        elif not metrics["clipped"]:
            self.surface = .55 * self.surface + .45 * surface
        c, rotation, k = camera(rig)
        # Place the actual box center on the optical axis. Reproject this observed
        # bearing at flight rate between detections to compensate for body motion.
        u, b = np.array(metrics["uv"]) * [rig.width, rig.height]
        ray = rotation @ np.linalg.solve(k, [u, b, 1.])
        self.aim = c + ray * np.dot(point - c, rotation[:, 2])
        pitch = pointing_request(rig, self.aim)
        horizontal = self.surface[:2] - c[:2]
        distance = max(.02, np.linalg.norm(horizontal))
        direction = horizontal / distance
        dx, dy = self.aim[:2] - c[:2]
        bearing = np.degrees(np.arctan2(dy, dx))
        target_yaw = yaw + np.clip(angle(bearing - yaw), -9., 9.)
        # Perspective area scales approximately as inverse squared range.
        max_step = np.clip(np.dot(raw[:2]-p[:2], direction), .10, .19)
        advance = np.clip(distance * (1 - np.sqrt(metrics["area"] / .31)), -.13, max_step)
        if metrics["clipped"]:
            advance = min(advance, -.075)
        dz = body_height_delta(self.surface, c, max(.05, distance - advance), self.band, pitch)
        delta_z = np.clip(dz, -.095, .095)
        # Do not close range faster than the body can recover the requested sightline.
        if abs(dz) > .20 and advance > 0:
            advance *= .45
        verified = obs["matched_edge_pixels"] >= 6 and obs["consistent_pixels"] >= 70
        ready = metrics["framing_ok"] and metrics["view_ok"] and metrics["parts_ok"] and verified
        self.ready = self.ready + 1 if ready else 0
        movement = np.r_[advance * direction, delta_z] if not ready else np.zeros(3)
        info.update(phase="terminal_hold" if ready else "adjust_view", cross_view_verified=bool(verified),
                    history_surface=self.surface.tolist(), anchor=dict(world=self.aim.tolist(), confirmed=True),
                    pitch_request=pitch, requested_dz_m=float(delta_z), requested_forward_m=float(advance))
        stop = ready if self.single_frame else self.ready >= 3 and np.linalg.norm(v) < .13
        return np.r_[p + movement, target_yaw], bool(stop), info

    def hold_metrics(self, image, depth, rig, box):
        labels = segment(np.rint(image * 255).astype(np.uint8), box, self.memory.noun, self.memory.instance_mask)
        _, _, _, xyz, parts = measured_parts(labels, depth, *camera(rig))
        point = np.median(xyz, axis=0) if len(xyz) else None
        _, metrics = self.inspect(point, parts, box, rig)
        return dict(**metrics, xyxy=box, pitch_deg=float(rig.pitch_deg),
                    optical_elevation_deg=elevation(camera(rig)[1][:, 2]))
