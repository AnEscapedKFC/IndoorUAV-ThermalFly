"""Terminal object observation using target-relative range and RGB centroid."""
import numpy as np

from metric_flight.control import Inspection, unit

MIN_RANGE_M = 0.50
MAX_RANGE_M = 1.50
# Stop as soon as the current view is acceptable, with room for estimation error.
RANGE_MARGIN_M = .10
RESULT_FOLDER = 'room_ac_0p5_1p5'
CENTER_MIN = 0.30
CENTER_MAX = 0.70
OBJECT_INSTRUCTION = (
    f" Finish hovering {MIN_RANGE_M:g} to {MAX_RANGE_M:g} meters from the air conditioner, "
    "with the air conditioner centered in the image."
)


def centered(x, y):
    return bool(CENTER_MIN <= x <= CENTER_MAX and CENTER_MIN <= y <= CENTER_MAX)


def in_band(distance):
    return bool(MIN_RANGE_M <= distance <= MAX_RANGE_M)


class StandoffInspection(Inspection):
    """Complete the requested viewing motion, then regulate the final observation."""
    def __init__(self, text):
        super().__init__(text)
        self.finishing = False
        self.valid_time = 0.

    def step(self, p, yaw, v, raw, det, rig, dt, image):
        target, _, info = super().step(p, yaw, v, raw, det, rig, dt, image)
        if info['phase'] == 'observation_lost':
            self.valid_time = 0.
            return target, False, info
        if info['phase'] == 'settle':
            self.finishing = True
        if not self.finishing:
            return target, False, info

        # Use the appearance-based position from the current image and recent rays.
        relative = self.estimate - p
        distance = info['range_est_m']
        near = MIN_RANGE_M + RANGE_MARGIN_M
        far = MAX_RANGE_M - RANGE_MARGIN_M
        # Target just inside the nearer edge of the valid interval to avoid
        # creeping indefinitely toward a boundary or flying closer than needed.
        error = distance - np.clip(distance, near + .05, far - .05)
        cx, cy = info['cx'], info['cy']
        horizontal = unit(relative[:2])
        speed = np.clip(0.85 * error, -0.18, 0.28)
        if not centered(cx, cy):
            speed = min(speed, 0.)
        # Keep the fixed camera aimed vertically as the viewing distance changes.
        climb = np.clip((.5 - cy) * 1.05, -.19, .19)
        target[:2] = p[:2] + horizontal * speed * dt
        target[2] = p[2] + climb * dt
        # Leave room on both sides for range estimation error and stopping drift.
        ready = near <= distance <= far and centered(cx, cy)
        if ready:
            target[:3] = p
        self.valid_time = self.valid_time + dt if ready else 0.
        stop = self.valid_time >= 1.5 and np.linalg.norm(v) < .12
        info.update(phase='terminal_hold' if ready else 'terminal_approach',
                    centered=centered(cx, cy), in_range=in_band(distance),
                    valid_time_s=self.valid_time)
        return target, bool(stop), info
