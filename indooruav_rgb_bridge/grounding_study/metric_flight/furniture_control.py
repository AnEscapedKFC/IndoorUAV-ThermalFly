"""RGB-only visual guidance for approaching furniture and sweeping the visible sofa front."""
import numpy as np
from metric_flight.control import angle, unit
from metric_flight.object_goal import centered


class FurnitureInspection:
    def __init__(self, noun, instruction, use_depth=False):
        self.anchor=None
        if use_depth:
            from metric_flight.depth_guidance import ObservationAnchor
            self.anchor=ObservationAnchor(noun)
        self.width_m = {'laptop': .35, 'sofa': 1.8}[noun]
        self.scan = 'left to right' in instruction.lower()
        self.lower = 'lower' in instruction.lower()
        self.origin_z = None
        self.estimate = None
        self.lost = 0
        self.streak = 0
        self.last_good_view=None
        self.scan_center = None
        self.phase = 'approach'

    def step(self, p, yaw, v, raw, det, rig, dt, image, depth=None):
        if self.origin_z is None:
            self.origin_z = p[2]
        boxes = det['boxes']
        self.lost = 0 if boxes else self.lost + 1
        if self.scan_center is not None:
            # Execute a relative pan from a stationary viewpoint. The offline
            # evaluator independently verifies which sofa surfaces were observed.
            offset = {'scan_left': 1., 'scan_right': -1., 'scan_recenter': 0.}[self.phase]
            error = angle(self.scan_center + self.span * offset - yaw)
            phase = self.phase
            stop = False
            if abs(error) < 3.5:
                if phase == 'scan_recenter':
                    stop = bool(boxes and np.linalg.norm(v) < .12)
                    if stop and self.anchor is not None:
                        measured=self.anchor.update(image,depth,boxes[0]['xyxy'],rig)
                        distance=np.linalg.norm(self.anchor.position-p)
                        stop=bool(measured and measured['confirmed'] and
                                  .4+measured['error_m']<=distance<=.8-measured['error_m'])
                else:
                    self.phase = 'scan_right' if phase == 'scan_left' else 'scan_recenter'
            return np.r_[self.scan_position, yaw + np.clip(error, -7., 7.)], stop, dict(phase=phase)
        if not boxes:
            self.streak = 0
            return np.r_[p, yaw], False, dict(phase='observation_lost')
        x0, y0, x1, y1 = boxes[0]['xyxy']
        cx, cy = (x0+x1)/2, (y0+y1)/2
        observation=None
        if self.anchor is not None:
            observation=self.anchor.update(image,depth,boxes[0]['xyxy'],rig)
            if observation is None or not observation['confirmed']:
                self.streak=0
                self.lost=self.anchor.missed
                if self.last_good_view is not None:
                    prev,previous_yaw=self.last_good_view
                    delta=prev-p
                    target=p+delta*min(1.,.10/max(1e-6,np.linalg.norm(delta)))
                    return np.r_[target,yaw+np.clip(angle(previous_yaw-yaw),-8.,8.)],False,dict(
                        phase='reobserve',anchor=observation)
                return np.r_[p,yaw],False,dict(phase='observation_lost',anchor=observation)
            if centered(*observation['uv']):
                self.last_good_view=(p.copy(),yaw)
            self.estimate=np.asarray(observation['world'])
            cx,cy=observation['uv']
            ray=unit(self.estimate-p)
            cropped=observation['cropped']
        else:
            k = rig.ks['rgb']
            opt = np.linalg.solve(k, [cx*rig.width, cy*rig.height, 1.])
            ray = unit(rig.R @ rig.W[:3,:3].T @ opt)
            c = rig.s * (rig.R @ (np.linalg.inv(rig.W)[:3,3] - rig.origin))
            estimate_m = self.width_m * k[0,0] * np.linalg.norm(opt) / max((x1-x0)*rig.width, 1.)
            point = c + ray * np.clip(estimate_m, .25, 7.)
            cropped = x0 < .025 or x1 > .975
            if self.estimate is None:
                self.estimate = point
            elif not cropped:
                self.estimate += .35 * (point-self.estimate)
        rel = self.estimate - p
        distance = float(np.linalg.norm(rel))
        aim_yaw = np.rad2deg(np.arctan2(ray[1], ray[0]))
        if cropped:
            aim_yaw = np.rad2deg(np.arctan2(rel[1], rel[0]))
        aim = yaw + np.clip(angle(aim_yaw-yaw), -12., 12.)
        direction = unit(rel[:2])
        # The frozen VLA proposes forward travel; constrain it by visual range.
        speed_limit = np.clip((raw[:2]-p[:2]) @ direction/dt, .16, .32)
        speed = np.clip(.85*(distance-.64), -.12, speed_limit)
        if abs(angle(aim_yaw-yaw)) > 18.:
            speed = np.clip(speed, -.05, .05)
        rise = np.clip((.50-cy)*1.1, -.20, .16)
        if self.lower and p[2] > self.origin_z-.12:
            rise = min(rise, -.09)
        target = p + dt*np.r_[direction*speed, rise]
        ready = (.48 <= distance <= .78 and centered(cx,cy)
                 and (not self.lower or p[2] <= self.origin_z-.10))
        if observation is not None:
            error=observation['error_m']
            ready=(observation['views']>=2 and distance-error>=.4 and distance+error<=.8
                   and centered(cx,cy) and (not self.lower or p[2]<=self.origin_z-.10))
        self.streak = self.streak+1 if ready else 0
        if ready:
            target = p
        settled = self.streak >= 3 and np.linalg.norm(v) < .12
        if self.scan and settled:
            self.scan_center = aim_yaw
            self.scan_position = p.copy()
            self.span = float(np.clip(np.rad2deg(np.arctan2(self.width_m/2, distance)), 28., 55.))
            self.phase = 'scan_left'
        return np.r_[target, aim], bool(settled and not self.scan), dict(
            phase='terminal_hold' if ready else 'approach', range_est_m=distance,
            anchor=observation,cx=cx, cy=cy, clipped=bool(cropped), centered=centered(cx,cy))
