"""Episode-local RGB-D witnesses for cells blocking an inspection waypoint."""
from collections import deque
from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from metric_flight.depth_guidance import camera
from metric_flight.view_control import PITCH_LIMITS
from metric_flight.short_view import history_corners, visible_corners
from sim_bridge import CONTROL_DT, controller, guided_curve


@dataclass
class RayFrame:
    frame: int
    center: np.ndarray
    axes: np.ndarray
    intrinsic: np.ndarray
    pitch: float
    rgb: np.ndarray
    near: np.ndarray
    far: np.ndarray


class RayHistory:
    # Use full-resolution depth extrema in each tile, including missing samples.
    tile = 4
    corners = np.array(list(product((-1., 1.), repeat=3)))

    def __init__(self):
        self.frames = deque(maxlen=192)
        self.serial = 0

    def add(self, rgb, depth, rig):
        center, axes, intrinsic = camera(rig)
        if self.frames:
            last = self.frames[-1]
            if (np.linalg.norm(center-last.center) < .015 and
                    np.trace(last.axes.T @ axes) > 1+2*np.cos(np.deg2rad(2))):
                return
        z = np.where(np.isfinite(depth) & (depth > .08), depth, 0).astype(np.float32)
        tile = self.tile
        h, w = z.shape
        h -= h % tile
        w -= w % tile
        blocks = z[:h, :w].reshape(h//tile, tile, w//tile, tile)
        near = blocks.min(axis=(1, 3))
        far = blocks.max(axis=(1, 3))
        image = cv2.resize(np.rint(255*np.clip(rgb[:h, :w], 0, 1)).astype(np.uint8),
                           (w//tile, h//tile), interpolation=cv2.INTER_AREA)
        self.serial += 1
        # axes already includes the measured pitch and body attitude.
        self.frames.append(RayFrame(self.serial, center.copy(), axes.copy(), intrinsic.copy(),
                                    float(rig.pitch_deg), image, near, far))

    def witnesses(self, world, cell):
        votes = []
        for frame in reversed(self.frames):
            delta = world-frame.center
            z = (delta @ frame.axes)[2]
            if z <= .08:
                continue
            # Bound the entire cell, 1 cm position error and 0.75 deg attitude error.
            half = cell/2 + .01 + np.linalg.norm(delta)*np.tan(np.deg2rad(.75))
            optical = (delta+self.corners*half) @ frame.axes
            if optical[:, 2].min() <= .08:
                continue
            pixels = optical @ frame.intrinsic.T
            tiles = np.floor(pixels[:, :2]/pixels[:, 2:]/self.tile).astype(int)
            l, t = tiles.min(axis=0)
            r, b = tiles.max(axis=0)
            if min(l, t) < 0 or r >= frame.near.shape[1] or b >= frame.near.shape[0]:
                continue
            roi = np.s_[t:b+1, l:r+1]
            near, far = frame.near[roi].min(), frame.far[roi].max()
            if near <= .10:
                continue
            margin = .04 + .025*z
            # A reliable first surface in the tested volume vetoes historical release.
            if far-near <= margin and near <= optical[:, 2].max()+margin and far >= optical[:, 2].min()-margin:
                return []
            if near <= optical[:, 2].max()+margin:
                continue
            # At a strong colour boundary require tighter depth agreement.
            colour_range = np.ptp(frame.rgb[roi].reshape(-1, 3), axis=0).max()
            if far-near > (.08 if colour_range > 55 else .16) + .04*z:
                continue
            if not votes or np.linalg.norm(frame.center-votes[0].center) >= .025 or (
                    frame.axes[:, 2] @ votes[0].axes[:, 2] < np.cos(np.deg2rad(6))):
                if len(votes) < 2:
                    votes.append(frame)
        return [f.frame for f in votes] if len(votes) == 2 else []


class BlockedViews:
    def __init__(self):
        self.history = RayHistory()
        self.examined = {}
        self.events = []
        self.releases = []
        self.looked = set()
        self.attempts = {}
        self.prediction = {}
        self.tried_corners = {}
        self.probes = []
        self.visited = []
        self.steps_without_progress = 0
        self.skips = []

    def track_probe_cells(self,guard,time):
        for event in self.probes:
            if 'goal' not in event:
                continue
            cells=np.asarray(event['cells'])
            evidence=guard.evidence[tuple(cells.T)]
            first=event.setdefault('first_free',{})
            for i in np.flatnonzero(evidence<0):
                first.setdefault(int(i),float(time))

    def clear(self, guard, ids):
        cleared = 0
        for raw in ids:
            key = tuple(map(int, raw))
            if guard.evidence[key] != 0 or self.examined.get(key) == self.history.serial:
                continue
            self.examined[key] = self.history.serial
            world = guard.origin+(np.asarray(key)+.5)*guard.cell
            witnesses = self.history.witnesses(world, guard.cell)
            if witnesses:
                guard.evidence[key] = -3
                self.releases.append(dict(cell=list(key), world=world.tolist(), witnesses=witnesses))
                cleared += 1
        return cleared

    def target(self, groups, guard, rig, state):
        center, axes, k = camera(rig)
        body = np.asarray(state['position'])
        pos_key = tuple(np.rint(body/.10).astype(int))
        ids = np.unique(np.vstack(list(groups.values())), axis=0)
        ids = [idx for idx in ids if guard.evidence[tuple(idx)] == 0 and
               self.attempts.get((pos_key,tuple(idx)),0) < 2]
        if not ids:
            return None
        ids = np.asarray(ids)
        world = guard.origin+(ids+.5)*guard.cell
        corners = world[:,None,:]+RayHistory.corners*(guard.cell/2)
        seen = history_corners(self.history.frames,corners)
        for row,idx in enumerate(ids):
            seen[row] |= self.tried_corners.get((pos_key,tuple(idx)),False)
        yaw = state['euler_deg'][2]
        pitches = np.unique(np.r_[np.linspace(*PITCH_LIMITS,7),rig.pitch_deg])
        turns = [0., *[sign*step for step in range(15,180,15) for sign in (1,-1)], 180.]
        best = None
        for turn in turns:
            spin = Rotation.from_euler('z',turn,degrees=True).as_matrix()
            # The optical centre follows the measured mount around the body pivot.
            new_center = body+spin@(center-body)
            for pitch in pitches:
                view_key = (pos_key, round(((yaw+turn)%360)/15)%24, round(pitch/10))
                if view_key in self.looked:
                    continue
                tilt = Rotation.from_euler('x',pitch-rig.pitch_deg,degrees=True).as_matrix()
                visible,_,_ = visible_corners(corners,new_center,spin@axes@tilt,k,rig.width,rig.height)
                counts = visible.sum(axis=1)
                novel = int((visible & ~seen).sum())
                full = int((counts==8).sum())
                if not novel and not full:
                    continue
                score = (full, novel, int(counts.sum()), -abs(turn), -abs(pitch-rig.pitch_deg))
                if best is None or score>best[0]:
                    best = (score,turn,float(pitch),view_key,visible)
        if best is None:
            return None
        score,turn,pitch,view_key,visible = best
        counts=visible.sum(axis=1)
        for row,idx in enumerate(ids):
            key=(pos_key,tuple(idx))
            self.tried_corners[key]=visible[row] | self.tried_corners.get(key,False)
        self.looked.add(view_key)
        for idx in ids[counts>0]:
            key = (pos_key,tuple(idx))
            self.attempts[key] = self.attempts.get(key,0)+1
        self.prediction = dict(full_cells=score[0], partial_or_full_cells=int((counts>0).sum()),
                               visible_corners=score[2], new_corners=score[1], considered_cells=len(ids))
        return world[np.argmax(counts)], pitch, turn

    def scan(self, sim, guard, groups, advance):
        """advance runs the actual dynamics and records each frame and termination."""
        initial = sim.last_receipt['state']
        if not groups or np.linalg.norm(initial['velocity']) > .035:
            return None
        plan = self.target(groups, guard, sim.scene, initial)
        if plan is None:
            self.skips.append(dict(t=sim.last_receipt['sim_time'],reason='no_new_partial_or_available_full_view'))
            return None
        point, pitch, turn = plan
        p = np.asarray(initial['position'])
        yaw = float(initial['euler_deg'][2])
        original_pitch = float(sim.scene.pitch_deg)
        cells = np.unique(np.vstack(list(groups.values())), axis=0)
        event = dict(t=sim.last_receipt['sim_time'], world=point.tolist(), pitch=pitch,
                     requested_turn_deg=turn, blocking_cells=cells.tolist(), predicted_visibility=self.prediction,
                     phases=[], max_drift_m=0.)
        self.events.append(event)
        def phase(name, goal_yaw, goal_pitch):
            st = sim.last_receipt['state']
            delta = (goal_yaw-st['euler_deg'][2]+180)%360-180
            rotating = name in ('body_yaw','return') and abs(delta) > .5
            seconds = max(.35, abs(delta)/28., abs(goal_pitch-sim.scene.pitch_deg)/30.)+.20
            seconds = np.ceil(seconds/CONTROL_DT)*CONTROL_DT
            entry = dict(phase=name, yaw=goal_yaw, pitch=goal_pitch, duration_s=0.)
            event['phases'].append(entry)
            # Known positive cells veto even an in-place yaw; the simulator also
            # performs continuous swept collision checks on its actual trajectory.
            quat = Rotation.from_quat(np.asarray(st['quaternion_wxyz'])[[1,2,3,0]])
            for angle in np.linspace(0, delta, max(2, int(np.ceil(abs(delta)/3))+1)):
                rot = Rotation.from_euler('z', angle, degrees=True)*quat
                ids = guard.body(p, rot.as_matrix())
                if (ids is None and not rotating) or (ids is not None and (guard.evidence[ids] > 0).any()):
                    entry['blocked'] = 'outside_grid' if ids is None else 'surface'
                    return 'scan_veto'
            curve = guided_curve(st, np.r_[p,goal_yaw], seconds, True)
            for i in range(round(seconds/CONTROL_DT)):
                receipt = sim.last_receipt
                sim.scene.pitch_deg += float(np.clip(goal_pitch-sim.scene.pitch_deg,-30*CONTROL_DT,30*CONTROL_DT))
                action,_ = controller(sim,receipt,None,(i+.5)*CONTROL_DT,sample=curve)
                fail = advance(action, rotating)
                state = sim.last_receipt['state']
                entry['duration_s'] = (i+1)*CONTROL_DT
                entry['actual_yaw'] = float(state['euler_deg'][2])
                drift = np.linalg.norm(np.asarray(state['position'])-p)
                event['max_drift_m'] = max(float(drift), event['max_drift_m'])
                if fail:
                    return fail
                if drift > .04:
                    return 'scan_position_drift'
                rot = Rotation.from_quat(np.asarray(state['quaternion_wxyz'])[[1,2,3,0]])
                ids = guard.body(np.asarray(state['position']),rot.as_matrix())
                if (ids is None and not rotating) or (ids is not None and (guard.evidence[ids]>0).any()):
                    return 'scan_veto'
            return None

        fail = phase('gimbal', yaw, pitch)
        self.clear(guard, cells)
        if not fail and turn and np.any(guard.evidence[tuple(cells.T)] == 0):
            fail = phase('body_yaw', yaw+turn, pitch)
        if not fail:
            fail = phase('return', yaw, original_pitch)
        self.clear(guard, cells)
        event.update(end_t=sim.last_receipt['sim_time'],status=fail or 'completed',
                     remaining_unknown=int((guard.evidence[tuple(cells.T)]==0).sum()))
        # A veto before moving is a refused observation, not a physical collision.
        if fail == 'scan_veto' and not any(ph['duration_s'] for ph in event['phases']):
            return None
        return fail
