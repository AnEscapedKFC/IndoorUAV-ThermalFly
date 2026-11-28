"""Episode-local RGB-D witnesses for cells blocking an inspection waypoint."""
from collections import deque
from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from metric_flight.depth_guidance import camera
from metric_flight.view_control import PITCH_LIMITS, elevation
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
        heading = np.degrees(np.arctan2(axes[1, 2], axes[0, 2]))
        hfov = np.degrees(np.arctan(rig.width/(2*k[0, 0])))
        for ids in groups.values():
            if not len(ids):
                continue
            ids = np.asarray(ids)
            ids = ids[guard.evidence[tuple(ids.T)]==0]
            if not len(ids):
                continue
            world = guard.origin+(ids+.5)*guard.cell
            # Split opposite blind spots before computing a mean sightline.
            rel = world-center
            bearings = np.degrees(np.arctan2(rel[:,1],rel[:,0]))
            elev = np.degrees(np.arctan2(rel[:,2],np.linalg.norm(rel[:,:2],axis=1)))
            sectors = np.column_stack((np.floor((bearings+22.5)/45)%8,np.floor((elev+15)/30)))
            _,group,counts=np.unique(sectors,axis=0,return_inverse=True,return_counts=True)
            for sector in np.argsort(-counts,kind='stable'):
                point=world[group==sector].mean(axis=0)
                direction=point-center
                bearing = np.degrees(np.arctan2(direction[1], direction[0]))
                turn = (bearing-heading+180) % 360-180
                pitch = float(np.clip(rig.pitch_deg+elevation(direction)-elevation(axes[:, 2]), *PITCH_LIMITS))
                # Pitch first. Rotate the body only if horizontal coverage needs it.
                yaw_delta = 0. if abs(turn) < .70*hfov else float(turn)
                pos = tuple(np.rint(np.asarray(state['position'])/.10).astype(int))
                key = (pos, round(bearing/18), round(elevation(direction)/15))
                if key not in self.looked:
                    self.looked.add(key)
                    return point, pitch, yaw_delta
        return None

    def scan(self, sim, guard, groups, advance):
        """advance runs the actual dynamics and records each frame and termination."""
        initial = sim.last_receipt['state']
        if not groups or np.linalg.norm(initial['velocity']) > .035:
            return None
        plan = self.target(groups, guard, sim.scene, initial)
        if plan is None:
            return None
        point, pitch, turn = plan
        p = np.asarray(initial['position'])
        yaw = float(initial['euler_deg'][2])
        original_pitch = float(sim.scene.pitch_deg)
        cells = np.unique(np.vstack(list(groups.values())), axis=0)
        event = dict(t=sim.last_receipt['sim_time'], world=point.tolist(), pitch=pitch,
                     requested_turn_deg=turn, blocking_cells=cells.tolist(), phases=[], max_drift_m=0.)
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
