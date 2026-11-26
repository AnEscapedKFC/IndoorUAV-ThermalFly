"""Postprocess real 3DGS-flight traces, without starting a simulator or model."""
from pathlib import Path
import json
import math
import sys

import numpy as np
from scipy.spatial.transform import Rotation

BRIDGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRIDGE))
from sim_bridge import CorridorSpace, RoomSpace
from sim_coordinates import signed_degrees
from pointcloud import DEFAULT_DRONE_SIZE

DATA = BRIDGE.parent / 'gs_gradnav_simulator' / 'outputs' / 'expert_rgbt_20261123' / 'episodes'
HALF = np.asarray(DEFAULT_DRONE_SIZE) / 2 + .05


def read_rows(path):
    return [json.loads(line) for line in path.open()]


def pose(frame):
    orientation = Rotation.from_quat(np.asarray(frame['q_w_from_b_wxyz'])[[1, 2, 3, 0]])
    return np.r_[frame['position_w_m'], orientation.as_euler('xyz', degrees=True)[2]]


def flu(p, q):
    yaw = np.deg2rad(p[3])
    delta = q[:3] - p[:3]
    return [float(delta[0]*np.cos(yaw) + delta[1]*np.sin(yaw)),
            float(-delta[0]*np.sin(yaw) + delta[1]*np.cos(yaw)), float(delta[2]),
            float(signed_degrees(q[3]-p[3]))]


def known_level(grid, target):
    return bool(grid.contains_box(target[:3], Rotation.from_euler('z',target[3],degrees=True).as_matrix(), HALF))


def rate_counts(frames):
    applied = np.asarray([f['applied_control'] for f in frames])
    return {'ticks': len(applied),
            'rates_at_controller_limit': int((np.abs(applied[:,:3]) >= .9-1e-7).any(axis=1).sum()),
            'thrust_at_controller_limit': int(((applied[:,3] <= 1e-7) | (applied[:,3] >= 1-1e-7)).sum())}


def analyze(path, grid, expert, meta):
    flight = read_rows(path/'frames.jsonl')
    decisions = read_rows(path/'decisions.jsonl')
    terminal_goal = np.asarray(meta['task_spec']['goal_w_m'])
    last = flight[-1]
    last_pose = pose(last)
    angles = Rotation.from_quat(np.asarray(last['q_w_from_b_wxyz'])[[1,2,3,0]]).as_euler('xyz',degrees=True)
    last_cell = np.floor((last_pose[:3]-grid.lo)/grid.step).astype(int)
    inside = (last_cell >= 0).all() and (last_cell < grid.free.shape).all()
    output = {'outcome': json.loads((path/'summary.json').read_text())['outcome'],
              'whole_flight_controls': rate_counts(flight[1:]),
              'end_pose_world_m_deg': last_pose.tolist(), 'end_roll_pitch_deg': angles[:2].tolist(),
              'end_spatial_cell_known': bool(inside and grid.free[tuple(last_cell)]),
              'end_actual_box_known': last['in_known_space'], 'end_if_body_level_at_same_pose_known': known_level(grid,last_pose),
              'end_obstacle_clearance_m': last['clearance_m'], 'initial_gimbal_pitch_deg': flight[0]['pitch_deg'],
              'expert_gimbal_pitch_deg_range': [min(f['rig_pitch_deg'] for f in expert), max(f['rig_pitch_deg'] for f in expert)],
              'decisions': []}
    prior_raw = pose(flight[0])
    for i, d in enumerate(decisions):
        fid = d['frame_id']
        end_fid = decisions[i+1]['frame_id'] if i+1 < len(decisions) else last['frame_id']
        start, end = pose(flight[fid]), pose(flight[end_fid])
        raw = np.array(d.get('raw_target_pose_w_m_deg',d['target_pose_w_m_deg']))
        commanded = np.array(d['target_pose_w_m_deg'])
        is_hold = bool((d.get('stop_heuristic') or d.get('oracle_stop')) and not d.get('advanced_stage'))
        used = start if is_hold else commanded
        raw_duration = 1. if not is_hold else 2.
        if path.parents[0].name == 'oracle' and not is_hold:
            raw_duration = min(raw_duration,(meta['waypoint_spec']['stop_frame_id']-fid)/20)
        dt = (end_fid-fid)/20
        new = {'frame': fid, 't_s': fid/20, 'actual_segment_s': dt, 'holding': is_hold,
                'start_w_m_deg': start.tolist(), 'raw_target_w_m_deg': raw.tolist(),
                'raw_delta_flu_m_deg': flu(start,raw), 'sent_delta_flu_m_deg': flu(start,used),
                'actual_delta_flu_m_deg': flu(start,end),
                'expert_clock_1s_delta_flu_m_deg': flu(pose(expert[fid]),pose(expert[min(fid+20,len(expert)-1)])),
                'raw_target_known_if_level': known_level(grid,raw), 'sent_target_known_if_level': known_level(grid,used),
                'actual_segment_ended_inside': flight[end_fid]['in_known_space'],
                'raw_pose_step_m_from_previous_prediction': float(np.linalg.norm(raw[:3]-prior_raw[:3])),
                'raw_pose_step_yaw_from_previous_prediction_deg': abs(signed_degrees(raw[3]-prior_raw[3])),
                'full_segment_endpoint_tracking_error_m': float(np.linalg.norm(end[:3]-used[:3])) if math.isclose(dt,raw_duration) else None,
                'goal_error_at_start_m': float(np.linalg.norm(start[:3]-terminal_goal)),
                'goal_error_at_end_m': float(np.linalg.norm(end[:3]-terminal_goal)),
                'start_velocity_mps': flight[fid]['velocity_w_mps'], 'end_velocity_mps': flight[end_fid]['velocity_w_mps'],
                **rate_counts(flight[fid+1:end_fid+1])}
        if d['native_chunk']:
            c = np.asarray(d['native_chunk'])
            new['native_final_0p2s_mean_speed_mps'] = float(np.linalg.norm(c[-1,:3]-c[-3,:3])/.2)
            new['tracker_terminal_desired_speed_mps'] = 0. if is_hold else float(np.linalg.norm(used[:3]-start[:3])/raw_duration)
        output['decisions'].append(new)
        prior_raw = raw
    return output


def main():
    out = {}
    for level in ['model','bounded','oracle']:
        out[level] = {}
        for e in sorted((BRIDGE/'runs'/level).iterdir()):
            meta = json.loads((DATA/e.name/'episode.json').read_text())
            ex = read_rows(DATA/e.name/'frames.jsonl')
            grid = CorridorSpace() if e.name.startswith('corridor') else RoomSpace()
            out[level][e.name] = analyze(e,grid,ex,meta)
    results = {'method': 'Outputs compared in actual starting FLU (forward,left,up,yaw deg). Expert comparison uses expert own heading and the same simulation time; only first decision is identical in observation/state. Level tests query the unchanged known-space grid with the original safety half-box and level yaw; they are feasibility diagnostics, not flown policy inputs. Saturation threshold +/-0.9 rad/s is the controller command cap. Native terminal speed is a two-interval finite difference, not an independently predicted velocity. Time-truncated segments have no endpoint tracking error.',
               'conditions':out}
    (Path(__file__).parent/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
