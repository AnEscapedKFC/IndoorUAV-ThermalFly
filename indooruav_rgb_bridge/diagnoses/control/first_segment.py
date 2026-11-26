"""Hold the first recorded native inference constant, vary only Hermite terminal velocity."""
import json
from pathlib import Path
from functools import partial
import sys

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation
import torch

from trace import BRIDGE, DATA, read_rows
sys.path.insert(0, str(BRIDGE))
from sim_bridge import PolicyFlightSimulator, RoomSpace, CorridorSpace, guided_curve, controller
from sim_coordinates import EpisodeCoordinates, signed_degrees


class PhysicsOnly:
    """The stored inference is the sole planner input; no new policy image is used."""
    def __init__(self, path, width, height, up_axis, scale, device, *, scene_dir):
        self.width = self.height = 1
        self.device = torch.device('cpu')
        cfg=json.loads((scene_dir/'inputs/frame.json').read_text())
        self.bounds = np.array(cfg['bounds_m'])
        self.K = torch.eye(3).unsqueeze(0)

    def render(self, _camera):
        return np.zeros((1,1,3)),np.ones((1,1)),np.ones((1,1))

    def metadata(self):
        return {'mode': 'fixed pre-recorded model chunk, CPU dynamics only'}


def sample_curve(st, goal, terminal):
    if terminal is None:
        return guided_curve(st, goal, 1., False)
    start = np.r_[st['position'], st['euler_deg'][2]]
    finish = goal.copy()
    finish[3] = start[3] + signed_degrees(finish[3]-start[3])
    initial_velocity = np.r_[st['velocity'], np.rad2deg(st['angular_velocity'][2])]
    curve = CubicHermiteSpline([0.,1.], [start,finish], [initial_velocity,terminal])
    def sample(unused, t):
        return curve(t),curve(t,1),curve(t,2),0
    return sample


def branch(episode, run, source, meta, fitted):
    frames = read_rows(run/'frames.jsonl')
    inference = read_rows(run/'decisions.jsonl')[0]
    start = frames[0]
    coord = EpisodeCoordinates(start['position_w_m'],start['yaw_deg'])
    poses = np.array([coord.to_world(n) for n in inference['native_chunk']])
    goal = poses[-1]
    terminal = (poses[-1]-poses[-3])/.2
    terminal[3] = signed_degrees(poses[-1,3]-poses[-3,3])/.2
    scene = BRIDGE.parent/'gs_gradnav_simulator'/'scenes'/meta['scene_version']
    grid = CorridorSpace() if episode.startswith('corridor') else RoomSpace()
    path = []; saturations = 0
    with PolicyFlightSimulator(scene/'models/rgb.ply',point_cloud=scene/'collision/rgb_surface_m.ply',
                              seed=meta['seed'],spawn=start['position_w_m'],yaw=start['yaw_deg'],
                              spawn_velocity_w_mps=start['velocity_w_mps'],scene_factory=partial(PhysicsOnly,scene_dir=scene),device='cpu') as sim:
        r = sim.last_receipt
        fn = sample_curve(r['state'],goal,terminal if fitted else None)
        for i in range(20):
            u,_ = controller(sim,r,None,(i+.5)/20,sample=fn)
            saturations += int(np.any(np.abs(u[:3])>=.9-1e-7))
            r = sim.step(u,format='body_rates_thrust')
            st = r['state']; xyz = np.array(st['position'])
            rot = Rotation.from_quat(np.asarray(st['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
            known = grid.contains_box(xyz,rot,sim.obstacles.safety_half_size)
            path.append({'frame':i+1,'w_m':xyz.tolist(),'velocity_w_mps':st['velocity'],'clearance_m':r['min_sampled_safety_clearance'],
                         'known':bool(known),'collision':r['collision']['event']})
            if not known or r['terminated'] or r['truncated']: break
    overlap=min(len(path),len(frames)-1)
    discrepancy=max(np.linalg.norm(np.asarray(path[i]['w_m'])-frames[i+1]['position_w_m']) for i in range(overlap))
    return {'duration_s':len(path)/20,'truncated_by_geometry':not path[-1]['known'] or bool(path[-1]['collision']),
            'max_position_difference_from_original_over_common_frames_m':float(discrepancy),
            'simultaneous_comparison_frames':overlap,'rate_saturation_ticks':saturations,
            'predicted_pose_w_m_deg':goal.tolist(),'requested_end_velocity_mps_degps':
            (terminal if fitted else np.r_[goal[:3]-start['position_w_m'],signed_degrees(goal[3]-start['yaw_deg'])]).tolist(),
            'actual_end_speed_mps':float(np.linalg.norm(path[-1]['velocity_w_mps'])),
            'tracking_error_m_at_1s':float(np.linalg.norm(goal[:3]-path[-1]['w_m'])) if len(path)==20 else None,
            'expert_position_error_m_at_same_physical_time':float(np.linalg.norm(np.asarray(path[-1]['w_m'])-source[len(path)]['position_w_m'])),
            'trajectory':path}


def main():
    out={'model_input':'Recorded first inference per episode, seed 42. Same selected endpoint, same initial state and controls. Both variants use original CPU engine, collision point cloud, known-space check and flight controller. Blank render feeds the observer but no image is used to generate a new decision. Held-out future poses are used solely for evaluation. Alternative terminal velocity uses slope of last 0.2 s in native trajectory; no new predictions, no full-episode performance claim.','episodes':{}}
    for run in sorted((BRIDGE/'runs/model').iterdir()):
        ep=run.name; meta=json.loads((DATA/ep/'episode.json').read_text()); source=read_rows(DATA/ep/'frames.jsonl')
        base=branch(ep,run,source,meta,False); alt=branch(ep,run,source,meta,True)
        assert base['max_position_difference_from_original_over_common_frames_m']<1e-6
        n=min(len(read_rows(run/'frames.jsonl'))-1,20)
        assert base['simultaneous_comparison_frames']==n and len(base['trajectory'])==n
        out['episodes'][ep]={'actual_bridge':base,'alternative_terminal_velocity':alt}
    (Path(__file__).parent/'first_segment.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
