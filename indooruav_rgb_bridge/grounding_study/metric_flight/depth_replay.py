"""Offline: ask the online depth guard about the *same* RGB-control waypoints."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parents[1])]
from metric_flight.depth_guard import DepthGuard
from metric_flight.furniture import deny_expert_reads
from sim_bridge import SIM,RoomSpace
from cameras import fpv_camera
from pointcloud import DEFAULT_DRONE_SIZE
from scenes.room65_selected_anchored.room import NativeModel,intrinsic


def read(p):return json.loads(p.read_text())


def replay(name,base,net,rig,half,known):
    folder=base/'rgb'/name
    e=read(folder/'episode.json'); frames=read(folder/'frames.json'); ds=read(folder/'decisions.json')
    poses=[json.loads(row) for row in (folder/'video_frames.jsonl').read_text().splitlines()]
    guard=DepthGuard(e['spawn'],half)
    guard.seed_body(np.array(e['spawn']),Rotation.from_euler('z',e['spawn_yaw'],degrees=True).as_matrix())
    def observe(w2c):
        rig.W=np.array(w2c)
        depth=net.sample(rig.W,rig.ks['rgb'],640,480)[1]*rig.s
        guard.observe(depth,rig)
    survey=read(base/'depth'/name/'depth_preflight.json')
    for row in survey['samples']:observe(row['w2c'])
    state=np.zeros(13);state[6]=1.
    mount=(fpv_camera(state).numpy()[:3,:3]@Rotation.from_euler('x',e['fixed_pitch_deg'],degrees=True).as_matrix()
           @np.diag([1.,-1.,-1.]))
    body=Rotation.from_matrix(rig.R@np.asarray([r['rgb_w2c'] for r in poses])[:,:3,:3].transpose(0,2,1)@mount.T)
    decisions={round(d['t']*20):d for d in ds}
    rows=[]
    for i,f in enumerate(frames):
        if i%4==0:observe(poses[i]['rgb_w2c'])
        if i not in decisions:continue
        d=decisions[i]
        omega=np.zeros(3) if i==0 else (body[i-1].inv()*body[i]).as_rotvec()/.05
        st=dict(position=f['xyz'],euler_deg=body[i].as_euler('xyz',degrees=True),velocity=f['v'],
                quaternion_wxyz=body[i].as_quat()[[3,0,1,2]],angular_velocity=omega)
        g=np.array(d['world_target']);duration=.5
        if d['stopping']:g=np.r_[f['xyz'],f['yaw']];duration=2.
        verdict=guard.check(st,g,duration,np.array(d['reference_velocity_in']),np.array(d['reference_velocity_out']))
        ok=known.contains_box(g[:3],Rotation.from_euler('z',g[3],degrees=True).as_matrix(),half)
        rows.append(dict(n=d['n'],t=d['t'],guard=verdict,level_body_at_proposed_waypoint_known=bool(ok)))
    return rows


def main():
    deny_expert_reads()
    base=HERE/'results/furniture_640';inputs=SIM/'scenes/room65_selected_anchored/inputs'
    cfg=read(inputs/'frame.json')
    row=next(v for v in read(inputs/'manifest.json')['views'] if v['sequence']==62 and v['modality']=='rgb')
    rig=SimpleNamespace(R=np.array(cfg['rotation']),s=cfg['meters_per_source_unit'],origin=np.array(cfg['origin']),
                        ks={'rgb':intrinsic(row,640,480)})
    net=NativeModel('rgb',640,480);known=RoomSpace()
    results={n:replay(n,base,net,rig,np.array(DEFAULT_DRONE_SIZE)/2+.05,known)
             for n in ('scan_1','sofa_1','sofa_2')}
    for name,rows in results.items():print(name,rows[-1],flush=True)
    (base/'baseline_waypoint_replay.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
