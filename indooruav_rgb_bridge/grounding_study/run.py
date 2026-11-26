"""Run paired direct and physical episodes on the existing simulator through its public flight interface."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from sim_bridge import (SIM, RoomRig, CorridorRig, RoomSpace, CorridorSpace, PolicyFlightSimulator, rgb_factory,
                        projection, link_targets, guided_curve, controller, CONTROL_DT, small_update)
from sim_coordinates import EpisodeCoordinates
from bridge_client import exchange
from cameras import fpv_camera
from policy import Policy
DATA=SIM/'outputs/expert_rgbt_20261123'
OUT=HERE/'results'
SOCKET=ROOT/'runtime/grounding-study.sock'
BASE=ROOT/'runs/gimbal/model_fixed'
OFFSETS={'room_inspect_ac':(.12,.08,12),'room_search_laptop_sofa':(-.12,.08,-12),
         'corridor_high_return':(.15,-.10,12),'corridor_climb_accelerate':(-.15,.10,-12)}
def read(p):return json.loads(p.read_text())
def line(p):
    with p.open() as f:return json.loads(next(f))
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2))
def infer(req, *, use_cache=True):
    # Reuse only byte-identical paired inputs; the actual RNG seed is included.
    key=[hashlib.sha256(Path(req[k]).read_bytes()).hexdigest() for k in ('rgb_path','ref_rgb_path')]
    digest=hashlib.sha256(json.dumps([*key,req['instruction'],req['state'],req['seed']]).encode()).hexdigest()
    cache=OUT/'model_cache';cache.mkdir(exist_ok=True)
    p=cache/(digest+'.json')
    if use_cache and p.exists():return np.array(read(p))
    a=exchange(SOCKET,req)
    if a.get('id')!=req['id'] or 'waypoints' not in a:raise RuntimeError(a)
    chunk=np.array(a['waypoints'])
    assert chunk.shape==(10,4) and np.isfinite(chunk).all()
    if use_cache:dump(p,chunk.tolist())
    return chunk
def world_ray(rig,b,yaw):
    x0,y0,x1,y1=b['xyxy']
    pixel=np.array([(x0+x1)*.5*rig.width,(y0+y1)*.5*rig.height,1])
    ray=rig.R @ rig.W[:3,:3].T @ np.linalg.solve(rig.ks['rgb'],pixel)
    return (yaw-np.rad2deg(np.arctan2(ray[1],ray[0]))+180)%360-180
def upright(p,yaw):
    q=Rotation.from_euler('z',yaw,degrees=True).as_quat()[[3,0,1,2]]
    state=np.zeros(13);state[:3]=p;state[6:10]=q
    return q,state
def run(name,new,arm,mode,detector,waypoint_check='auto'):
    case=('new_' if new else '')+name
    folder=OUT/'scoped_checks'/waypoint_check/mode/arm/case
    if (folder/'episode.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    ep=DATA/'episodes'/name;meta=read(ep/'episode.json');f0=line(ep/'frames.jsonl')
    source=meta['scene_version'];room=source.startswith('room')
    rig,known=(RoomRig,RoomSpace()) if room else (CorridorRig,CorridorSpace())
    xyz=np.array(f0['position_w_m'],dtype=float)
    yaw0=Rotation.from_quat(np.array(f0['q_w_from_b_wxyz'])[[1,2,3,0]]).as_euler('xyz',degrees=True)[2]
    if new:
        f,l,dy=OFFSETS[name];ang=np.deg2rad(yaw0)
        xyz[:2]+=np.array([np.cos(ang)*f-np.sin(ang)*l,np.sin(ang)*f+np.cos(ang)*l]);yaw0+=dy
    ins=meta['instruction'];map_=EpisodeCoordinates(xyz,yaw0,0.)
    policy=Policy(ins);raw_audit=Policy(ins);auditor=Policy(ins)
    from metric_flight.control import gate, quantities, Odometer, metric_waypoint
    odo=Odometer(xyz,yaw0);altitude,length=quantities(ins)
    use_guard=arm in ('guard','both') and gate(ins,waypoint_check);use_hint=arm in ('hint','both')
    scene=SIM/'scenes'/source
    # Landmarks enter telemetry only, never the policy.
    landmarks=link_targets(meta,DATA)
    with PolicyFlightSimulator(scene/'models/rgb.ply',point_cloud=scene/'collision/rgb_surface_m.ply',
            config={'limits':{'max_steps':1500}},seed=meta['seed'],spawn=xyz.tolist(),yaw=yaw0,resolution=.5,
            spawn_velocity_w_mps=f0['velocity_w_mps'],scene_factory=rgb_factory(rig,f0['rig_pitch_deg'])) as sim:
        r=sim.last_receipt;st=r['state']
        pos=xyz.copy();yaw=yaw0;vel=np.array(f0['velocity_w_mps']);q,body=upright(pos,yaw)
        image=sim.last_rgb;t=0.;reason=None
        if sim.obstacles.distance(pos,q)['safety_clearance']<=0:raise ValueError(('spawn clearance',case))
        frames=[];decisions=[];hold=[]
        def record():
            odo.tick(pos)
            rotation=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            inside=bool(known.contains_box(pos,rotation,sim.obstacles.safety_half_size))
            coll=r['collision']['event'] is not None if mode=='flight' else None
            frames.append(dict(t=t,xyz=pos.tolist(),yaw=yaw,v=vel.tolist(),inside=inside,collision=coll,
                objects={key:projection(sim,v) for key,v in landmarks.items()}))
            return inside
        assert record(),('outside at spawn',case)
        previous=map_.to_native(pos,yaw)
        for i in range(60):
            now=map_.to_native(pos,yaw)
            path=folder/f'{i:03}.png'
            im=np.rint(image*255).astype(np.uint8);Image.fromarray(im).save(path,compress_level=1)
            if not new and i==0:
                assert np.array_equal(im,np.asarray(Image.open(BASE/name/'rgb/000000.png')))
            text,found=policy.update(now,path,detector if use_hint else None,lambda b:world_ray(sim.scene,b,yaw))
            auditor.update(now,path,None);raw_audit.update(now,path,None)
            req=dict(rgb_path=str(path),ref_rgb_path=str(folder/'000.png'),instruction=text,state=now.tolist(),seed=42+i,id=f'{mode}_{arm}_{case}_{i}')
            chunk=infer(req)
            raw=chunk[9].copy();stop=bool(small_update(raw,previous));speed=float(np.linalg.norm(vel[:2]))
            _,_,flags=raw_audit.check(now,raw,stop,speed)
            chosen=raw.copy();stopping=stop
            if use_guard:
                corrected=metric_waypoint(ins,waypoint_check,map_.to_world(raw),pos,yaw,odo,1.)
                chosen=map_.to_native(corrected[:3],corrected[3])
                stopping=bool(stop and (altitude is None or abs(now[2]-altitude)<=.2))
                if length is not None:stopping=abs(odo.path-length)<=.4
            _,_,applied_flags=auditor.check(now,now if stopping else chosen,stopping,speed)
            target=map_.to_world(chosen) if not stopping else np.r_[pos,yaw]
            decisions.append(dict(n=i+1,t=t,raw=raw.tolist(),raw_stop=stop,chosen=chosen.tolist(),stopping=stopping,
                state=now.tolist(),waypoint_check_active=use_guard,raw_flags=flags,executed_flags=applied_flags,object=found,prompt=text,world_target=target.tolist()))
            previous=raw.copy()
            if mode=='direct':
                if stopping:reason='stop';break
                dest=target[:3];vel=dest-pos;pos=dest.copy();yaw=float(target[3]);q,body=upright(pos,yaw)
                image=sim.scene.render(fpv_camera(body))[0];t+=1;record()
            else:
                seconds=2. if stopping else 1.
                sample=guided_curve(st,target,seconds,stopping)
                if stopping:hold=[dict(xyz=pos.tolist(),speed=float(np.linalg.norm(vel)))]
                for k in range(round(seconds/CONTROL_DT)):
                    u,_=controller(sim,r,None,(k+.5)*CONTROL_DT,sample=sample)
                    r=sim.step(u,format='body_rates_thrust');st=r['state'];t=r['sim_time']
                    pos=np.asarray(st['position']);yaw=float(st['euler_deg'][2]);vel=np.asarray(st['velocity']);q=np.asarray(st['quaternion_wxyz'])
                    inside=record()
                    if stopping:hold.append(dict(xyz=pos.tolist(),speed=st['speed']))
                    if r['terminated'] or r['truncated'] or not inside:
                        reason=r['reason'] or 'outside_known_space';break
                    if not stopping and t+1.e-6>=60:reason='limit';break
                image=sim.last_rgb
                if stopping:reason=reason or 'stop'
                if reason:break
        reason=reason or 'limit'
        Image.fromarray(np.rint(image*255).astype(np.uint8)).save(folder/'end.png')
        dump(folder/'frames.json',frames);dump(folder/'decisions.json',decisions)
        dump(folder/'episode.json',dict(case=case,source=name,new=new,mode=mode,arm=arm,instruction=ins,spawn=xyz.tolist(),spawn_yaw=yaw0,
            reason=reason,hold=hold,num_decisions=len(decisions),end=pos.tolist(),end_yaw=yaw,duration=t))
        print(mode,arm,case,reason,len(decisions),flush=True)
if __name__=='__main__':
    if sys.argv[1:3] == ['flight','furniture']:
        from metric_flight.furniture import main
        main(sys.argv[3:]);sys.exit(0)
    if sys.argv[1:3] in (['flight','corridor'],['flight','room']):
        from metric_flight.experiment import main
        main(sys.argv[2:]);sys.exit(0)
    parser=argparse.ArgumentParser();parser.add_argument('--waypoint-check',choices=['auto','off'],default='auto');parser.add_argument('mode',choices=['direct','flight']);parser.add_argument('arms',nargs='+');parser.add_argument('--subset',choices=['all','old','new'],default='all');args=parser.parse_args()
    detector=None
    for new in [False,True]:
        if (new and args.subset=='old') or (not new and args.subset=='new'):continue
        for name in (OFFSETS if new else sorted(p.name for p in BASE.iterdir() if p.is_dir())):
            for arm in args.arms:
                if arm in ('both','hint') and name.startswith('room') and detector is None:
                    from detector import Detector
                    detector=Detector()
                run(name,new,arm,args.mode,detector,args.waypoint_check)
