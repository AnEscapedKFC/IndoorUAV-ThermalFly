import json
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from run import HERE,ROOT,BASE,DATA,OUT,read,line,dump,infer,RoomRig,rgb_factory,SIM,fpv_camera,world_ray
from detector import Detector
from policy import sector,SECTORS,BearingMemory
TEXT={'laptop':'Turn in place to face the open laptop.','sofa':'Turn in place to face the black sofa.','ac':'Turn in place to face the wall-mounted air conditioner.'}
def yaw_turn(req):
    chunk=infer(req)
    return float(np.rad2deg((chunk[9,3]-req['state'][3]+np.pi)%(2*np.pi)-np.pi))
def directional(s,side):
    if side in ('left','right'):return s.replace('Turn in place',f'Turn {side} in place')
    return s+(' The object is centered ahead.' if side=='center' else '')
def base_request(path):
    return dict(rgb_path=str(path),ref_rgb_path=str(BASE/'room_search_laptop_sofa/rgb/000000.png'),state=[0.,0.,1.5,0.])
def static(det):
    records=[]
    original=[r for r in map(json.loads,(ROOT/'diagnoses/image/raw.jsonl').read_text().splitlines()) if r['mode']=='fixed_reference' and r['goal'] in TEXT]
    tasks={(r['view'],r['goal']):(Path(r['request']['rgb_path']), ('left' if r['side']==-1 else 'right')) for r in original}
    for episode in ['corridor_high_return','corridor_climb_accelerate','room_search_laptop_sofa','room_obj_ac_east_to_west']:
        for obj in (['laptop'] if episode=='room_obj_ac_east_to_west' else TEXT):
            tasks[(episode,obj)]=(BASE/episode/'rgb/000000.png',None)
    for (name,obj),(file,truth) in tasks.items():
        detection=det(file,obj)
        for seed in (16,42,77):
            for arm,side in [('base',None),('gold',truth),('rgb',detection['side'])]:
                req=base_request(file);req.update(id=f'static_{name}_{obj}_{arm}_{seed}',seed=seed,instruction=directional(TEXT[obj],side))
                deg=yaw_turn(req)
                records.append(dict(view=name,obj=obj,truth=truth,seed=seed,arm=arm,hint=side,turn=deg))
                if truth and arm=='base':
                    prior=next(r for r in original if r['view']==name and r['goal']==obj and r['request']['seed']==seed)
                    assert abs(prior['yaw_right_deg']-deg)<1e-6
    dump(OUT/'static.json',records)
def angles(det):
    views={'laptop_left_sofa_right':('room_search_laptop_sofa',177,['laptop','sofa']),
           'sofa_left_laptop_right':('room_orbit_low',603,['laptop','sofa']),
           'ac_left':('room_search_full_survey',196,['ac']),'ac_right':('room_search_full_survey',360,['ac'])}
    scene='room65_selected_anchored';assets=DATA/'scenes'/scene
    lm=read(assets/'landmarks.json')|read(assets/'object_landmarks.json')
    rig=rgb_factory(RoomRig,0)(SIM/'scenes'/scene/'models/rgb.ply',320,180,'z',1.,device='cuda')
    results=[];pics=OUT/'azimuth_views';pics.mkdir(exist_ok=True)
    for name,(ep,idx,objs) in views.items():
        record=json.loads((DATA/'episodes'/ep/'frames.jsonl').read_text().splitlines()[idx])
        rig.pitch_deg=record['rig_pitch_deg'];pos=np.array(record['position_w_m'])
        att=Rotation.from_quat(np.array(record['q_w_from_b_wxyz'])[[1,2,3,0]])
        initial=att.as_euler('xyz',degrees=True)[2]
        mem={obj:BearingMemory() for obj in objs}
        for dy in (0,90,-90,180):
            rot=Rotation.from_euler('z',dy,degrees=True)*att
            body=np.zeros(13);body[:3]=pos;body[6:10]=rot.as_quat()[[3,0,1,2]]
            rgb=rig.render(fpv_camera(body))[0];file=pics/f'{name}_{dy}.png';img=np.rint(rgb*255).astype(np.uint8);Image.fromarray(img).save(file)
            yaw=initial+dy;state=np.array([0.,0.,pos[2],np.deg2rad((-dy)%360)])
            for obj in objs:
                d=det(file,obj);ray=world_ray(rig,d['boxes'][0],yaw) if d['boxes'] else None
                # Each independent probe can consult only the original view and its current rotated image.
                m=BearingMemory();m.last=mem[obj].last.copy()
                angle,source=m.estimate(obj,d,state,ray,0 if dy==0 else 1)
                if dy==0:mem[obj]=m
                vec=np.array(lm[obj]['center_w_m'])-pos
                truth=(yaw-np.rad2deg(np.arctan2(vec[1],vec[0]))+180)%360-180
                for arm,est in [('base',None),('gold',truth),('rgb_current',ray),('rgb_memory',angle)]:
                    for seed in (16,42,77):
                        prompt=TEXT[obj]
                        if est is not None:prompt+=f' The target is {SECTORS[sector(est)]} relative to your current heading.'
                        req=base_request(file);req.update(ref_rgb_path=str(pics/f'{name}_0.png'),id=f'angle_{name}_{obj}_{dy}_{arm}_{seed}',state=state.tolist(),seed=seed,instruction=prompt)
                        turn=yaw_turn(req)
                        after=abs((truth-turn+180)%360-180)
                        results.append(dict(view=name,obj=obj,dy=dy,truth=truth,sector=sector(truth),arm=arm,seed=seed,
                                            estimate=est,estimated_sector=None if est is None else sector(est),source=source if arm=='rgb_memory' else None,
                                            yaw_turn=turn,angle_reduction=abs(truth)-after))
    dump(OUT/'angles.json',results)
if __name__=='__main__':
    detector=Detector()
    static(detector);print('static complete',flush=True)
    angles(detector);print('azimuth complete',flush=True)
