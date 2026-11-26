"""Score physical histories only. Reference distances and target coordinates are evaluation-only."""
import json
from pathlib import Path
import math
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'results';STUDY=HERE.parent
DATA=STUDY.parent.parent/'gs_gradnav_simulator/outputs/expert_rgbt_20261123'
def load(p):return json.loads(p.read_text())
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2))
def max_span(mask,az):
    ids=np.flatnonzero(mask)
    segments=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
    return max([float(np.ptp(az[seg])) for seg in segments if len(seg)],default=0.)
def measure(folder,room):
    e=load(folder/'episode.json');f=load(folder/'frames.json');ds=load(folder/'decisions.json')
    xyz=np.array([x['xyz'] for x in f]);times=np.array([x['t'] for x in f]);col=any(x['collision'] for x in f)
    outside=any(not x['inside'] for x in f);p0=np.array(e['spawn']);yaw=math.radians(e['spawn_yaw'])
    path=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum());forward=float((xyz[-1]-p0)[:2]@[math.cos(yaw),math.sin(yaw)])
    result=dict(case=e['case'],source=e['source'],reason=e['reason'],duration_s=e['duration'],collision=col,outside=outside,decisions=len(ds),path_m=path,forward_m=forward)
    if not room:
        assert len(f)==len(set(x['t'] for x in f))
        assert abs(path-f[-1]['cumulative_m'])<1.e-7 and abs(forward-f[-1]['forward_m'])<1.e-7
        assert all(d['gate']==(e['switch']=='auto') for d in ds)
        distance=float(e['distance_setpoint'])
        relative_goal=p0+np.array([math.cos(yaw)*distance,math.sin(yaw)*distance,0.])
        result.update(straight_goal_xy_error_m=float(np.linalg.norm((xyz[-1]-relative_goal)[:2])),
                      no_collision=not col,strict=bool(not col and abs(path-distance)<=.4),forward_pass=bool(not col and abs(forward-distance)<=.4),forward_error_m=abs(forward-distance),
                      path_error_m=abs(path-distance),path_within_tolerance_collision_free=bool(not col and abs(path-distance)<=.4))
        return result
    n=e['source'];base=STUDY.parent/'runs/gimbal/model_fixed'/n;spec=load(DATA/'episodes'/n/'episode.json')['task_spec']
    center=np.array(load(DATA/'scenes/room65_selected_anchored/landmarks.json')['ac']['center_w_m'])
    dist=np.linalg.norm(xyz-center,axis=1);gap=float(np.linalg.norm(xyz[-1]-spec['goal_w_m']))
    threshold=load(base/'summary.json')['task_evidence']['ordered_inspections'][0]['reference_range_plus_20cm_m']
    uv=np.array([r['objects']['ac']['uv'] or [-1.e6,-1.e6] for r in f])/np.array([320,240])
    visible=np.array([r['objects']['ac']['in_image'] for r in f]);central=visible & np.all((uv>=.1)&(uv<=.9),axis=1)
    tight=visible & np.all((uv>=.3)&(uv<=.7),axis=1)
    close=central & (dist<=threshold);hit=np.flatnonzero(close);start=hit[0] if len(hit) else len(f)
    az=np.rad2deg(np.unwrap(np.arctan2(xyz[:,1]-center[1],xyz[:,0]-center[0])))
    after=np.arange(len(f))>=start
    arc=max_span(visible & after,az);central_arc=max_span(central & after,az)
    stopped=[d['t'] for d in ds if d['stopping']]
    prestop=bool(stopped and stopped[-1]>=1. and tight[(times<=stopped[-1]+1.e-7)&(times>=stopped[-1]-1.-1.e-7)].all())
    hold_centered=bool(stopped and tight[times>=stopped[-1]-1.e-7].all())
    h=e['hold'];stable=bool(e['reason']=='stop' and len(h)>=41 and max(math.dist(s['xyz'],h[0]['xyz']) for s in h)<=.12 and max(s['speed'] for s in h[-10:])<=.1)
    yaw_err=abs((e['end_yaw']-spec['goal_yaw_deg']+180)%360-180)
    has_arc='along an arc' in e['instruction'];climb='Climb slightly' in e['instruction']
    semantics=bool(hit.size and central[-1] and (not has_arc or central_arc>=20) and (not climb or xyz[:,2].max()-xyz[0,2]>=.05))
    strict=not col and not outside and stable and gap<=.4 and yaw_err<=8 and semantics
    result.update(strict=bool(strict),strict_with_prestop_center=bool(strict and prestop and hold_centered),
                  near=bool(hit.size),near_bound_m=threshold,min_distance_m=float(dist.min()),min_visible_distance_m=float(dist[visible].min()) if visible.any() else None,
                  first_close_s=float(times[start]) if start<len(f) else None,arc_task=has_arc,continuous_visible_arc_deg=arc,
                  continuous_centered_arc_deg=central_arc,visible_fraction=float(visible.mean()),last_second_centered=prestop,
                  hold_centered=hold_centered,terminal_center_80=bool(central[-1]),yaw_error_deg=yaw_err,terminal_error_m=gap,
                  position_pass=gap<=.4,stable=stable,climb_pass=(bool(xyz[:,2].max()-xyz[0,2]>=.05) if climb else None))
    return result
def main():
    metrics={}
    for scene in ('corridor','room'):
        metrics[scene]={}
        for arm in (('off','auto') if scene=='corridor' else ('hint','visual')):
            runs=[measure(p.parent,scene=='room') for p in sorted((OUT/scene/arm).glob('*/episode.json'))]
            if scene=='room' and arm=='hint':
                for n in ['room_inspect_ac','room_obj_ac_east_to_west','room_obj_ac_west_to_east','new_room_inspect_ac']:
                    if all(r['case']!=n for r in runs):
                        runs.append(measure(STUDY/'results/runs/flight/hint'/n,True))
            metrics[scene][arm]={r['case']:r for r in runs}
    write(OUT/'metrics.json',metrics)
if __name__=='__main__':
    main()
    from score_object import summarize
    summarize()
