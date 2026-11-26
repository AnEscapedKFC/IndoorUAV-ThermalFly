"""All task goals, landmarks and reference-derived ranges are read here when scoring already completed episodes."""
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent.parent/'gs_gradnav_simulator/outputs/expert_rgbt_20261123'
OUT=ROOT/'results'
def get(p):return json.loads(p.read_text())
def save(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False,allow_nan=False))
def wrapped(a):return (a+180)%360-180
def first(ds,k,exclude=()):
    return next(({'decision':d['n'],'flags':[w for w in d[k] if w not in exclude]} for d in ds if any(w not in exclude for w in d[k])),None)
def score(folder):
    e=get(folder/'episode.json');f=get(folder/'frames.json');ds=get(folder/'decisions.json');n=e['source'];direct=e['mode']=='direct';base=ROOT.parent/'runs/gimbal/model_fixed'/n
    meta=get(DATA/'episodes'/n/'episode.json');task=meta['task_spec'];t=np.array([r['t'] for r in f]);xyz=np.array([r['xyz'] for r in f]);speed=np.array([np.linalg.norm(r['v'][:2]) for r in f])
    target=np.array(json.loads((DATA/'episodes'/n/'frames.jsonl').read_text().splitlines()[-1])['position_w_m'])
    ne=float(np.linalg.norm(xyz[-1]-target));dnom=float(np.linalg.norm(xyz[-1]-task['goal_w_m']));ye=abs(wrapped(f[-1]['yaw']-task['goal_yaw_deg']))
    pose=dnom<=task['position_tol_m'] and ye<=task['yaw_tol_deg']
    outside=any(not r['inside'] for r in f);collision=None if direct else any(r['collision'] for r in f)
    h=e['hold'];stable=bool(e['reason']=='stop' and len(h)>=41 and max(math.dist(v['xyz'],h[0]['xyz']) for v in h)<=task['position_tol_m'] and max(v['speed'] for v in h[-10:])<=task['speed_tol_mps']) if not direct else None
    semantic={};obs=[];arc=None;room=n.startswith('room');cin={}
    thresholds=get(base/'summary.json')['task_evidence'].get('ordered_inspections',[])
    if room:
        landmarks_folder=DATA/'scenes'/meta['scene_version'];landmarks=get(landmarks_folder/'landmarks.json')|get(landmarks_folder/'object_landmarks.json')
        curr=0
        visible_cursor=0
        for s in thresholds:
            key=s['target'];center=np.array(landmarks[key]['center_w_m']);central=[]
            for row in f:
                v=row['objects'][key];x,y=(v['uv'] or [-1,-1]);central.append(v['in_image'] and 32<=x<=288 and 24<=y<=216)
            cin[key]=np.array(central)
            valid=cin[key] & (np.linalg.norm(xyz-center,axis=1)<=s['reference_range_plus_20cm_m'])
            found=np.where(valid & (np.arange(len(f))>=curr))[0];hit=int(found[0]) if len(found) else None
            vis=np.array([x['objects'][key]['in_image'] for x in f])
            indices=np.where(vis & (np.arange(len(f))>=visible_cursor))[0]
            ordered_visible=float(t[indices[0]]) if len(indices) else None
            visible_cursor=int(indices[0])+1 if len(indices) else len(f)
            obs.append(dict(target=key,first_ordered_visible_s=ordered_visible,center_ever=bool(np.any(cin[key])),visible_ever=any(x['objects'][key]['in_image'] for x in f),hit_at=None if hit is None else float(t[hit])))
            curr=hit+1 if hit is not None else len(f)
        semantic['ordered_inspections']=all(s['hit_at'] is not None for s in obs)
        if 'along an arc' in meta['instruction']:
            center=np.array(landmarks['ac']['center_w_m']);az=np.rad2deg(np.unwrap(np.arctan2(xyz[:,1]-center[1],xyz[:,0]-center[0])))
            keep=cin['ac'] & (t>=obs[0]['hit_at']) if obs[0]['hit_at'] is not None else np.zeros(len(t),dtype=bool)
            segments=np.split(np.flatnonzero(keep),np.flatnonzero(np.diff(np.flatnonzero(keep))>1)+1)
            arc=max((float(np.ptp(az[seg])) for seg in segments if len(seg)),default=0.)
            semantic['observed_arc']=arc>=20 and bool(cin['ac'][-1])
        elif 'centered' in meta['instruction']:semantic['finish_centered']=bool(cin['ac'][-1])
    z=xyz[:,2];height_mae=None;gain=float(z.max()-z[0]);peak=int(np.argmax(z));descent=float(z[peak]-z[peak:].min())
    hp=int(np.argmax(speed));speedup=float(speed[hp]-speed[0]);slow=float(speed[hp]-speed[hp:].min())
    if '1.82' in e['instruction']:
        err=np.abs(z-1.82)
        height_mae=float(np.trapz(err,t)/(t[-1]-t[0])) if t[-1]>t[0] else float(err[0])
        semantic['keep_altitude']=bool(err.max()<=.2)
    if 'accelerate' in e['instruction']:
        semantic['climb_then_descend']=gain>=.12 and descent>=.12
        semantic['accelerate_then_slow']=speedup>=.15 and slow>=.15
    if 'Climb slightly' in e['instruction']:semantic['slight_climb']=gain>=.05
    strict=bool(pose and stable and (not outside) and (not collision) and all(semantic.values())) if not direct else None
    safety=['horizontal_step','vertical_step','yaw_step']
    item=dict(case=e['case'],new=e['new'],arm=e['arm'],mode=e['mode'],strict=strict,pose=bool(pose),stable=stable,SR=ne<=2,NE=ne,reason=e['reason'],
              outside=outside,collision=collision,ordered=obs,arc_deg=arc,semantics=semantic,altitude_MAE=height_mae,gain=gain,descent=descent,
              speedup=speedup,slow=slow,steps=len(ds),first_raw=first(ds,'raw_flags'),first_executed=first(ds,'executed_flags'),
              first_raw_stage=first(ds,'raw_flags',safety),first_executed_stage=first(ds,'executed_flags',safety),
              hint_steps=sum(d['object'] is not None and d['object']['bearing_source'] is not None for d in ds),
              memory_steps=sum(d['object'] is not None and d['object']['bearing_source']=='memory' for d in ds))
    # Independent task audit. Use scene/goal geometry only after the episode has ended.
    import sys
    sys.path.insert(0,str(ROOT.parent))
    from sim_coordinates import EpisodeCoordinates
    coords=EpisodeCoordinates(e['spawn'],e['spawn_yaw'],0.)
    first_task={}
    for kind in ('raw','executed'):
        first_bad=None
        for d in ds:
            j=int(np.argmin(np.abs(t-d['t'])));p=xyz[j];yaw=f[j]['yaw']
            stop=d['raw_stop'] if kind=='raw' else d['stopping']
            pose_d=coords.to_world(d['raw']) if kind=='raw' else np.asarray(d['world_target'])
            flags=[]
            if room:
                stage=next((o for o in obs if o['hit_at'] is None or o['hit_at']>t[j]+1.e-7),obs[-1])
                obj=stage['target'];vec=np.array(landmarks[obj]['center_w_m'])-p
                if f[j]['objects'][obj]['in_image'] and not stop and (stage['hit_at'] is None or stage['hit_at']>t[j]+1.e-7):
                    before=abs(wrapped(np.rad2deg(np.arctan2(vec[1],vec[0]))-yaw))
                    pred_vec=np.array(landmarks[obj]['center_w_m'])-pose_d[:3]
                    after=abs(wrapped(np.rad2deg(np.arctan2(pred_vec[1],pred_vec[0]))-pose_d[3]))
                    if before>12 and after>before+5:flags.append('visible_target_further_off_axis')
                if len(obs)>1 and not any(x['objects']['laptop']['in_image'] for x in f[:j+1]) and np.linalg.norm(pose_d[:2]-p[:2])>.25 and not stop:
                    flags.append('translation_before_acquisition')
                if stop and (stage['hit_at'] is None or stage['hit_at']>t[j]+1.e-7 or ('observed_arc' in semantic and not semantic['observed_arc'])):flags.append('stop_before_inspection')
            else:
                if '1.82' in e['instruction'] and abs((p if stop else pose_d)[2]-1.82)>.2:flags.append('outside_altitude_band')
                if 'accelerate' in e['instruction']:
                    achieved_climb=z[:j+1].max()-z[0]>=.12
                    achieved_descent=z[:j+1].max()-z[j]>=.12
                    if not achieved_climb and pose_d[2]<p[2]-.04 and not stop:flags.append('descending_before_climb')
                    if stop and not (achieved_climb and achieved_descent):flags.append('stop_before_height_sequence')
                axis=target[:2]-xyz[0,:2];axis/=np.linalg.norm(axis)
                delta=pose_d[:2]-p[:2]
                if not stop and (delta@axis<-.1 or abs(axis[0]*delta[1]-axis[1]*delta[0])>.35):flags.append('off_corridor_direction')
            if flags:
                first_bad=dict(n=d['n'],flags=flags);break
        first_task[kind]=first_bad
    item['first_task_violation']=first_task
    item['cohort']='reverse' if 'reverse' in folder.parts else 'new' if e['new'] else 'old'
    vec=(np.array(landmarks[obs[0]['target']]['center_w_m']) if room else target)-xyz[0]
    item['initial_goal_azimuth_right_deg']=float(wrapped(e['spawn_yaw']-np.rad2deg(np.arctan2(vec[1],vec[0]))))
    return item
def summary(rows):
    n=len(rows);flight=rows[0]['mode']=='flight'
    return dict(n=n,strict=sum(r['strict'] for r in rows) if flight else None,SR=sum(r['SR'] for r in rows),NE=float(np.mean([r['NE'] for r in rows])),
                collision=sum(r['collision'] for r in rows) if flight else None,outside=sum(r['outside'] for r in rows),
                ordered_rooms=sum(all(x['hit_at'] is not None for x in r['ordered']) for r in rows if r['ordered']),
                rooms=sum(bool(r['ordered']) for r in rows))
def test_panels():
    static=get(OUT/'static.json');angles=get(OUT/'angles.json');stat={};az={}
    for arm in ('base','gold','rgb'):
        pos=[r for r in static if r['arm']==arm and r['truth']];neg=[r for r in static if r['arm']==arm and not r['truth']]
        signed=[r['turn']*(1 if r['truth']=='right' else -1) for r in pos]
        stat[arm]=dict(n=len(pos),right_sign=sum(v>0 for v in signed),beyond_2=sum(v>2 for v in signed),
                       absent_n=len(neg)//3,absent_direction_count=sum(r['seed']==42 and r['hint'] in ['left','right'] for r in neg),
                       absent_any_hint=sum(r['seed']==42 and r['hint'] is not None for r in neg),absent_turn_2=sum(abs(r['turn'])>2 for r in neg),absent_trials=len(neg),
                       visible_hint_correct=sum(r['seed']==42 and r['hint']==r['truth'] for r in pos),
                       visible_hint_center=sum(r['seed']==42 and r['hint']=='center' for r in pos),
                       visible_abstain=sum(r['seed']==42 and r['hint'] is None for r in pos))
    for arm in ('base','gold','rgb_current','rgb_memory'):
        az[arm]={}
        for group in ('all','front','rear'):
            rs=[r for r in angles if r['arm']==arm and (group=='all' or (abs(r['truth'])>90)==(group=='rear'))]
            cells=[r for r in rs if r['seed']==42];provided=[r for r in cells if r['estimate'] is not None]
            errors=[abs(wrapped(r['estimate']-r['truth'])) for r in provided]
            az[arm][group]=dict(cells=len(cells),provided=len(provided),hemisphere_correct=sum((abs(r['estimate'])>90)==(abs(r['truth'])>90) for r in provided),
                                 sector_correct=sum(r['sector']==r['estimated_sector'] for r in provided),median_abs_angle_error=float(np.median(errors)) if errors else None,
                                 better_2deg=sum(r['angle_reduction']>2 for r in rs),trials=len(rs),mean_angle_reduction=float(np.mean([r['angle_reduction'] for r in rs])))
    examples=[r for r in angles if r['arm']=='rgb_memory' and r['seed']==42]
    initial={(r['view'],r['obj']):r['estimate'] for r in examples if r['dy']==0}
    carry=[]
    for r in examples:
        est=wrapped(initial[r['view'],r['obj']]+r['dy'])
        carry.append(dict(view=r['view'],target=r['obj'],rotation=r['dy'],hemisphere_correct=(abs(est)>90)==(abs(r['truth'])>90),
                          sector_correct=int(np.floor((est+22.5)/45))%8==r['sector']))
    return dict(static=stat,azimuth=az,stationary_initial_bearing_extrapolation=carry)
if __name__=='__main__':
    items=[score(p.parent) for p in sorted([*(OUT/'runs').glob('*/*/*/episode.json'),*(OUT/'reverse/runs').glob('*/*/*/episode.json')])]
    assert len(items)==96,(len(items),'incomplete')
    groups={f'{m}/{g}/{a}':summary([r for r in items if r['mode']==m and r['arm']==a and (g=='all' or r['cohort']==g)]) for m in ('direct','flight') for g in ('old','new','reverse','all') for a in ('baseline','hint','guard','both')}
    save(OUT/'metrics.json',dict(groups=groups,episodes=items,panels=test_panels()))
