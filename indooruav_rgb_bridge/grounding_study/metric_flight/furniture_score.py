"""Offline scoring of close furniture observations and ordered sofa mosaics."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parents[1])]
from metric_flight.furniture import RESULTS, MANIFEST, deny_expert_reads
from metric_flight.furniture_reference import PARTS, reference, SCENE
from metric_flight.object_goal import centered

read=lambda p:json.loads(p.read_text())


def dump(p,data):
    p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def center_of_detection(d):
    boxes=d['detection']['boxes']
    if not boxes:return False
    x,y,r,b=boxes[0]['xyxy']
    return centered((x+r)/2,(y+b)/2)


def seen_frame(rgb,depth,W,atlas):
    labels=atlas['labels']
    yy,xx=np.nonzero(labels)
    xyz=atlas['native'][yy,xx]@W[:3,:3].T+W[:3,3]
    h,w=depth.shape
    K=np.diag([w/320,h/240,1.])@atlas['k']
    uv=xyz@K.T
    uv=uv[:,:2]/np.maximum(uv[:,2:],1.e-8)
    ix,iy=np.rint(uv).astype(int).T
    valid=atlas['supported'][yy,xx]&(xyz[:,2]>0)&(ix>=2)&(ix<w-2)&(iy>=2)&(iy<h-2)
    ix,iy=ix.clip(0,w-1),iy.clip(0,h-1)
    z=xyz[:,2]*atlas['scale']
    residual=np.abs(z-depth[iy,ix])
    colors=rgb[iy,ix]
    photo_error=np.abs(colors.astype(float)-atlas['photo'][yy,xx]).mean(axis=1)
    valid &= (depth[iy,ix]>0)&(residual<=.08+.03*z)&(photo_error<=50)
    cost=(np.abs(uv[:,0]/w-.5)+np.abs(uv[:,1]/h-.5)+3*residual+photo_error/255)
    cost[~valid]=np.inf
    xmin,xmax=xx.min(),xx.max()
    focus=valid & (abs(uv[:,0]/w-.5)<=.10)&(abs(uv[:,1]/h-.5)<=.30)
    pos=float((np.median(xx[focus])-xmin)/(xmax-xmin)) if focus.sum()>=80 else None
    return dict(valid=valid,cost=cost,colors=colors,focus=pos)


def ordered(rows):
    steps=[]
    for row in rows:
        x=row['focus']
        if x is None:continue
        test=(x<=.32, .40<=x<=.60, x>=.68)
        if test[len(steps)]:
            steps.append(row['t'])
            if len(steps)==3:break
    return steps


def sweep_windows(component):
    """Find ordered, near-contiguous pans by the observed position on the sofa."""
    start=middle=right=None
    windows=[]
    for i,r in enumerate(component):
        x=r['focus']
        if x is None:continue
        if right is not None and x < component[right]['focus']-.15:
            windows.append(component[start:right+1])
            start=middle=right=None
        if x<=.32:
            if start is None or middle is not None or x<=component[start]['focus']:
                start=i
            middle=right=None
        elif start is not None and middle is None and .40<=x<=.60:
            middle=i
        elif middle is not None and x>=.68:
            if right is None or x>=component[right]['focus']:right=i
    if right is not None:windows.append(component[start:right+1])
    return windows


def summarize_history(folder,rows,atlas,name):
    yy,xx=np.nonzero(atlas['labels']); labels=atlas['labels'][yy,xx]
    components=[]; component=[]
    for row in rows:
        if not row['valid'].any():continue
        if component:
            prev=component[-1]
            overlap=np.count_nonzero(prev['valid']&row['valid'])/min(prev['valid'].sum(),row['valid'].sum())
            if row['t']-prev['t']>.151 or overlap<.12:
                components.append(component); component=[]
        component.append(row)
    if component:components.append(component)
    if name=='close_sweep_mosaic':
        components=[window for component in components for window in (sweep_windows(component) or [component])]
    candidates=[]
    for component in components or [[]]:
        best=np.full(len(xx),np.inf)
        rgb=np.full((len(xx),3),160,np.uint8)
        for row in component:
            take=row['cost']<best
            rgb[take]=row['colors'][take]
            best[take]=row['cost'][take]
        ok=np.isfinite(best)
        # Also verify the left, middle and right thirds of the complete foreground.
        parts={part:float(ok[labels==i].mean()) for i,part in enumerate(PARTS,1)}
        for side,lo,hi in [('left',0,.33),('middle',.33,.67),('right',.67,1.01)]:
            frac=(xx-xx.min())/(xx.max()-xx.min())
            parts[side]=float(ok[(frac>=lo)&(frac<hi)].mean())
        progress=ordered(component)
        all_seen=float(ok.mean())
        sufficient=all_seen>=.90 and min(parts.values())>=.70
        success=bool(sufficient and len(progress)==3)
        entry=dict(coverage=all_seen,per_part=parts,frames=len(component),
                   start_s=component[0]['t'] if component else None,end_s=component[-1]['t'] if component else None,
                   best_single_frame=max((float(r['valid'].mean()) for r in component),default=0.),
                   left_middle_right_times_s=progress,surface_pass=sufficient,ordered_full_front=success)
        candidates.append((entry,rgb))
    strongest,rgb=max(candidates,key=lambda v:(v[0]['ordered_full_front'],len(v[0]['left_middle_right_times_s'])==3,v[0]['coverage']))
    canvas=np.full_like(atlas['photo'],34);canvas[yy,xx]=rgb
    Image.fromarray(canvas).save(folder/(name+'.png'))
    # A fixed frame and the same pan in reverse must not satisfy the order by accident.
    if name=='close_sweep_mosaic' and strongest['frames'] and len(strongest['left_middle_right_times_s'])==3:
        run=[r for r in rows if strongest['start_s']<=r['t']<=strongest['end_s']]
        strongest['reverse_order_check']=len(ordered(run[::-1]))==3
        strongest['stationary_order_check']=len(ordered([run[len(run)//2]]*3))==3
    strongest['best_single_frame_any_segment']=max((float(r['valid'].mean()) for r in rows),default=0.)
    return strongest


def scan_score(folder,decisions,poses,frames,distance,atlas):
    rows=[]
    video=cv2.VideoCapture(str(folder/'rgb_fpv.mp4'))
    if not video.isOpened():raise ValueError('RGB stream could not be decoded')
    decision_at={round(d['t']*20):i for i,d in enumerate(decisions)}
    phase=None
    try:
        for idx,pose in enumerate(poses):
            ok,bgr=video.read()
            if not ok:raise ValueError(f'Missing RGB frame {idx}')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            h,w=rgb.shape[:2]
            W=np.array(pose['rgb_w2c'])
            assert abs(pose['t']-frames[idx]['t'])<1e-5
            # Offline visibility only: use the camera pose from this actual dynamics
            # receipt and the same room RGB 3DGS expected depth used in the simulator.
            K=np.diag([w/320,h/240,1.])@atlas['k']
            _,z,_=atlas['model'].sample(W,K,w,h)
            depth=z*atlas['scale']
            if idx in decision_at:
                i=decision_at[idx]
                phase=decisions[i]['info']['phase']
                tape=np.asarray(Image.open(folder/f'depth_{i:03d}.png'),dtype=float)*.001
                if not np.allclose(depth.clip(0,65),tape,atol=.002):
                    raise ValueError(f'Offline depth mismatch at physical frame {idx}')
            obs=seen_frame(rgb,depth,W,atlas)
            rows.append(dict(t=pose['t'],phase=phase,near=distance[idx]<=1.5,**obs))
        if video.read()[0]:raise ValueError('RGB stream extends beyond pose log')
    finally:
        video.release()
    # Ground the order in actual image evidence regardless of the online phase label.
    right=[i for i,r in enumerate(rows) if r['phase']=='scan_right']
    diag=summarize_history(folder,rows,atlas,'all_history_mosaic')
    formal=summarize_history(folder,[r for r in rows if r['near']],atlas,'close_sweep_mosaic')
    dump(folder/'coverage_frames.json',[dict(t=r['t'],phase=r['phase'],within_1p5m=bool(r['near']),
                coverage=float(r['valid'].mean()),focus_reference_x=r['focus']) for r in rows])
    return dict(sweep=formal,all_history_diagnostic=diag,scan_started=bool(right))


def score(folder,task,ref,atlas):
    e=read(folder/'episode.json'); f=read(folder/'frames.json'); ds=read(folder/'decisions.json')
    poses=[json.loads(s) for s in (folder/'video_frames.jsonl').read_text().splitlines()]
    rec=read(folder/'recording.json')
    assert len(poses)==len(f)==rec['frames'] and all(v['passed'] for v in rec['streams'].values())
    if e.get('render_wh')==[640,480]:
        for stream,wh in [('rgb_fpv',(640,480)),('thermal_fpv',(640,480)),('rgb_thermal_fpv',(1280,480))]:
            video=cv2.VideoCapture(str(folder/(stream+'.mp4')))
            try:
                assert video.isOpened()
                assert (round(video.get(cv2.CAP_PROP_FRAME_WIDTH)),round(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))==wh
                assert round(video.get(cv2.CAP_PROP_FRAME_COUNT))==len(f)
            finally:video.release()
    assert len(e['hold']) in range(42) and all(not row['objects'] for row in f)
    xyz=np.array([row['xyz'] for row in f]); times=np.array([row['t'] for row in f])
    target=np.array(ref[task['target']]); distance=np.linalg.norm(xyz-target,axis=1)
    stop=next((d for d in ds if d['stopping']),None)
    hold=times>=stop['t']-1e-6 if stop else np.zeros(len(times),bool)
    stable=bool(e['reason']=='stop' and hold.sum()==41 and
        np.linalg.norm(xyz[hold]-xyz[hold][0],axis=1).max()<=.12 and
        max(h['speed'] for h in e['hold'][-10:])<=.10)
    center=bool(stop and center_of_detection(stop) and len(e['hold_rgb'])==4 and all(h['centered'] for h in e['hold_rgb']))
    collisions=sum(r['collision'] for r in f);outside=sum(not r['inside'] for r in f)
    in_strict=bool(hold.any() and ((distance[hold]>=.40)&(distance[hold]<=.80)).all())
    in_rough=bool(hold.any() and ((distance[hold]>=.50)&(distance[hold]<=1.50)).all())
    descent=float(xyz[0,2]-xyz[-1,2])
    kind=folder.name.rsplit('_',1)[0]
    semantics=True;extra={}
    cfg=read(SCENE/'frame.json')
    R=np.array(cfg['rotation'])
    Ws=np.array([r['rgb_w2c'] for r in poses])
    src=np.array(cfg['origin'])+R.T@target/cfg['meters_per_source_unit']
    camera_xyz=Ws[:,:3,:3]@src+Ws[:,:3,3]
    uv=camera_xyz@atlas['k'].T
    uv=uv[:,:2]/np.maximum(uv[:,2:],1e-8)/[320,240]
    geo_center=(camera_xyz[:,2]>0)&((uv>=.30)&(uv<=.70)).all(axis=1)
    if kind=='scan':
        extra=scan_score(folder,ds,poses,f,distance,atlas)
        semantics=extra['sweep']['ordered_full_front']
        # End facing the sofa. No requirement for a whole-sofa box or a centered whole-object centroid.
        face=bool(stop and stop['detection']['boxes'] and len(e['hold_rgb'])==4 and all(h['uv'] for h in e['hold_rgb']))
    elif kind=='laptop':
        cams=(np.linalg.inv(Ws)[:,:3,3]-cfg['origin'])@R.T*cfg['meters_per_source_unit']
        optical=Ws[:,2,:3]@R.T
        # At hover, see the lit face of the open screen from above. Confirm the
        # physical screen center itself projects into the central region too.
        screen_in_view=bool(((camera_xyz[:,2]>0)&((uv>=0)&(uv<1)).all(axis=1)).all()
                            and all(d['detection']['boxes'] for d in ds))
        front=bool(hold.any() and (((cams-target)@ref['screen_normal']>0)&(cams[:,2]>target[2])&
                                        (optical[:,2]<0))[hold].all())
        semantics=front and screen_in_view
        face=center and bool(hold.any() and geo_center[hold].all())
        extra=dict(front_screen_from_above=front,screen_in_view=screen_in_view,screen_centered=bool(hold.any() and geo_center[hold].all()))
    else:
        semantics=descent>=.10
        face=center and bool(hold.any() and geo_center[hold].all())
    safe=not collisions and not outside
    common=stable and safe and semantics
    # Approximate range is diagnostic; task-level approximate still requires the instruction and stable hover.
    return dict(case=folder.name,target=task['target'],reason=e['reason'],initial_detected=bool(ds[0]['detection']['boxes']),
                end_range_m=float(distance[-1]),initial_range_m=float(distance[0]),minimum_range_m=float(distance.min()),
                end_in_rough_band=bool(.50<=distance[-1]<=1.50),stable_hover=stable,hold_in_strict=in_strict,
                hold_in_rough=in_rough,terminal_observation_ok=face,height_drop_m=descent,
                collision_events=collisions,outside_samples=outside,strict=bool(common and face and in_strict),
                approximate=bool(common and in_rough and (face if kind=='scan' else True)),
                duration_s=float(times[-1]),video_frames=len(f),**extra)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['rgb','depth','historical'],default='depth')
    parser.add_argument('--cases',nargs='+')
    args=parser.parse_args()
    root=RESULTS if args.mode=='historical' else HERE/'results/furniture_640'/args.mode
    deny_expert_reads()
    ref,atlas=reference(root)
    from metric_flight.furniture import flight_tasks
    tasks=read(MANIFEST) if args.mode=='historical' else flight_tasks()
    result={}
    for n in args.cases or tasks:
        task=tasks[n]
        folder=root/n
        if not (folder/'recording.json').exists():raise FileNotFoundError(f'Incomplete flight: {n}')
        result[n]=score(folder,task,ref,atlas)
        dump(folder/'metrics.json',result[n])
    if not args.cases:dump(root/'metrics.json',result)
    for n,row in result.items():
        print(n,row['reason'],round(row['end_range_m'],2),row['strict'],row['approximate'])
if __name__=='__main__':main()
