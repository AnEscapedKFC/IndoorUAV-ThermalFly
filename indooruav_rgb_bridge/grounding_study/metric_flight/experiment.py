"""Run the metric-instruction and visible-object tasks on physical dynamics only."""
import argparse
import fcntl
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run as common
from sim_bridge import (SIM, RoomRig,CorridorRig,RoomSpace,CorridorSpace,PolicyFlightSimulator,rgb_factory,
                        projection,link_targets,guided_curve,controller,CONTROL_DT,small_update)
from sim_coordinates import EpisodeCoordinates
from policy import Policy
from metric_flight.control import Inspection,Odometer,gate,quantities,metric_waypoint
from metric_flight.course import VisualCourse
from metric_flight.record_rgbt import RgbtRecording, SHIFTS, shifted_spawn
from metric_flight.object_goal import OBJECT_INSTRUCTION, RESULT_FOLDER, MIN_RANGE_M, MAX_RANGE_M, StandoffInspection, centered
OUT=HERE/'results';DATA=common.DATA
ROOMS=['room_inspect_ac','room_obj_ac_east_to_west','room_obj_ac_west_to_east','new_room_inspect_ac',
       'holdout_room_obj_ac_east_to_west','holdout_room_obj_ac_west_to_east']
CORRIDORS=['corridor_high_return','corridor_climb_accelerate','new_corridor_high_return','new_corridor_climb_accelerate','reverse_corridor_high_return']
HOLDOUT={'room_obj_ac_east_to_west':(.16,-.12,8),'room_obj_ac_west_to_east':(-.16,.12,-8)}
# Two independent starts for the revised AC completion rule. Offsets are
# forward, left (meters) and yaw (degrees) relative to the dataset spawn.
AC_CONFIRMATION={'room_inspect_ac':(.70,.16,-7.),'room_obj_ac_west_to_east':(.55,-.22,7.)}

def speed_results(room, distance, group, mode, case):
    scene_key='room' if room else f'corridor_{distance:g}m'
    return OUT/'waypoint_speed'/scene_key/(group or 'rgb_only')/mode/case


def execute(case,switch,inspect,detector,seed=42,room_goal="standoff",record_rgbt=None,
            waypoint_speed="through",corridor_distance=13.,furniture_task=None,furniture_mode='depth',
            history_output=None):
    window=furniture_mode=='window'
    coordinated=furniture_mode in ('coordinated','window')
    history=furniture_mode=='history' or coordinated
    room='room_' in case;source=case.removeprefix('fresh_').removeprefix('holdout_').removeprefix('reverse_').removeprefix('new_')
    if furniture_task is not None:
        room=True
        meta=dict(instruction=furniture_task['instruction'],scene_version='room65_selected_anchored',seed=65)
        final_object=furniture_task['target']
    else:
        meta=common.read(DATA/'episodes'/source/'episode.json')
        stages=meta['task_spec']['ordered_subgoals']
        final_object=next((stage['landmark'] for stage in reversed(stages) if 'landmark' in stage), None)
    standoff=room and final_object=='ac' and room_goal=='standoff'
    if furniture_task is not None:
        out=history_output if history else OUT/'furniture_640'/furniture_mode/case
    else:
        out=OUT/(RESULT_FOLDER if standoff else 'room' if room else 'corridor')/(('visual' if inspect else 'hint') if room else switch)/case
        if record_rgbt:
            if room and not (standoff and inspect and switch=='auto'):
                raise ValueError('RGBT comparison requires the previous visual AC configuration')
            out=OUT/'room_ac_rgbt'/record_rgbt/case
        if waypoint_speed=='through' or (not room and (record_rgbt or corridor_distance!=13.5)):
            out=speed_results(room,corridor_distance,record_rgbt,waypoint_speed,case)
            if room and not inspect:out=out/'hint'
            if room and not standoff:out=out/'reference'
            if not room and switch!='auto':out=out/switch
        if room and switch!='auto':out=out/switch
    if furniture_task is None:out=OUT/'640x480'/out.relative_to(OUT)
    if seed!=42:out=out/str(seed)
    if (out/'episode.json').exists() and (not record_rgbt or (out/'recording.json').exists()):
        if standoff:
            from metric_flight.score_object import score
            common.dump(out/'two_tier_metrics.json',score(out))
        return
    out.mkdir(parents=True,exist_ok=True)
    init=furniture_task['initial'] if furniture_task is not None else common.line(DATA/'episodes'/source/'frames.jsonl')
    xyz=np.array(init['position_w_m'],float);yaw0=Rotation.from_quat(np.array(init['q_w_from_b_wxyz'])[[1,2,3,0]]).as_euler('xyz',degrees=True)[2]
    offsets=HOLDOUT if case.startswith('holdout_') else {'corridor_high_return':(.45,-.10,-160)} if case.startswith('reverse_') else common.OFFSETS
    if case.startswith('fresh_'):offsets=AC_CONFIRMATION
    if case!=source:
        fw,lf,dy=offsets[source];rot=np.deg2rad(yaw0);xyz[:2]+=np.array([fw*np.cos(rot)-lf*np.sin(rot),fw*np.sin(rot)+lf*np.cos(rot)]);yaw0+=dy
    if record_rgbt=='shifted':xyz,yaw0=shifted_spawn(xyz,yaw0,SHIFTS[case])
    text=meta['instruction'] if room else f'Fly forward {corridor_distance:g} meters.'
    if standoff:text+=OBJECT_INSTRUCTION
    active=gate(text,switch);height,length=quantities(text)
    if room:assert not active
    mapping=EpisodeCoordinates(xyz,yaw0);prior=mapping.to_native(xyz,yaw0)
    policy=Policy(text);servo=StandoffInspection(text) if standoff else Inspection(text);odo=Odometer(xyz,yaw0)
    if history:
        from metric_flight.history_control import HistoryInspection
        if coordinated:
            from metric_flight.view_control import CoordinatedInspection, PITCH_LIMITS, pointing_request, elevation
            from metric_flight.depth_guidance import camera
            servo=CoordinatedInspection(out,final_object,text,single_frame=window)
        else:
            servo=HistoryInspection(out,final_object,text)
    elif furniture_task is not None:
        from metric_flight.furniture_control import FurnitureInspection
        servo=FurnitureInspection(final_object,text,use_depth=furniture_mode=='depth')
    rig,known=(RoomRig,RoomSpace()) if room else (CorridorRig,CorridorSpace())
    scene=SIM/'scenes'/meta['scene_version']
    landmarks=link_targets(meta,DATA) if room and furniture_task is None else {}
    course=VisualCourse(odo.axis) if active and length is not None else None
    frames=[];ds=[];hold=[];hold_rgb=[];reason=None
    hold_duration=2.
    attempts=[]
    if window:
        from metric_flight.observation_window import DoorChair, HoldTracker, box_iou, hold_seconds, write_sample
        hold_duration=hold_seconds(text)
        instance=DoorChair() if final_object=='armchair' else None
    previous_exit_velocity=None
    pitch_setpoint=init['rig_pitch_deg']
    camera_factory=rgb_factory(rig,init['rig_pitch_deg'],render_thermal=bool(record_rgbt))
    depth_enabled=furniture_task is not None and furniture_mode in ('depth','history','coordinated','window')
    guard=None
    if depth_enabled:
        from metric_flight.depth_guard import DepthGuard,preflight,survey_rig
        camera_factory=survey_rig(camera_factory, PITCH_LIMITS if coordinated else None)
    with (RgbtRecording(out,meta['scene_version']) if record_rgbt else nullcontext()) as video, PolicyFlightSimulator(scene/'models/rgb.ply',point_cloud=scene/'collision/rgb_surface_m.ply',config={'limits':{'max_steps':1500}},
        seed=meta['seed'],spawn=xyz.tolist(),yaw=yaw0,resolution=1.0,spawn_velocity_w_mps=init['velocity_w_mps'],scene_factory=camera_factory) as sim:
        if depth_enabled:
            guard=DepthGuard(xyz,sim.obstacles.safety_half_size)
            survey=preflight(sim,guard,pitch_limits=PITCH_LIMITS if coordinated else None)
            common.dump(out/'depth_preflight.json',survey)
            if coordinated and survey['after']['status']!='free':
                reason='start_clearance_unconfirmed'
        r=sim.last_receipt
        def detect_current(path, noun):
            if window and instance is not None:
                return instance.detect(detector,path,sim.scene,sim.last_depth)
            return detector(path,noun)
        def record():
            if guard is not None and r['step_id']%4==0 and (not coordinated or r['step_id']>0):guard.observe(sim.last_depth,sim.scene)
            if video is not None:video.frame(sim,r)
            st=r['state'];p=np.asarray(st['position']);q=np.array(st['quaternion_wxyz'])
            inside=bool(known.contains_box(p,Rotation.from_quat(q[[1,2,3,0]]).as_matrix(),sim.obstacles.safety_half_size))
            odo.tick(p);frames.append(dict(t=r['sim_time'],xyz=p.tolist(),yaw=st['euler_deg'][2],v=st['velocity'],
                     rig_pitch_deg=float(sim.scene.pitch_deg),inside=inside,collision=r['collision']['event'] is not None,objects={k:projection(sim,v) for k,v in landmarks.items()},
                     cumulative_m=odo.path,forward_m=odo.forward))
            if coordinated:
                frames[-1]['optical_elevation_deg']=elevation(camera(sim.scene)[1][:,2])
            if window and r['step_id']%10==0:
                write_sample(out,r['step_id'],sim.last_rgb,sim.last_depth,sim.scene)
            return inside
        assert record() and sim.obstacles.distance(xyz,r['state']['quaternion_wxyz'])['safety_clearance']>0
        if coordinated:
            Image.fromarray(np.rint(sim.last_rgb*255).astype(np.uint8)).save(out/'start.png')
        for i in range(130):
            if reason:break
            if r['sim_time']>=60-1e-6:reason='limit';break
            st=r['state'];p=np.array(st['position']);yaw=st['euler_deg'][2];v=np.array(st['velocity']);now=mapping.to_native(p,yaw)
            path=out/f'{i:03d}.png'
            Image.fromarray(np.rint(sim.last_rgb*255).astype(np.uint8)).save(path,compress_level=1)
            if room:
                prompt,det=policy.update(now,path,detect_current,lambda b:common.world_ray(sim.scene,b,yaw))
                if furniture_task is not None and final_object=='sofa' and servo.scan:
                    # Output-only sensor tape for offline occlusion checks.
                    z=np.nan_to_num(sim.last_depth,nan=0.,posinf=0.,neginf=0.)
                    Image.fromarray(np.rint(1000*np.clip(z,0.,65.)).astype(np.uint16)).save(out/f'depth_{i:03d}.png')
                if furniture_task is None and i==0 and not det['boxes']:raise ValueError(f'Target missing at declared visible spawn: {case}')
            else:
                det=None;prompt=text+f' Distance flown: {odo.path:.2f} m. Forward displacement: {odo.forward:.2f} m. Remaining distance: {max(0.,length-odo.path):.2f} m.'
            req=dict(rgb_path=str(path),ref_rgb_path=str(out/'000.png'),instruction=prompt,state=now.tolist(),seed=seed+i,id=f'metric_{case}_{i}')
            raw=common.infer(req,use_cache=not record_rgbt)[9];raw_stop=bool(small_update(raw,prior));prior=raw.copy()
            target=mapping.to_world(raw);seconds=1.;stop=raw_stop;info={}
            if not room:
                if active:
                    course_info=course.update(sim.last_rgb,sim.scene) if course is not None else None
                    target=metric_waypoint(text,switch,target,p,yaw,odo,seconds,course.axis if course is not None else None)
                    stop=False
                info=dict(remaining_m=length-odo.path,course=course_info if active else None)
            elif inspect:
                seconds=.5
                extra={'depth':sim.last_depth} if depth_enabled else {}
                if history:extra['receipt']=r
                target,stop,info=servo.step(p,yaw,v,target,det,sim.scene,seconds,sim.last_rgb,**extra)
                if depth_enabled:
                    # Detection may disappear briefly on purpose at the edge of a sofa scan.
                    if servo.lost>=8 and info.get('phase') not in {'scan_left','scan_right'}:reason='target_lost'
                elif servo.lost>=4:reason='target_lost'
                if furniture_task is not None and i==0 and not det['boxes']:reason='initial_target_missing'
            ds.append(dict(n=i+1,t=r['sim_time'],raw_world=mapping.to_world(raw).tolist(),world_target=target.tolist(),raw_stop=raw_stop,stopping=stop,
                           prompt=prompt,gate=active,info=info,detection=det))
            if reason:break
            legacy_hold=stop or (room and inspect)
            holding_phase=info.get('phase') in {'terminal_hold','observation_lost','settle'}
            passing=waypoint_speed=='through' and not stop and not holding_phase
            if guard is not None:
                proposed=target.copy()
                goal=np.r_[p,yaw] if stop else target
                observation=info.get('anchor') or {}
                anchor=observation.get('world') if observation.get('confirmed') else None
                target,safety=guard.select(st,goal,hold_duration if stop else seconds,passing,previous_exit_velocity,
                                           rig=sim.scene,anchor=anchor)
                # When a lower observation would move the body near a surface, keep
                # the safety envelope and re-aim the sensor from the reachable pose.
                if history:
                    pitch_setpoint=info.get('pitch_request',sim.scene.pitch_deg)
                elif observation.get('confirmed') and info.get('range_est_m',np.inf)<1.30:
                    vertical=observation['uv'][1]
                    if abs(vertical-.5)>.09:
                        error=np.rad2deg(np.arctan((vertical-.5)*sim.scene.height/sim.scene.ks['rgb'][1,1]))
                        pitch_setpoint=float(np.clip(sim.scene.pitch_deg-error,-65.,15.))
                    else:pitch_setpoint=sim.scene.pitch_deg
                info['gimbal_setpoint_deg']=float(pitch_setpoint)
                info['depth_guard']=safety
                ds[-1]['proposed_world']=proposed.tolist()
                ds[-1]['world_target']=target.tolist()
                if safety['choice']=='brake':
                    stop=passing=False
                    ds[-1]['stopping']=False
                if guard.stalled>=8:
                    reason='depth_path_unconfirmed';break
            if stop:
                target=np.r_[p,yaw];seconds=hold_duration;hold=[dict(xyz=p.tolist(),speed=st['speed'])]
                if window:
                    hold_rgb=[]
                    first_box=det['boxes'][0]['xyxy']
                    mask=servo.memory.instance_mask(np.rint(255*sim.last_rgb).astype(np.uint8),first_box)
                    tracker=HoldTracker(sim.last_rgb,sim.last_depth,sim.scene,first_box,mask)
                    attempt=dict(t=r['sim_time'],decision=i,qualified=info['phase']=='terminal_hold')
                    attempts.append(attempt)
            start_velocity=previous_exit_velocity if passing else None
            end_velocity=(target[:3]-p)/seconds if passing else None
            sample=guided_curve(st,target,seconds,legacy_hold,
                                start_velocity=start_velocity,end_velocity=end_velocity)
            ds[-1]['reference_velocity_in']=sample(None,0.)[1][:3].tolist()
            ds[-1]['reference_velocity_out']=sample(None,seconds)[1][:3].tolist()
            ds[-1]['passing_waypoint']=bool(passing)
            previous_exit_velocity=end_velocity
            for k in range(round(seconds/CONTROL_DT)):
                u,_=controller(sim,r,None,(k+.5)*CONTROL_DT,sample=sample)
                if guard is not None:
                    if coordinated and servo.aim is not None:
                        pitch_setpoint=pointing_request(sim.scene,servo.aim)
                    sim.scene.pitch_deg+=float(np.clip(pitch_setpoint-sim.scene.pitch_deg,-12*CONTROL_DT,12*CONTROL_DT))
                r=sim.step(u,format='body_rates_thrust');inside=record()
                if stop:
                    hold.append(dict(xyz=r['state']['position'],speed=r['state']['speed']))
                    if (standoff or furniture_task is not None) and (k+1)%round(.5/CONTROL_DT)==0:
                        # Independent RGB checks during the final hover, at 2 Hz.
                        hold_path=out/(f'attempt_{len(attempts):02d}_{k+1:03d}.png' if window else f'hold_{k+1:03d}.png')
                        Image.fromarray(np.rint(sim.last_rgb*255).astype(np.uint8)).save(hold_path)
                        boxes=detect_current(hold_path,det['object'])['boxes']
                        box=boxes[0]['xyxy'] if boxes else None
                        uv=[(box[0]+box[2])/2,(box[1]+box[3])/2] if box else None
                        metrics=servo.hold_metrics(sim.last_rgb,sim.last_depth,sim.scene,box) if coordinated else dict(uv=uv,centered=bool(uv and centered(*uv)))
                        if window:
                            tracked=tracker.measure(sim.last_rgb,sim.last_depth,sim.scene)
                            source='detector' if box is not None else 'missing'
                            if tracked is not None and (box is None or box_iou(box,tracked)<.35):
                                metrics=servo.hold_metrics(sim.last_rgb,sim.last_depth,sim.scene,tracked)
                                source='rgbd_track'
                            metrics.update(source=source,detected_box=box,tracked_box=tracked)
                        hold_rgb.append(dict(t=r['sim_time'],**metrics))
                if r['terminated'] or r['truncated'] or (room and not inside):
                    reason=r['reason'] or 'outside_known_space';break
                if not room and odo.path>=length-1e-6:
                    reason='distance_reached';break
                if r['sim_time']>=60-1e-6:reason='limit';break
            if stop and window:
                drift=max(np.linalg.norm(np.subtract(v['xyz'],hold[0]['xyz'])) for v in hold)
                confirmed=bool(not reason and len(hold_rgb)==round(2*hold_duration) and drift<=.12 and
                               all(row['framing_ok'] and row['view_ok'] and row['parts_ok'] for row in hold_rgb))
                attempt.update(confirmed=confirmed,max_drift_m=float(drift),checks=hold_rgb)
                if confirmed:reason='stop'
                else:servo.ready=0
            elif stop and not reason:reason='stop'
            if reason:break
        Image.fromarray(np.rint(sim.last_rgb*255).astype(np.uint8)).save(out/'end.png')
        for filename,obj in [('frames',frames),('decisions',ds),('episode',dict(case=case,source=source,mode='flight',instruction=text,gate=active,
                switch=switch,inspection=inspect,seed=seed,spawn=xyz.tolist(),spawn_yaw=yaw0,end=frames[-1]['xyz'],end_yaw=frames[-1]['yaw'],hold=hold,
                waypoint_speed=waypoint_speed,render_wh=[sim.scene.width,sim.scene.height],furniture_mode=furniture_mode if furniture_task else None,recording_group=record_rgbt, fixed_pitch_deg=None if depth_enabled else init['rig_pitch_deg'],
                initial_pitch_deg=init['rig_pitch_deg'],
                room_goal='furniture' if furniture_task is not None else ('standoff' if standoff else 'reference') if room else None,hold_rgb=hold_rgb,
                object_distance_range_m=[MIN_RANGE_M,MAX_RANGE_M] if standoff else None,
                reason=reason or 'limit',duration=frames[-1]['t'],distance_setpoint=length,altitude_setpoint=height))]:
            common.dump(out/(filename+'.json'),obj)
        if window:common.dump(out/'hold_attempts.json',attempts)
        if history:
            servo.memory.finish()
        if standoff:
            from metric_flight.score_object import score
            common.dump(out/'two_tier_metrics.json',score(out))
        print(case,switch,inspect,reason,len(ds),round(odo.forward,2),round(odo.path,2),flush=True)
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene',choices=['corridor','room'])
    parser.add_argument('--waypoint-check',choices=['auto','off'],default='auto')
    parser.add_argument('--room-policy',choices=['visual','hint'],default='visual')
    parser.add_argument('--room-goal',choices=['standoff','reference'],default='standoff')
    parser.add_argument('--record-rgbt',choices=['repeat','shifted'])
    parser.add_argument('--waypoint-speed',choices=['legacy','through'],default='through')
    parser.add_argument('--corridor-distance',type=float,default=13.)
    parser.add_argument('--cases',nargs='+');parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args(argv);cases=ROOMS if args.scene=='room' else CORRIDORS
    if not np.isfinite(args.corridor_distance) or args.corridor_distance<=0:parser.error('distance must be positive and finite')
    if args.record_rgbt=='shifted' and args.scene!='room':parser.error('shifted camera group applies to the room')
    selected=args.cases or (list(SHIFTS) if args.record_rgbt and args.scene=='room' else cases)
    allowed=set(cases)
    if args.scene=='room':
        allowed.update('fresh_'+name for name in AC_CONFIRMATION)
    if any(n not in allowed for n in selected):parser.error('case outside selected scene')
    det=None
    if args.scene=='room':
        from detector import Detector
        det=Detector(use_cache=not args.record_rgbt)
    for n in selected:
        lock=nullcontext()
        if args.record_rgbt:
            folder=(OUT/'room_ac_rgbt'/args.record_rgbt/n if args.scene=='room' and args.waypoint_speed=='legacy'
                    else speed_results(args.scene=='room',args.corridor_distance,args.record_rgbt,args.waypoint_speed,n))
            folder=OUT/'640x480'/folder.relative_to(OUT)
            folder.mkdir(parents=True,exist_ok=True)
            lock=(folder/'.record.lock').open('a')
        with lock as handle:
            if args.record_rgbt:fcntl.flock(handle,fcntl.LOCK_EX)
            execute(n,args.waypoint_check,args.scene=='room' and args.room_policy=='visual',
                    det,args.seed,args.room_goal,args.record_rgbt,args.waypoint_speed,args.corridor_distance)
if __name__=='__main__':main()
