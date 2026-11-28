"""Select a short, stop-to-stop observation move inside measured free space."""
import numpy as np
from scipy.spatial.transform import Rotation

from metric_flight.depth_guidance import camera
from metric_flight.view_control import PITCH_LIMITS
from sim_bridge import CONTROL_DT, controller, guided_curve


def visible_corners(corners, center, axes, intrinsic, width, height):
    optical = (corners-center) @ axes
    pixels = optical @ intrinsic.T
    uv = pixels[..., :2] / np.maximum(optical[..., 2:], 1.e-7)
    visible = ((optical[..., 2] > .08) & (uv[..., 0] >= 0) & (uv[..., 0] < width)
               & (uv[..., 1] >= 0) & (uv[..., 1] < height))
    return visible, uv, optical[..., 2]


def history_corners(frames, corners):
    """Corner identities with an actual valid line of sight, including preflight."""
    seen = np.zeros(corners.shape[:2], bool)
    for frame in frames:
        h,w = frame.near.shape
        visible,uv,z = visible_corners(corners,frame.center,frame.axes,frame.intrinsic,4*w,4*h)
        if not visible.any():
            continue
        u,v = (uv[visible]/4).astype(int).T
        measured = frame.near[v,u]
        seen[visible] |= (measured > .10) & (measured >= z[visible]-.025)
    return seen


def checked_brake(guard, state):
    velocity=np.asarray(state['velocity'])
    for seconds in (.8,1.2,.5):
        # Constant deceleration to rest covers half the initial speed times duration.
        goal=np.r_[np.asarray(state['position'])+.5*seconds*velocity,state['euler_deg'][2]]
        if guard.check(state,goal,seconds,None,None)=='free':
            return goal,seconds
    return None


def check_step(guard,state,goal,seconds):
    status=guard.check(state,goal,seconds,None,None)
    if status!='free':
        return status
    curve=guided_curve(state,goal,seconds,True)
    # Check stopping options throughout this move before issuing its first command.
    for t in np.linspace(0.,seconds,1+int(np.ceil(seconds/.15))):
        pos,velocity,_,_=curve(None,t)
        moving=dict(state,position=pos[:3],velocity=velocity[:3])
        if checked_brake(guard,moving) is None:
            return 'braking_unconfirmed'
    return 'free'


def pick_step(guard, state, rig, corners, seen, visited, proposed):
    """Rank only reachable baselines; evaluate pitch views from each new camera centre."""
    body = np.asarray(state['position'])
    yaw = state['euler_deg'][2]
    center,axes,k = camera(rig)
    pitches = np.unique(np.r_[np.linspace(*PITCH_LIMITS,7), rig.pitch_deg])
    cameras = [axes @ Rotation.from_euler('x',p-rig.pitch_deg,degrees=True).as_matrix() for p in pitches]
    forward = np.array([np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw)), 0.])
    left = np.array([-forward[1], forward[0], 0.])
    up = np.array([0.,0.,1.])
    task = np.asarray(proposed)[:3]-body
    task /= max(.001,np.linalg.norm(task))
    best = None
    checks = []
    for distance in (.08,.04,.025):
        for direction in (forward,-forward,left,-left,up,-up,(forward+left)/np.sqrt(2),
                          (forward-left)/np.sqrt(2),(-forward+left)/np.sqrt(2),(-forward-left)/np.sqrt(2)):
            shift = distance*direction
            endpoint = body+shift
            if any(np.linalg.norm(endpoint-p)<.024 for p in visited):
                continue
            goal = np.r_[endpoint,yaw]
            duration = 2.0
            status = check_step(guard,state,goal,duration)
            checks.append(dict(shift=shift.tolist(),status=status))
            if status!='free':
                continue
            for pitch,cam in zip(pitches,cameras):
                mask,_,_ = visible_corners(corners,center+shift,cam,k,rig.width,rig.height)
                score=(int(mask.all(axis=1).sum()),int((mask & ~seen).sum()),int(mask.sum()),
                       float(distance),float(direction@task),-abs(pitch-rig.pitch_deg))
                if best is None or score>best[0]:
                    best=(score,goal,duration,float(pitch))
    return best,checks


def move_for_view(recovery, sim, guard, groups, advance, proposed):
    state = sim.last_receipt['state']
    if np.linalg.norm(state['velocity'])>.035 or guard.stationary(state)[0]['status']!='free':
        return None
    if recovery.steps_without_progress>=2:
        return None
    ids = np.unique(np.vstack(list(groups.values())),axis=0)
    ids = ids[guard.evidence[tuple(ids.T)]==0]
    if not len(ids):
        return None
    p = np.asarray(state['position'])
    corners = guard.origin+(ids[:,None,:]+.5+recovery.history.corners*.5)*guard.cell
    seen = history_corners(recovery.history.frames,corners)
    if not recovery.visited:
        recovery.visited.append(p.copy())
    best,checks=pick_step(guard,state,sim.scene,corners,seen,recovery.visited,proposed)
    event=dict(t=sim.last_receipt['sim_time'],start=p.tolist(),cells=ids.tolist(),checks=checks,
               status='no_free_step',end_t=sim.last_receipt['sim_time'],end=p.tolist())
    recovery.probes.append(event)
    if best is None:
        return None
    score,goal,seconds,pitch=best
    event.update(goal=goal.tolist(),pitch=pitch,predicted_full=score[0],predicted_new_corners=score[1])
    # Execute the exact certified cubic with fixed yaw and zero terminal velocity.
    traj=guided_curve(state,goal,seconds,True)
    start_pitch=sim.scene.pitch_deg
    fail=None
    for i in range(round(seconds/CONTROL_DT)):
        s=sim.last_receipt['state']
        brake=checked_brake(guard,s)
        if guard.stationary(s)[0]['status']!='free' or brake is None:
            event['current_envelope']=guard.stationary(s)[0]
            fail='probe_braking_unconfirmed'
            break
        # New depth can contradict the admitted path. Recheck its remaining
        # reference from the current physical state at each map-update cadence.
        elapsed=i*CONTROL_DT
        def remaining(_,t):
            return traj(None,elapsed+t)
        if i and sim.last_receipt['step_id']%4==0 and guard.check(s,goal,seconds-elapsed,None,None,reference=remaining)!='free':
            braking,brake_seconds=brake
            stop=guided_curve(s,braking,brake_seconds,True)
            for j in range(round(brake_seconds/CONTROL_DT)):
                action,_=controller(sim,sim.last_receipt,None,(j+.5)*CONTROL_DT,sample=stop)
                fail=advance(action,False)
                if fail or guard.stationary(sim.last_receipt['state'])[0]['status']!='free':
                    fail=fail or 'probe_braking_unconfirmed'
                    break
            event['interrupted']=True
            break
        sim.scene.pitch_deg+=float(np.clip(pitch-sim.scene.pitch_deg,-30*CONTROL_DT,30*CONTROL_DT))
        action,_=controller(sim,sim.last_receipt,None,(i+.5)*CONTROL_DT,sample=traj)
        fail=advance(action,False)
        if fail:
            break
    # Keep the camera at the reached view while observing, then restore the
    # task pitch before the next language-policy decision. Monitor hover too.
    for returning in (False,True):
        if fail:
            break
        duration=(1. if not returning else max(.6,abs(sim.scene.pitch_deg-start_pitch)/30+.2))
        ticks=int(np.ceil(duration/CONTROL_DT))
        s=sim.last_receipt['state']
        hold=np.r_[s['position'],s['euler_deg'][2]]
        if guard.check(s,hold,ticks*CONTROL_DT,None,None)!='free':
            fail='probe_hold_unconfirmed'
            break
        curve=guided_curve(s,hold,ticks*CONTROL_DT,True)
        for j in range(ticks):
            if returning:
                sim.scene.pitch_deg+=float(np.clip(start_pitch-sim.scene.pitch_deg,-30*CONTROL_DT,30*CONTROL_DT))
            action,_=controller(sim,sim.last_receipt,None,(j+.5)*CONTROL_DT,sample=curve)
            fail=advance(action,False)
            if fail or guard.stationary(sim.last_receipt['state'])[0]['status']!='free':
                fail=fail or 'probe_hold_unconfirmed'
                break
    recovery.clear(guard,ids)
    seen_after=history_corners(recovery.history.frames,corners)
    event['new_observed_corners']=int((seen_after & ~seen).sum())
    event['cells_with_new_corners']=int((seen_after & ~seen).any(axis=1).sum())
    end=np.asarray(sim.last_receipt['state']['position'])
    evidence=guard.evidence[tuple(ids.T)]
    recovery.visited.append(end.copy())
    recovery.steps_without_progress=0 if (evidence<0).any() else recovery.steps_without_progress+1
    event.update(status=fail or ('interrupted' if event.get('interrupted') else 'completed'),
                 end=end.tolist(),end_t=sim.last_receipt['sim_time'],displacement=float(np.linalg.norm(end-p)),
                 free_after=int((evidence<0).sum()),surface_after=int((evidence>0).sum()),
                 unknown_after=int((evidence==0).sum()))
    return fail
