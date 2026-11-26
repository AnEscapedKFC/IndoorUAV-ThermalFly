"""Run an external IndoorUAV policy on two unchanged RGB 3DGS simulator scenes."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from PIL import Image
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation
from sim_coordinates import EpisodeCoordinates, signed_degrees
from sim_evidence import add_evidence
from sim_limits import bounded_goal

BRIDGE = Path(__file__).resolve().parent
SIM = Path(os.environ.get('GS_GRADNAV_SIM', BRIDGE.parent / 'gs_gradnav_simulator')).resolve()
sys.path.insert(0, str(SIM))
sys.path.insert(1, str(SIM / 'scenes/room65_selected_anchored'))
from dynamics import CONTROL_DT
from engine import PolicyFlightSimulator
from record_sequence import FpvVideoWriter, verify_video
from scenes.room65_selected_anchored.fly import KnownSpace as RoomSpace, controller
from scenes.room65_selected_anchored.room import RigScene as RoomRig
from scenes.corridor_native.room import KnownSpace as CorridorSpace, RigScene as CorridorRig


def get_json(path):
    return json.loads(path.read_text())


def write_json_line(handle, value):
    handle.write(json.dumps(value, allow_nan=False) + '\n')
    handle.flush()


class ModelProcess:
    def __init__(self, cmd, log):
        self.proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=log, text=True, bufsize=1)

    def close(self):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=20)
        self.proc.stdout.close()

    def predict(self, request):
        write_json_line(self.proc.stdin, request)
        for line in self.proc.stdout:
            response = json.loads(line)
            if response.get('ready') or response.get('type') == 'ready':
                continue
            if response.get('id') != request['id'] or 'error' in response:
                raise RuntimeError(f'Worker failed: {response}')
            result = np.asarray(response['waypoints'], dtype=float)
            if result.ndim != 2 or result.shape[0] < 10 or result.shape[1] < 4 or not np.isfinite(result).all():
                raise ValueError(f'Invalid action chunk for {request["id"]}: {result.shape}')
            return result[:, :4]
        raise RuntimeError(f'Worker closed stdout (exit={self.proc.poll()}); see worker.stderr')


def rgb_factory(base, pitch_deg, *, render_thermal=False):
    class RgbRig(base):
        def __init__(self, *args, **kwargs):
            self.pitch_deg = float(pitch_deg)
            super().__init__(*args, render_thermal=render_thermal, **kwargs)

        def render(self, camera):
            c2w = np.asarray(camera).copy()
            c2w[:3, :3] = c2w[:3, :3] @ Rotation.from_euler('x', self.pitch_deg, degrees=True).as_matrix()
            return super().render(c2w)
    return RgbRig


def link_targets(meta, dataset):
    scene = dataset / 'scenes' / meta['scene_version']
    landmarks = {}
    for name in ('landmarks.json', 'object_landmarks.json'):
        if (scene / name).exists():
            landmarks.update(get_json(scene / name))
    needed = {s['landmark'] for s in meta['task_spec']['ordered_subgoals'] if 'landmark' in s}
    return {name: np.asarray(value['center_w_m']) for name,value in landmarks.items() if name in needed}


def projection(sim, point):
    rig = sim.scene
    native = rig.origin + rig.R.T @ point / rig.s
    optical = (rig.W @ np.r_[native, 1.])[:3]
    if optical[2] <= 0:
        return dict(in_image=False, uv=None)
    homogeneous = rig.ks['rgb'] @ optical
    uv = homogeneous[:2] / homogeneous[2]
    return dict(in_image=bool(np.all(uv >= 0) and np.all(uv < [rig.width, rig.height])), uv=uv.tolist())


def guided_curve(state, goal, seconds, holding, *, start_velocity=None, end_velocity=None):
    start = np.r_[state['position'], state['euler_deg'][2]]
    finish = np.array(goal, dtype=float)
    finish[3] = start[3] + signed_degrees(finish[3] - start[3])
    start_rate = np.r_[state['velocity'], np.rad2deg(state['angular_velocity'][2])]
    finish_rate = np.zeros(4) if holding else (finish - start) / seconds
    # Optional world-frame translation derivatives; yaw keeps its original boundary conditions.
    if start_velocity is not None:
        start_rate[:3] = start_velocity
    if end_velocity is not None:
        finish_rate[:3] = end_velocity
    curve = CubicHermiteSpline([0., seconds], [start, finish], [start_rate, finish_rate])
    def sample(_unused, t):
        return curve(t), curve(t, 1), curve(t, 2), 0
    return sample


def small_update(a, b):
    gap = np.linalg.norm(a[:3] - b[:3])
    angle = abs((a[3] - b[3] + np.pi) % (2*np.pi) - np.pi)
    return gap < .2 and angle < np.pi/12


def run(args):
    episode_dir = args.dataset.resolve() / 'episodes' / args.episode
    meta = get_json(episode_dir / 'episode.json')
    with (episode_dir / 'frames.jsonl').open() as f:
        initial = json.loads(next(f))
    if args.mode == 'oracle':
        with (episode_dir / 'frames.jsonl').open() as f:
            oracle_rows = [json.loads(line) for line in f]
        oracle_stop = int(meta['waypoint_spec']['stop_frame_id'])
    source = meta['scene_version']
    limits = get_json(args.limits)['scenes'][source]['limits'] if args.limits else None
    if source == 'room65_selected_anchored':
        rig, known = RoomRig, RoomSpace()
    elif source == 'corridor_native':
        rig, known = CorridorRig, CorridorSpace()
    else:
        raise ValueError(source)
    # Legacy programmatic callers run fixed-camera control experiments.
    pitch_mode = getattr(args, 'gimbal', None) or ('recorded' if args.mode == 'replay' else 'fixed')
    if pitch_mode == 'auto':
        if args.mode == 'replay':
            pitch_mode = 'recorded'
        else:
            pitch_mode = 'learned' if rig is RoomRig else 'fixed'
    if pitch_mode == 'recorded' and args.mode != 'replay':
        raise ValueError('Recorded pitch is available with expert control replay')
    if pitch_mode == 'learned' and rig is not RoomRig:
        raise ValueError('The learned pitch policy was trained for the meeting room. Use auto or fixed for the corridor.')
    servo = None
    if pitch_mode == 'learned':
        from gimbal.policy import PitchPolicy, PitchServo
        servo = PitchServo(PitchPolicy(args.gimbal_weights), initial['rig_pitch_deg'])
    initial_yaw = float(Rotation.from_quat(np.asarray(initial['q_w_from_b_wxyz'])[[1,2,3,0]]).as_euler('xyz',degrees=True)[2])
    pose_map = EpisodeCoordinates(initial['position_w_m'], initial_yaw, args.native_start_yaw_deg)
    out = args.output.resolve()
    if out == SIM or SIM in out.parents:
        raise ValueError('Choose an output directory outside the simulator source tree')
    out.mkdir(exist_ok=False, parents=True)
    rgb_dir = out/'rgb'
    rgb_dir.mkdir()
    anchors = link_targets(meta, args.dataset)
    thresholds = meta['task_spec']
    min_gap = None
    closest_goal = float('inf')
    path_length = 0.
    first_seen = {}
    outcome = None
    last_xyz = None
    frame_count = 0
    n_infers = 0
    hold_start = None
    hold_positions = []
    hold_speeds = []
    clipped_moves = 0
    initial_image_delta = None
    folder = SIM/'scenes'/source
    actual_instruction = meta['instruction'] if args.instruction is None else args.instruction
    subinstructions = get_json(args.subinstructions) if args.subinstructions else [actual_instruction]
    if not isinstance(subinstructions, list) or not subinstructions or any(
            not isinstance(s,str) or not s.strip() for s in subinstructions):
        raise ValueError('Subinstructions must be a nonempty JSON list of nonempty strings')
    stage = 0
    config = {'limits':{'max_steps': max(int(args.max_seconds/CONTROL_DT)+60, meta['final_frame_id']+10)}}
    with ExitStack() as stack:
        sim = stack.enter_context(PolicyFlightSimulator(folder/'models/rgb.ply',
            point_cloud=folder/'collision/rgb_surface_m.ply', config=config,seed=meta['seed'],
            spawn=initial['position_w_m'], yaw=initial_yaw, resolution=.5,
            spawn_velocity_w_mps=initial['velocity_w_mps'],scene_factory=rgb_factory(rig, initial['rig_pitch_deg'])))
        frames = stack.enter_context((out/'frames.jsonl').open('x'))
        decisions = stack.enter_context((out/'decisions.jsonl').open('x'))
        pitch_log = stack.enter_context((out/'gimbal.jsonl').open('x')) if servo else None
        if args.mode == 'model':
            log = stack.enter_context((out/'worker.stderr').open('x'))
            worker = ModelProcess(args.worker, log)
            stack.callback(worker.close)
        video = FpvVideoWriter(out/'rgb_fpv.mp4', sim.scene.width, sim.scene.height)
        stack.callback(video.close)
        r = sim.last_receipt

        def step_with_camera(control):
            if servo:
                pitch = sim.scene.pitch_deg
                if servo.update(r['sim_time'], sim.last_rgb, subinstructions[stage], pitch):
                    write_json_line(pitch_log, dict(frame_id=r['frame_id'], t_s=r['sim_time'],
                        instruction=subinstructions[stage], current_pitch_deg=pitch,
                        target_pitch_deg=servo.target_deg, commanded_rate_deg_s=servo.rate))
                sim.scene.pitch_deg = servo.advance(pitch, CONTROL_DT)
            return sim.step(control, format='body_rates_thrust')

        def record():
            nonlocal min_gap, closest_goal, path_length, last_xyz, frame_count, outcome, initial_image_delta
            st = r['state']
            pos = np.asarray(st['position'])
            if last_xyz is not None:
                path_length += float(np.linalg.norm(pos - last_xyz))
            last_xyz = pos.copy()
            rot = Rotation.from_quat(np.asarray(st['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
            inside = known.contains_box(pos, rot, sim.obstacles.safety_half_size)
            views = {k:projection(sim, p) for k,p in anchors.items()}
            for k,v in views.items():
                if v['in_image']:
                    first_seen.setdefault(k, r['sim_time'])
            gap = r['min_sampled_safety_clearance']
            if gap is not None:
                min_gap = gap if min_gap is None else min(min_gap, gap)
            closest_goal = min(closest_goal,float(np.linalg.norm(pos - thresholds['goal_w_m'])))
            if r['terminated'] or r['truncated']:
                outcome = r['reason'] or 'simulator_termination'
            elif not inside:
                outcome = 'outside_known_space'
            write_json_line(frames,dict(frame_id=r['frame_id'], t_s=r['sim_time'], position_w_m=st['position'],
                q_w_from_b_wxyz=st['quaternion_wxyz'], yaw_deg=st['euler_deg'][2], velocity_w_mps=st['velocity'],
                speed_mps=st['speed'], commanded_body_rates_thrust=r['raw_action'],applied_control=r['applied_control'],
                collision_event=r['collision']['event'],clearance_m=gap,in_known_space=bool(inside),
                pitch_deg=sim.scene.pitch_deg,landmark_projections=views))
            video.write(sim.last_rgb)
            frame_count += 1
            if r['frame_id']==0:
                rgb = np.rint(sim.last_rgb*255).astype(np.uint8)
                recorded = np.asarray(Image.open(episode_dir/initial['rgb_path']).convert('RGB'), dtype=np.int16)
                initial_image_delta = float(np.abs(rgb.astype(np.int16)-recorded).mean())
        record()
        if args.mode == 'replay':
            with (episode_dir/'frames.jsonl').open() as f:
                next(f)
                for line in f:
                    src = json.loads(line)
                    if outcome: break
                    if pitch_mode == 'recorded':
                        sim.scene.pitch_deg = src['rig_pitch_deg']
                    r = step_with_camera(src['commanded_body_rates_thrust'])
                    record()
            outcome = outcome or 'expert_replay_complete'
        else:
            last_prediction = pose_map.to_native(initial['position_w_m'], initial_yaw)
            reference = rgb_dir/'000000.png'
            while not outcome and r['sim_time'] + 1.e-6 < args.max_seconds:
                st = r['state']
                now = pose_map.to_native(st['position'], st['euler_deg'][2])
                path = rgb_dir/f'{r["frame_id"]:06d}.png'
                Image.fromarray(np.rint(sim.last_rgb*255).astype(np.uint8)).save(path, compress_level=1)
                request = dict(id=f'{args.episode}-{n_infers}', instruction=subinstructions[stage],
                               rgb_path=str(path), ref_rgb_path=str(reference), state=now.tolist(),seed=args.seed+n_infers)
                started = time.monotonic()
                seconds = args.segment_seconds
                if args.mode == 'oracle':
                    # Measured future expert pose through exactly the model's coordinate and tracker path.
                    stopping = r['frame_id'] >= oracle_stop
                    target_index = min(r['frame_id']+round(seconds/CONTROL_DT), oracle_stop)
                    row = oracle_rows[target_index]
                    yaw = Rotation.from_quat(np.asarray(row['q_w_from_b_wxyz'])[[1,2,3,0]]).as_euler('xyz',degrees=True)[2]
                    chosen = pose_map.to_native(row['position_w_m'], yaw)
                    chunk = None
                    if not stopping:
                        seconds = (target_index-r['frame_id'])*CONTROL_DT
                else:
                    chunk = worker.predict(request)
                    chosen = chunk[9]
                    stopping = bool(small_update(chosen, last_prediction))
                advanced = stopping and stage < len(subinstructions)-1
                raw_target = pose_map.to_world(chosen)
                target = raw_target
                bounding = None
                if limits is not None and (not stopping or advanced):
                    target,bounding = bounded_goal(st,raw_target,limits,seconds)
                    if bounding['translation_scale']<1. or bounding['yaw_clipped']:
                        clipped_moves += 1
                write_json_line(decisions,dict(request=request, frame_id=r['frame_id'], latency_s=time.monotonic()-started,
                    stage=stage, advanced_stage=advanced, native_chunk=None if chunk is None else chunk.tolist(),
                    target_pose_w_m_deg=target.tolist(), raw_target_pose_w_m_deg=raw_target.tolist(), bounding=bounding, stop_heuristic=stopping if args.mode=='model' else None,
                    oracle_stop=stopping if args.mode=='oracle' else None))
                last_prediction = chosen
                n_infers += 1
                if advanced:
                    # Upstream switches instruction and reference on the current frame, then executes its pending pose.
                    stage += 1
                    reference = path
                    stopping = False
                if stopping:
                    target = np.r_[st['position'],st['euler_deg'][2]]
                    hold_start = np.asarray(st['position'])
                    seconds = args.hold_seconds
                ticks = round(seconds/CONTROL_DT)
                sample = guided_curve(st, target, seconds, stopping)
                for i in range(ticks):
                    u,_ = controller(sim, r, None, (i+.5)*CONTROL_DT,sample=sample)
                    r = step_with_camera(u)
                    record()
                    if stopping:
                        hold_positions.append(r['state']['position'])
                        hold_speeds.append(r['state']['speed'])
                    if outcome or (not stopping and r['sim_time'] + 1.e-6 >= args.max_seconds):
                        break
                if stopping:
                    outcome = outcome or ('motion_threshold_stop' if args.mode=='model' else 'expert_timed_stop')
            outcome = outcome or 'time_limit'

        st = r['state']
        dist = float(np.linalg.norm(last_xyz - thresholds['goal_w_m']))
        yaw_err = abs(signed_degrees(st['euler_deg'][2] - thresholds['goal_yaw_deg']))
        near = dist <= thresholds['position_tol_m'] and yaw_err <= thresholds['yaw_tol_deg'] and st['speed'] <= thresholds['speed_tol_mps']
        hold_drift = float(np.max(np.linalg.norm(np.asarray(hold_positions)-hold_start,axis=1))) if hold_positions else None
        hold_final_speed = max(hold_speeds[-10:]) if hold_speeds else None
        stable = bool(hold_drift is not None and hold_drift <= thresholds['position_tol_m'] and hold_final_speed <= thresholds['speed_tol_mps'])
        summary = dict(episode=args.episode, split=next((k for k,ids in get_json(args.dataset/'splits.json')['splits'].items() if args.episode in ids),None),
            mode=args.mode, scene=source, instruction=actual_instruction,outcome=outcome,duration_s=r['sim_time'],
            frames=frame_count,inferences=n_infers if args.mode=='model' else 0,decisions=n_infers,subinstructions=subinstructions,final_stage_index=stage,spawn_w_m=initial['position_w_m'],end_w_m=st['position'],
            goal_w_m=thresholds['goal_w_m'],terminal_position_error_m=dist,closest_goal_distance_m=closest_goal,
            terminal_yaw_error_deg=yaw_err,terminal_speed_mps=st['speed'],path_m=path_length,
            min_sampled_clearance_m=min_gap,collision_event=r['collision']['event'],outside_known_space=outcome=='outside_known_space',
            first_landmark_center_in_rgb_s=first_seen,terminal_landmark_projections={k:projection(sim,p) for k,p in anchors.items()},
            initial_image_mae_u8=initial_image_delta,terminal_within_tolerance=bool(near),stop_hold_max_drift_m=hold_drift,
            stop_final_half_second_max_speed_mps=hold_final_speed,stable_stop=stable,
            strict_navigation_success=bool(near and outcome=='motion_threshold_stop' and stable) if args.mode=='model' else None,
            source_expert_frames=meta['final_frame_id']+1,source_expert_duration_s=meta['flight_metrics']['duration_s'],
            segment_seconds=args.segment_seconds,hold_seconds=args.hold_seconds,max_seconds=args.max_seconds,
            gimbal=pitch_mode, pitch_policy={'fixed':'fixed at recorded spawn pitch',
                'recorded':'recorded at every step (expert control replay)',
                'learned':'independent RGB pitch prediction at 1 Hz, rate-limited at 20 Hz'}[pitch_mode],
            gimbal_weights=str(args.gimbal_weights.resolve()) if servo else None,
            motion_limits=limits, motion_limits_source=None if args.limits is None else str(args.limits.resolve()),
            clipped_move_decisions=clipped_moves, native_start_yaw_deg=pose_map.native_heading,
            state_gauge='episode-local x=y=0, native horizontal basis rotated with chosen initial heading, height retained, clockwise yaw in [0,2pi)',
            native_action_rule='absolute waypoint index 9, small-update stop against last output or spawn, full original instruction',
            landmark_metric='Geometric RGB projection only; visibility does not resolve occlusion or recognition.')
    summary['video_verification']=verify_video(out/'rgb_fpv.mp4',frame_count)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    summary = add_evidence(args.dataset, args.episode, out)
    print(json.dumps(summary))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=SIM/'outputs/expert_rgbt_20261123')
    parser.add_argument('--episode',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['model','replay','oracle'],default='model')
    parser.add_argument('--gimbal',choices=['auto','learned','fixed','recorded'],
                        help='Select auto for learned room pitch in navigation and fixed corridor pitch. '
                             'With no flag, navigation uses fixed pitch and expert replay uses recorded pitch.')
    parser.add_argument('--gimbal-weights',type=Path,default=BRIDGE/'gimbal/room_pitch.npz')
    parser.add_argument('--worker',default=str(BRIDGE/'scripts/model_worker.sh'))
    parser.add_argument('--instruction',default=None,help='Override instruction; original goal is still used for descriptive distance metrics')
    parser.add_argument('--subinstructions',type=Path,default=None,help='Manual text-only decomposition JSON; switch on native motion criterion')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--limits',type=Path,help='Secondary experiment: simulation motion limits computed by sim_limits.py from training trajectories')
    parser.add_argument('--native-start-yaw-deg',type=float,default=0.,help='Rotate horizontal pose coordinates and initial heading together; flight and RGB stay unchanged')
    parser.add_argument('--segment-seconds',type=float,default=1.)
    parser.add_argument('--hold-seconds',type=float,default=2.)
    parser.add_argument('--max-seconds',type=float,default=60.)
    args=parser.parse_args()
    for duration in [args.segment_seconds,args.hold_seconds,args.max_seconds]:
        if not np.isfinite(duration) or duration <= 0 or not np.isclose(duration/CONTROL_DT,round(duration/CONTROL_DT)):
            parser.error('Time intervals must be positive multiples of the simulation control period')
    if not np.isfinite(args.native_start_yaw_deg):
        parser.error('Native starting yaw must be finite')
    run(args)


if __name__ == '__main__':
    main()
