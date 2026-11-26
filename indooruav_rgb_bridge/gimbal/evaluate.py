"""Score paired camera and navigation runs; the ground truth is accessed after flight."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from policy import DEFAULT_WEIGHTS, PitchPolicy

BRIDGE = Path(__file__).resolve().parents[1]
SIM = BRIDGE.parent/'gs_gradnav_simulator'
DATA = SIM/'outputs/expert_rgbt_20261123'
RUNS = BRIDGE/'runs/gimbal'
OUT = BRIDGE/'results/gimbal'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def offline(episodes):
    policy = PitchPolicy(DEFAULT_WEIGHTS)
    result = {}
    samples = []
    for ep in episodes:
        folder = DATA/'episodes'/ep
        meta = read(folder/'episode.json')
        frames = rows(folder/'frames.jsonl')
        times = [r['t_s'] for r in frames]
        pitch = [r['rig_pitch_deg'] for r in frames]
        errors = []
        for d in rows(folder/'decisions.jsonl'):
            if d['teacher']['chunk'][0]['kind'] != 'move':
                continue
            i = d['observed_frame_id']
            angle = pitch[i]
            target = float(np.interp(times[i]+1.,times,pitch))
            rgb = np.array(Image.open(folder/frames[i]['rgb_path']).convert('RGB'))
            pred = policy.target(rgb,meta['instruction'],angle)
            applied = angle + np.clip(pred-angle,-policy.max_rate,policy.max_rate)
            errors.append(np.abs(np.array([pitch[0],angle,pred,applied])-target))
            samples.append(dict(episode=ep,frame=i,pitch=angle,label=target,prediction=pred,rate_limited=applied))
        result[ep] = dict(n=len(errors),mae_deg=dict(zip(['spawn','current','predicted','rate_limited'],np.mean(errors,axis=0))))
    (OUT/'offline_decisions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in samples))
    return result


def replay_pair(ep):
    folder = DATA/'episodes'/ep
    meta = read(folder/'episode.json')
    wh = np.array(meta['observation_config']['rgb_wh'])
    expert = rows(folder/'frames.jsonl')
    stages = meta['task_spec']['ordered_subgoals']
    result = {}
    for cam in ['fixed','learned']:
        run = RUNS/f'replay_{cam}'/ep
        frames = rows(run/'frames.jsonl')
        summary = read(run/'summary.json')
        assert len(frames)==len(expert) and summary['video_verification']['passed']
        physical_error = max(np.linalg.norm(np.array(f['position_w_m'])-e['position_w_m']) for f,e in zip(frames,expert))
        attitude_error = max(np.max(np.abs(np.array(f['q_w_from_b_wxyz'])-e['q_w_from_b_wxyz'])) for f,e in zip(frames,expert))
        assert physical_error<1e-8 and attitude_error<1e-8
        error = [abs(f['pitch_deg']-e['rig_pitch_deg']) for f,e in zip(frames,expert)]
        counts = {}
        starts = [s['at_s'] for s in stages]
        for f in frames:
            stage = stages[max(0,np.searchsorted(starts,f['t_s']+1e-7, side='right')-1)]
            target = stage.get('landmark')
            if target is None:
                continue
            group = counts.setdefault(stage['id'],dict(n=0,inside=0,central=0))
            group['n']+=1
            v=f['landmark_projections'][target]
            if v['in_image']:
                group['inside']+=1
                uv=np.array(v['uv'])
                group['central']+=int(np.all(uv>=0.1*wh) and np.all(uv<=.9*wh))
        result[cam] = dict(frames=len(frames),pitch_mae_deg=float(np.mean(error)),max_position_difference_m=physical_error,
            max_quaternion_component_difference=attitude_error,stage_visibility=counts,
            ordered_geometry=summary['task_evidence'].get('all_geometry_inspections_passed'),
            inspections=summary['task_evidence'].get('ordered_inspections',[]))
    return result


def flight_pair(ep, root=RUNS):
    result = {}
    folders = [root/f'model_{cam}'/ep for cam in ['fixed','auto']]
    for cam,folder in zip(['fixed','auto'],folders):
        s = read(folder/'summary.json')
        assert s['video_verification']['passed'] and s['initial_image_mae_u8']==0
        frames = rows(folder/'frames.jsonl')
        result[cam]={k:s[k] for k in ['gimbal','outcome','duration_s','terminal_position_error_m','stable_stop','strict_navigation_success','path_m']}
        result[cam].update(ordered_geometry=s['task_evidence'].get('all_geometry_inspections_passed'),
            observed_pitch_range_deg=[min(f['pitch_deg'] for f in frames),max(f['pitch_deg'] for f in frames)])
    first=[rows(folder/'decisions.jsonl')[0] for folder in folders]
    assert np.array_equal(first[0]['native_chunk'],first[1]['native_chunk'])
    result['same_first_native_chunk']=True
    if result['auto']['gimbal']=='fixed':
        assert (folders[0]/'frames.jsonl').read_bytes()==(folders[1]/'frames.jsonl').read_bytes()
        result['all_frames_identical']=True
    return result


if __name__ == '__main__':
    splits = read(DATA/'splits.json')['splits']
    rooms=[ep for group in ['val','test'] for ep in splits[group] if ep.startswith('room_')]
    all_flights=[*rooms,'corridor_high_return','corridor_climb_accelerate']
    OUT.mkdir(parents=True, exist_ok=True)
    offline_scores=offline(rooms)
    (OUT/'offline.json').write_text(json.dumps(offline_scores,indent=2)+'\n')
    if sys.argv[1:]==['--offline-only']:
        sys.exit(0)
    else:
        from hashlib import sha256
        sys.path.insert(0,str(BRIDGE))
        from audit_results import audit_inputs
        train=read(DEFAULT_WEIGHTS.with_suffix('.json'))
        assert not set(train['episodes']).intersection(all_flights)
        assert train['weights_sha256']==sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest()
        report=dict(training_decisions=train['move_decisions'],training_rows=train['fitted_rows'],
            evaluation_episodes={k:[ep for ep in ids if ep in all_flights] for k,ids in splits.items() if k!='train'},
            train_overlaps_evaluation=False,offline=offline_scores,
            replay={ep:replay_pair(ep) for ep in rooms},model={ep:flight_pair(ep) for ep in all_flights},
            additional_seeds={seed:{ep:flight_pair(ep,RUNS/f'seed_{seed}') for ep in ['room_inspect_ac','room_search_laptop_sofa']}
                              for seed in [16,77]},
            protected_inputs=audit_inputs())
        (OUT/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
