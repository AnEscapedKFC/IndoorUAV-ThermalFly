"""Read-only, post-flight task metrics for the recorded simulator trajectories."""
import json

import numpy as np


def _rows(path):
    with path.open() as rows:
        return [json.loads(row) for row in rows]


def _kinematics(frames):
    positions = np.array([f['position_w_m'] for f in frames])
    speeds = np.linalg.norm([f['velocity_w_mps'] for f in frames], axis=1)
    peak = int(np.argmax(positions[:, 2]))
    return dict(peak_speed_mps=float(speeds.max()),
                height_gain_m=float(positions[peak,2]-positions[0,2]),
                descent_after_peak_m=float(positions[peak,2]-positions[-1,2]),
                path_m=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()))


def score(dataset, episode, output):
    # The task reference enters only here, after the closed loop has completed.
    ep = dataset / 'episodes' / episode
    meta = json.loads((ep/'episode.json').read_text())
    task = meta['task_spec']
    flown = _rows(output/'frames.jsonl')
    expert = _rows(ep/'frames.jsonl')
    result = dict(recorded_reference=_kinematics(expert), flown=_kinematics(flown))
    stages = [s for s in task['ordered_subgoals'] if s.get('predicate')=='approach_and_observe']
    if not stages:
        return result

    scene = dataset/'scenes'/meta['scene_version']
    cal = json.loads((scene/'calibration.json').read_text())
    from scripts.audit_room_expert_semantics import projection
    landmarks = {}
    for filename in ['landmarks.json','object_landmarks.json']:
        if (scene/filename).exists():
            landmarks.update(json.loads((scene/filename).read_text()))
    wh = np.array(cal['wh'])
    refpos = np.array([f['position_w_m'] for f in expert])
    times = np.array([f['t_s'] for f in expert])
    positions = np.array([f['position_w_m'] for f in flown])
    meta_path=output/'summary.json'
    flown_wh=np.array(json.loads(meta_path.read_text()).get('render_wh',wh)) if meta_path.exists() else wh
    inspections = []
    cursor = 0
    for stage in stages:
        obj = stage['landmark']
        center = np.array(landmarks[obj]['center_w_m'])
        uv,_,vis = projection(expert,center,cal,'rgb')
        central_ref = vis & np.all((uv>=0.1*wh)&(uv<=0.9*wh),axis=1)
        i = task['ordered_subgoals'].index(stage)
        end = task['ordered_subgoals'][i+1]['at_s'] if i+1 < len(task['ordered_subgoals']) else times[-1]
        window = central_ref & (times>=stage['at_s']) & (times<=end)
        bound = float(np.linalg.norm(refpos[window]-center,axis=1).min()+.2) if window.any() else None
        seen = []
        hits = []
        for j,frame in enumerate(flown):
            v = frame['landmark_projections'][obj]
            point = v['uv']
            if v['in_image'] and np.all(np.asarray(point)>=0.1*flown_wh) and np.all(np.asarray(point)<=.9*flown_wh):
                dist = float(np.linalg.norm(positions[j]-center))
                seen.append(dist)
                if j>=cursor and bound is not None and dist<=bound:
                    hits.append(j)
        index = hits[0] if hits else None
        inspections.append(dict(target=obj,reference_range_plus_20cm_m=bound,
            closest_centered_distance_m=min(seen) if seen else None,
            first_ordered_hit_s=flown[index]['t_s'] if index is not None else None))
        cursor = (index+1) if index is not None else len(flown)
    result['ordered_inspections'] = inspections
    result['all_geometry_inspections_passed'] = all(i['first_ordered_hit_s'] is not None for i in inspections)
    return result


def add_evidence(dataset, episode, output):
    summary_file = output/'summary.json'
    s = json.loads(summary_file.read_text())
    s['task_evidence'] = score(dataset,episode,output)
    s.setdefault('strict_terminal_success', s['strict_navigation_success'])
    if s['mode']=='model' and 'all_geometry_inspections_passed' in s['task_evidence']:
        s['strict_navigation_success'] = bool(s['strict_terminal_success'] and s['task_evidence']['all_geometry_inspections_passed'])
    temp = summary_file.with_suffix('.tmp')
    temp.write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    temp.replace(summary_file)
    return s
