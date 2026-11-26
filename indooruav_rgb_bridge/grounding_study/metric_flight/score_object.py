"""Score target-relative physical flights without consulting a demonstration endpoint."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from metric_flight.evaluate import DATA, OUT, load, write, max_span
from metric_flight.object_goal import MIN_RANGE_M, MAX_RANGE_M, RESULT_FOLDER, OBJECT_INSTRUCTION, centered

# These are evaluation bands. The instruction and the controller's stop gate
# remain frozen at the historical setting in object_goal.py.
STRICT_RANGE_M = (0.4, 0.8)
APPROX_RANGE_M = (0.5, 1.5)


def distance_mask(distance, band):
    return (band[0] <= distance) & (distance <= band[1])


def score(folder):
    episode = load(folder / 'episode.json')
    frames = load(folder / 'frames.json')
    decisions = load(folder / 'decisions.json')
    assert episode['mode'] == 'flight'
    if episode.get('room_goal') == 'standoff':
        if (episode.get('object_distance_range_m') != [MIN_RANGE_M, MAX_RANGE_M]
                or not episode['instruction'].endswith(OBJECT_INSTRUCTION)):
            raise ValueError(f'Flight uses a different object goal: {folder}')
    task = load(DATA / 'episodes' / episode['source'] / 'episode.json')
    stages = task['task_spec']['ordered_subgoals']
    target = next(s['landmark'] for s in reversed(stages) if 'landmark' in s)
    landmarks = {}
    for name in ('landmarks.json', 'object_landmarks.json'):
        path = DATA / 'scenes' / task['scene_version'] / name
        if path.exists():
            landmarks.update(load(path))
    center = landmarks[target]['center_w_m']
    wh = np.array(task['observation_config']['rgb_wh'])
    xyz = np.array([f['xyz'] for f in frames])
    distance = np.linalg.norm(xyz - center, axis=1)
    in_range = distance_mask(distance, STRICT_RANGE_M)
    approx = distance_mask(distance, APPROX_RANGE_M)
    observations = [f['objects'][target] for f in frames]
    in_center = np.array([o['in_image'] and centered(*(np.array(o['uv']) / wh)) for o in observations])
    times = np.array([f['t'] for f in frames])
    stop = next((d for d in decisions if d['stopping']), None)
    hold = episode['hold']
    stable = bool(stop and episode['reason'] == 'stop' and len(hold) >= 41
                  and times[-1] - stop['t'] >= 2 - 1e-7
                  and max(np.linalg.norm(np.array(s['xyz']) - hold[0]['xyz']) for s in hold) <= .12
                  and max(s['speed'] for s in hold[-10:]) <= .1)
    mask = times >= stop['t'] - 1e-7 if stop else np.zeros(len(times), dtype=bool)
    hold_goal = bool(stable and np.all((in_range & in_center)[mask]))
    collision = any(f['collision'] for f in frames)
    outside = any(not f['inside'] for f in frames)
    pre = times <= stop['t'] + 1e-7 if stop else np.ones(len(times), dtype=bool)
    az = np.rad2deg(np.unwrap(np.arctan2(xyz[:, 1] - center[1], xyz[:, 0] - center[0])))
    arc = max_span(pre & in_center, az)
    arc_required = 'along an arc' in task['instruction'].lower()
    phase_arc = None
    if arc_required and episode.get('inspection'):
        arc_start = next((d['t'] for d in decisions if d['info'].get('phase') == 'arc'), None)
        if arc_start is None:
            phase_arc = 0.
        else:
            finish = next((d['t'] for d in decisions if d['t'] > arc_start and
                           d['info'].get('phase') in {'settle', 'terminal_hold', 'terminal_approach'}), times[-1])
            interval = (times >= arc_start - 1e-7) & (times <= finish + 1e-7)
            phase_arc = max_span(pre & interval & in_center, az)
    climb_required = 'climb slightly' in task['instruction'].lower()
    climb_gain = float(xyz[pre, 2].max() - xyz[0, 2])
    observed_arc = phase_arc if phase_arc is not None else arc
    semantics = (not arc_required or observed_arc >= 20.) and (not climb_required or climb_gain >= .05)
    common_pass = bool(stable and semantics and not collision and not outside)
    geometry_success = bool(common_pass and hold_goal)
    approximate_success = bool(common_pass and approx[mask].all())
    stop_info = stop['info'] if stop else {}
    rgb_samples = episode.get('hold_rgb', [])
    stop_uv = None
    if 'cx' in stop_info:
        stop_uv = [stop_info['cx'], stop_info['cy']]
    elif stop and stop.get('detection') and stop['detection']['boxes']:
        box = stop['detection']['boxes'][0]['xyxy']
        stop_uv = [(box[0]+box[2])/2, (box[1]+box[3])/2]
    has_rgb_checks = episode.get('room_goal') == 'standoff' and 'hold_rgb' in episode
    rgb_hold = bool(stop and stop_uv and centered(*stop_uv) and len(rgb_samples) == 4
                    and np.allclose([row['t'] - stop['t'] for row in rgb_samples], [.5, 1., 1.5, 2.])
                    and all(row['uv'] is not None and centered(*row['uv']) for row in rgb_samples))
    task_success = bool(geometry_success and rgb_hold) if has_rgb_checks else None
    return dict(case=episode['case'], target=target, reason=episode['reason'],
                strict_range_m=STRICT_RANGE_M, approximate_range_m=APPROX_RANGE_M,
                strict_success=task_success, approximate_success=approximate_success,
                end_in_approximate_band=bool(approx[-1]),
                end_distance_m=float(distance[-1]), end_in_band=bool(in_range[-1]),
                min_distance_m=float(distance.min()), any_under_min=bool((distance < STRICT_RANGE_M[0]).any()),
                end_uv=observations[-1]['uv'], end_centered=bool(in_center[-1]),
                terminal_goal=bool(in_range[-1] and in_center[-1]),
                hold_distance_min_m=float(distance[mask].min()) if mask.any() else None,
                hold_distance_max_m=float(distance[mask].max()) if mask.any() else None,
                stable=stable, stable_hold_in_band_and_center=hold_goal,
                stop_estimate_m=stop_info.get('range_est_m'), stop_rgb_center=stop_uv,
                hold_rgb_centered=rgb_hold if has_rgb_checks else None,
                centered_arc_deg=arc if arc_required else None, arc_phase_centered_deg=phase_arc,
                height_gain_m=climb_gain if climb_required else None,
                instruction_motion_pass=bool(semantics), collision=collision, outside=outside,
                geometry_task_success=geometry_success, task_success=task_success,
                rgb_confirmed_success=task_success)


def summarize():
    base = OUT / RESULT_FOLDER
    results = {}
    for policy in ('visual', 'hint'):
        current = {}
        paired = {}
        for episode in sorted((base / policy).glob('*/episode.json')):
            folder = episode.parent
            current[folder.name] = score(folder)
            write(folder / 'two_tier_metrics.json', current[folder.name])
            previous = OUT / 'room' / policy / folder.name
            if not (previous / 'episode.json').exists() and policy == 'hint':
                previous = HERE.parent / 'results/runs/flight/hint' / folder.name
            if (previous / 'episode.json').exists():
                with Image.open(folder / '000.png') as a, Image.open(previous / '000.png') as b:
                    assert np.array_equal(np.asarray(a), np.asarray(b))
                paired[folder.name] = score(previous)
        if current:
            results[policy] = dict(standoff=current, previous_rescored=paired)
    if results:
        write(base / 'metrics_two_tier.json', results)
    return results


if __name__ == '__main__':
    for policy, groups in summarize().items():
        for group, cases in groups.items():
            metric = 'geometry_task_success' if group == 'previous_rescored' else 'task_success'
            print(policy, group, metric, sum(r[metric] for r in cases.values()), '/', len(cases))
