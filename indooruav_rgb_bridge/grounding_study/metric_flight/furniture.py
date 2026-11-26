"""Build whitelisted initial conditions or fly using the resulting manifest."""
import argparse
import fcntl
import json
import os
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
EXPERT = STUDY.parents[1] / 'gs_gradnav_simulator/outputs/expert_rgbt_20261123'
MANIFEST = HERE / 'furniture_tasks.json'
RESULTS = HERE / 'results/furniture'
INITIAL = ('position_w_m', 'q_w_from_b_wxyz', 'velocity_w_mps', 'rig_pitch_deg')
DESIGN = {
    'laptop': ('laptop', ('room_inspect_laptop', 'room_inspect_laptop_north_high')),
    'sofa': ('sofa', ('room_inspect_sofa', 'room_inspect_sofa_west_high')),
    'scan': ('sofa', ('room_obj_sofa_east_to_west', 'room_obj_sofa_west_to_east')),
}
SCAN_TEXT = ('Approach the black sofa from the table side. At close range, look across its front '
             'from left to right, then face the sofa and hover for two seconds.')


def prepare():
    tasks = {}
    for kind, (noun, sources) in DESIGN.items():
        for i, source in enumerate(sources):
            folder = EXPERT/'episodes'/source
            with (folder/'episode.json').open() as f:
                instruction = json.load(f)['instruction']
            assert noun in instruction.lower()
            with (folder/'frames.jsonl').open() as f:
                first = json.loads(next(f))
            # The simulator's public reset accepts yaw and velocity. Verify these
            # particular source attitudes can be represented exactly by that reset.
            qw,qx,qy,qz=first['q_w_from_b_wxyz']
            if abs(qx)+abs(qy)>1e-8 or not math.isclose(qw*qw+qz*qz,1.,abs_tol=1e-6):
                raise ValueError(f'First-frame attitude cannot be reproduced: {source}')
            tasks[f'{kind}_{i+1}'] = dict(instruction=SCAN_TEXT if kind=='scan' else instruction,
                                          target=noun, initial={k:first[k] for k in INITIAL})
    MANIFEST.write_text(json.dumps(tasks, indent=2)+'\n')


def deny_expert_reads():
    root = EXPERT.resolve()
    def check(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            if Path(os.fsdecode(args[0])).resolve().is_relative_to(root):
                raise PermissionError('Expert recordings disabled during flight and scoring')
    sys.addaudithook(check)
    try:
        open(root/'episodes/room_inspect_laptop/frames.jsonl').close()
    except PermissionError:
        return
    raise AssertionError('Expert isolation probe failed')


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args(argv)
    if args.prepare:
        prepare()
        return
    deny_expert_reads()
    from metric_flight.experiment import execute
    from detector import Detector
    detector = Detector(use_cache=False)
    tasks = json.loads(MANIFEST.read_text())
    for name in args.cases or tasks:
        task = tasks[name]
        if set(task) != {'instruction','target','initial'} or set(task['initial']) != set(INITIAL):
            raise ValueError('unexpected task input')
        folder = RESULTS/name
        folder.mkdir(parents=True, exist_ok=True)
        with (folder/'.flight.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            execute(name, 'auto', True, detector, record_rgbt='furniture', furniture_task=task)
