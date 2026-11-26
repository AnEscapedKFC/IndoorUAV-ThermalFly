"""Train on room MOVE decisions only. Run with the simulator's thermal_gaussian_5070 environment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
from scipy.linalg import solve
from scipy.spatial.transform import Rotation

from policy import DEFAULT_WEIGHTS, ENCODER_NAME, PROMPT, RGBEncoder, kernel, language


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def train(dataset, output, encoder_path, regularization):
    from scenes.room65_selected_anchored.room import NativeModel
    splits = read(dataset/'splits.json')['splits']
    k = np.array(read(dataset/'scenes/room65_selected_anchored/calibration.json')['rgb_K'])
    raster = NativeModel('rgb', 320, 240)
    encoder = RGBEncoder(encoder_path)
    features, word_vectors, angles, targets = [], [], [], []
    rates, pitch_range, episodes = [], [], {}
    for name in splits['train']:
        folder = dataset/'episodes'/name
        meta = read(folder/'episode.json')
        if meta['scene_version'] != 'room65_selected_anchored':
            continue
        frames = rows(folder/'frames.jsonl')
        pitch = np.array([f['rig_pitch_deg'] for f in frames])
        times = np.array([f['t_s'] for f in frames])
        rates.append(float(np.max(np.abs(np.diff(pitch)/np.diff(times)))))
        pitch_range.extend((pitch.min(), pitch.max()))
        stages = {s['id']: s.get('landmark') for s in meta['task_spec']['ordered_subgoals']}
        episodes[name] = 0
        for decision in rows(folder/'decisions.jsonl'):
            teacher = decision['teacher']
            if teacher['chunk'][0]['kind'] != 'move':
                continue
            idx = decision['observed_frame_id']
            frame = frames[idx]
            actual = pitch[idx]
            target = np.interp(times[idx]+1., times, pitch)
            # Real 3DGS views at alternative gimbal settings teach recovery when the camera is off angle.
            choices = np.unique([actual, np.clip(actual-18., -75., 35.), np.clip(actual+18., -75., 35.), pitch[0]])
            w2c = np.array(frame['rgb_w2c_native'])
            images = []
            for angle in choices:
                if abs(angle-actual) < 1e-6:
                    images.append(np.array(Image.open(folder/frame['rgb_path']).convert('RGB')))
                else:
                    delta = np.eye(4)
                    delta[:3,:3] = Rotation.from_euler('x',actual-angle,degrees=True).as_matrix()
                    rgb, _, _ = raster.sample(delta@w2c, k, 320, 240)
                    images.append(np.rint(255*rgb).astype(np.uint8))
            encoded = encoder.encode(images)
            prompt = meta['instruction']
            texts = [language(prompt)]
            landmark = stages.get(teacher['subgoal_id'])
            if landmark in PROMPT:
                focused = language(PROMPT[landmark])
                if not np.array_equal(focused, texts[0]):
                    texts.append(focused)
            for text in texts:
                for v, a in zip(encoded, choices):
                    features.append(v)
                    word_vectors.append(text)
                    angles.append(a)
                    targets.append(target)
            episodes[name] += 1
        print(name, episodes[name], flush=True)
    visual = np.array(features)
    words = np.array(word_vectors)
    angles = np.array(angles)
    gram = kernel(visual, words, angles, visual, words, angles).astype(np.float64)
    gram.flat[::len(gram)+1] += regularization
    alpha = solve(gram, targets, assume_a='pos')
    output.parent.mkdir(parents=True,exist_ok=True)
    dest = output.parent/ENCODER_NAME
    if dest.resolve() != encoder_path.resolve():
        shutil.copyfile(encoder_path, dest)
    np.savez_compressed(output, visual=visual, words=words, angles=angles, alpha=alpha,
                        bounds=[min(pitch_range),max(pitch_range)], max_rate=max(rates))
    report = dict(episodes=episodes, move_decisions=sum(episodes.values()), fitted_rows=len(angles),
                  horizon_s=1.,rate_deg_per_s=max(rates),pitch_range_deg=[min(pitch_range),max(pitch_range)],
                  ridge=regularization,encoder_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                  weights_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  label='measured extra gimbal pitch at t+1s interpolated from frames',
                  input='RGB, current pitch, words from issued instruction; focused noun augmentation only on train')
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim',type=Path,default=Path(__file__).resolve().parents[2]/'gs_gradnav_simulator')
    parser.add_argument('--output',type=Path,default=DEFAULT_WEIGHTS)
    parser.add_argument('--encoder',type=Path,default=Path.home()/'.cache/torch/hub/checkpoints'/ENCODER_NAME)
    args=parser.parse_args()
    sys.path.insert(0,str(args.sim.resolve()))
    train(args.sim/'outputs/expert_rgbt_20261123',args.output,args.encoder,0.1)
