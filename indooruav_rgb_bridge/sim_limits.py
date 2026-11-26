"""One-second motion bounds estimated from the existing training episodes only."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def bounded_goal(state, target, limits, seconds):
    start = np.r_[state['position'], state['euler_deg'][2]]
    delta = np.asarray(target, dtype=float) - start
    horizontal = np.linalg.norm(delta[:2])
    vertical = abs(delta[2])
    scale = min(1., limits['horizontal_m_per_s']*seconds/max(horizontal,1.e-12),
                limits['vertical_m_per_s']*seconds/max(vertical,1.e-12))
    delta[:3] *= scale
    yaw = (delta[3]+180.)%360.-180.
    max_yaw = limits['yaw_deg_per_s'] * seconds
    delta[3] = np.clip(yaw,-max_yaw,max_yaw)
    return start+delta, dict(translation_scale=float(scale), yaw_clipped=bool(abs(yaw)>max_yaw))


def limits_from_train(dataset):
    ids = json.loads((dataset/'splits.json').read_text())['splits']['train']
    stats = {}
    for episode in ids:
        ep = dataset/'episodes'/episode
        meta = json.loads((ep/'episode.json').read_text())
        with (ep/'frames.jsonl').open() as f:
            allrows = [json.loads(line) for line in f]
        stop = meta['waypoint_spec']['stop_frame_id']
        rows = allrows[:stop+1]
        times = np.array([r['t_s'] for r in rows])
        if len(rows) < 21 or not np.allclose(times[20:]-times[:-20],1.):
            raise ValueError(f'Incomplete or irregular one-second windows in {episode}')
        positions=np.array([r['position_w_m'] for r in rows])
        q = np.array([r['q_w_from_b_wxyz'] for r in rows])
        headings = Rotation.from_quat(q[:,[1,2,3,0]]).as_euler('xyz',degrees=True)[:,2]
        delta = positions[20:]-positions[:-20]
        dyaw = (headings[20:]-headings[:-20]+180.)%360.-180.
        values = np.column_stack((np.linalg.norm(delta[:,:2],axis=1), abs(delta[:,2]), abs(dyaw)))
        scene=stats.setdefault(meta['scene_version'],dict(episodes=[],samples=0,maxima=np.zeros(3)))
        scene['episodes'].append(episode)
        scene['samples'] += len(delta)
        scene['maxima'] = np.maximum(scene['maxima'],values.max(axis=0))
    pooled = np.maximum.reduce([scene['maxima'] for scene in stats.values()])
    for scene in stats.values():
        maxima = scene['maxima']
        scene['limits']=dict(horizontal_m_per_s=float(maxima[0]),
                             vertical_m_per_s=float(pooled[1]), yaw_deg_per_s=float(pooled[2]))
        scene['maxima'] = maxima.tolist()
    return dict(source_split='train',n_episodes=len(ids),one_second_windows='Every frame through first STOP; actual positions and wrapped yaw 20 control steps apart.',
        statistic='Maximum observed one-second horizontal displacement within each scene. Shared maximum vertical and yaw changes across both scenes keep climb/turn possible in the corridor whose training routes rarely exercise them.',
        policy='Uniform translation shortening preserves the model-selected 3D direction. Wrapped yaw change is bounded separately; native outputs and small-motion stopping are unchanged.',
        columns=['horizontal_m','absolute_vertical_m','absolute_yaw_deg'], scenes=stats)


def validate():
    state={'position':[2.,-1.,1.5],'euler_deg':[0.,0.,179.]}
    limits=dict(horizontal_m_per_s=.6,vertical_m_per_s=.2,yaw_deg_per_s=20.)
    goal,info = bounded_goal(state,[5.,3.,2.5,-170.],limits,1.)
    assert np.allclose(goal,[2.36,-.52,1.62,190.])
    assert math.isclose(info['translation_scale'],.12) and not info['yaw_clipped']
    tiny,info=bounded_goal(state,[2.,-1.,1.55,-179.],limits,.5)
    assert np.allclose(tiny,[2.,-1.,1.55,181.]) and info['translation_scale']==1.
    vertical,info=bounded_goal(state,[2.,-1.,-1.,100.],limits,1.)
    assert np.allclose(vertical,[2.,-1.,1.3,159.]) and info['yaw_clipped']


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    validate()
    artifact=limits_from_train(args.dataset)
    artifact['validation']='Passed planar shortening at fixed inclination, vertical limiting, identity for short commands, wrapped yaw clipping across ±180 degrees.'
    args.output.write_text(json.dumps(artifact,indent=2)+'\n')
