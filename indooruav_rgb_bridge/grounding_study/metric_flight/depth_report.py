"""Summarize paired high-resolution dynamics flights after independent scoring."""
import sys
from collections import Counter
from pathlib import Path
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from metric_flight.furniture import flight_tasks, deny_expert_reads, HERE


def read(path):
    return json.loads(path.read_text())


def group(rows):
    return dict(n=len(rows),strict=sum(r['strict'] for r in rows),
                approximate=sum(r['approximate'] for r in rows),
                collisions=sum(r['collision_events'] for r in rows),
                outside=sum(r['outside_samples']>0 for r in rows))


def main():
    deny_expert_reads()
    base=HERE/'results/furniture_640'
    tasks=flight_tasks()
    reference=read(base/'rgb/reference.json')
    result={}
    summaries={}
    for mode in ('rgb','depth'):
        results=read(base/mode/'metrics.json')
        rows={}
        for name,task in tasks.items():
            folder=base/mode/name
            e=read(folder/'episode.json')
            frames=read(folder/'frames.json')
            ds=read(folder/'decisions.json');m=results[name]
            first=json.loads((folder/'video_frames.jsonl').open().readline())
            rec=read(folder/'recording.json')
            assert e['render_wh']==rec['rgb_wh']==rec['thermal_wh']==[640,480]
            assert np.allclose(frames[0]['xyz'],task['initial']['position_w_m'],atol=1e-6)
            assert np.allclose(frames[0]['v'],task['initial']['velocity_w_mps'],atol=1e-6)
            assert e['instruction']==task['instruction'] and e['seed']==42
            geom=np.array(reference[task['target']])
            guard=Counter(d['info']['depth_guard']['choice'] for d in ds if 'depth_guard' in d['info'])
            stop=next((d for d in ds if d['stopping']),None)
            last_anchor=None; errors=[]
            for d in ds:
                info=d['info'];p=frames[round(20*d['t'])]['xyz']
                if info.get('anchor') and info['anchor']['confirmed']:
                    last_anchor=info['anchor']['world']
                if 'range_est_m' in info and np.linalg.norm(geom-p)<=1.5:
                    errors.append(abs(info['range_est_m']-np.linalg.norm(geom-p)))
            stop_s=stop['t'] if stop else None
            measured=estimated=None
            if stop:
                p=np.array(frames[round(20*stop_s)]['xyz'])
                measured=float(np.linalg.norm(geom-p))
                estimated=stop['info'].get('range_est_m')
                if estimated is None and last_anchor is not None:
                    estimated=float(np.linalg.norm(p-last_anchor))
            survey=read(folder/'depth_preflight.json') if mode=='depth' else {}
            rows[name]=dict(case=name,reason=m['reason'],end_m=m['end_range_m'],strict=m['strict'],
                            approximate=m['approximate'],collision_events=m['collision_events'],
                            outside_samples=m['outside_samples'],stable=m['stable_hover'],
                            coverage=m.get('sweep',{}).get('coverage'),ordered_times=m.get('sweep',{}).get('left_middle_right_times_s'),
                            guard=guard,estimated_at_stop_m=estimated,reference_at_stop_m=measured,
                            near_range_mae_m=float(np.mean(errors)) if errors else None,near_decisions=len(errors),
                            survey_seconds=survey.get('duration_s',0),mission_seconds=m['duration_s'],
                            first_rgb=first['rgb_sha256'],first_thermal=first['thermal_sha256'],
                            first_rgb_w2c=first['rgb_w2c'],first_thermal_w2c=first['thermal_w2c'])
        result[mode]=rows
        summaries[mode]={part:group([r for n,r in rows.items() if ('_h3' in n)==extra])
                         for part,extra in [('original',False),('additional',True)]}
    for n in tasks:
        for key in ('first_rgb','first_thermal','first_rgb_w2c','first_thermal_w2c'):
            assert result['rgb'][n][key]==result['depth'][n][key],(n,key)
    output=dict(groups=summaries,episodes=result,
                sim_time_includes_preflight=False,matching_first_frames_and_camera_poses=True,
                external_depth='RGB 3DGS expected optical Z in meters',backbone='frozen RGB IndoorUAV-Agent 29999')
    (base/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
