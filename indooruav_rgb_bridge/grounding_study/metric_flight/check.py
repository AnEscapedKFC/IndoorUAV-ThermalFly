from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from PIL import Image
from evaluate import OUT,HERE,STUDY,load,write
sys.path.insert(0,str(STUDY));sys.path.insert(1,str(STUDY.parent))
from metric_flight.control import quantities,gate,Odometer,metric_waypoint
from audit_results import audit_inputs

def main():
    m=load(OUT/'metrics.json');checked=[]
    sentences=[('向前飞行13.5米',(None,13.5)),('以1.82米高度向前飞行13.5米',(1.82,13.5)),('Fly forward 13.5 meters.',(None,13.5)),
               ('Fly at a height of 1.7 m.',(1.7,None)),('Climb slightly, then hover.',(None,None)),
               ('Inspect the unit and hover for two seconds.',(None,None)),('Fly at 2 m from the sofa.',(None,None)),
               ('Fly 2 m above the floor.',(2.,None))]
    for text,expected in sentences:
        assert quantities(text)==expected and not gate(text,'off') and gate(text,'auto')==(expected!=(None,None))
    pos=np.array([.3,.1,1.5]);raw=np.array([.7,.3,4.,23.]);od=Odometer(pos,0)
    t=metric_waypoint('Fly at a height of 1.7 m.','auto',raw,pos,0,od,1.)
    assert t[2]==1.7 and np.array_equal(t[[0,1,3]],raw[[0,1,3]])
    assert np.array_equal(raw,metric_waypoint('Fly forward 13.5 meters.','off',raw,pos,0,od,1.))
    for scene,arms,count in [('room',('hint','visual'),6),('corridor',('off','auto'),5)]:
        assert set(m[scene][arms[0]])==set(m[scene][arms[1]]) and len(m[scene][arms[0]])==count
        for case in m[scene][arms[0]]:
            folders=[]
            for a in arms:
                p=OUT/scene/a/case
                if not (p/'episode.json').exists():
                    assert (scene,a)==('room','hint')
                    p=STUDY/'results/runs/flight/hint'/case
                e=load(p/'episode.json');f=load(p/'frames.json');ds=load(p/'decisions.json')
                assert e['mode']=='flight' and len(f)>1 and all(r['t']<s['t'] for r,s in zip(f,f[1:]))
                assert np.allclose(f[-1]['xyz'],e['end']) and ds
                if scene=='room':
                    assert not gate(e['instruction'],'auto') and f[0]['objects']['ac']['in_image']
                    if a=='visual':assert all(not x['gate'] for x in ds)
                else:
                    assert all(d['gate']==(a=='auto') for d in ds)
                    assert e['altitude_setpoint'] is None and e['distance_setpoint']==13.5
                    if e['reason']=='distance_reached':assert f[-2]['cumulative_m']<13.5 and f[-1]['cumulative_m']>=13.5
                folders.append(p)
            np.testing.assert_array_equal(np.array(Image.open(folders[0]/'000.png')),np.array(Image.open(folders[1]/'000.png')))
            for key in ['spawn','spawn_yaw','instruction']:
                assert load(folders[0]/'episode.json')[key]==load(folders[1]/'episode.json')[key]
            if scene=='corridor':
                d0,d1=[load(p/'decisions.json')[0] for p in folders]
                assert d0['raw_world']==d1['raw_world'] and d0['prompt']==d1['prompt']
            checked.append(case)
    for ln in (OUT/'locked.sha256').read_text().splitlines():
        h,src=ln.split();assert hashlib.sha256((OUT/'locked_source'/Path(src).name).read_bytes()).hexdigest()==h
    write(OUT/'verification.json',dict(paired_starts=checked,metric_parser_cases=len(sentences),start_rgb_and_instructions_identical=True,physical_only=True,
               old_room_baselines_reused=4,new_physical_flights_in_final_comparison=18,protected_files=audit_inputs()))
    print('Verified 22 physical records, including four matched previous flights.')
if __name__=='__main__':main()
