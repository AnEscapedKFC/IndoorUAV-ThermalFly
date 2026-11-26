"""Isolate noun and image bearing effects at fixed IndoorUAV state, image and RNG."""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

from PIL import Image, ImageDraw, ImageOps
import numpy as np

OUT = Path(__file__).resolve().parent
BRIDGE = OUT.parents[1]
sys.path.insert(0, str(BRIDGE))
from offline_eval import DATA, projection, read_json, rows
from bridge_client import exchange, DEFAULT_SOCKET

VIEWS = {
    'laptop_left_sofa_right': ('room_search_laptop_sofa', 177, ['laptop','sofa']),
    'sofa_left_laptop_right': ('room_orbit_low', 603, ['laptop','sofa']),
    'ac_left': ('room_search_full_survey', 196, ['ac']),
    'ac_right': ('room_search_full_survey', 360, ['ac']),
}
OBJECTS = {'laptop':'Turn in place to face the open laptop.',
           'sofa':'Turn in place to face the black sofa.',
           'ac':'Turn in place to face the wall-mounted air conditioner.'}
LEXICAL = {'left':'Turn in place to the left.','right':'Turn in place to the right.'}
SEEDS = [16,42,77]
STATE = [0., 0., 1.5, 0.]
DEADBAND_DEG = 2.
NEUTRAL = DATA/'episodes/room_search_laptop_sofa/rgb/000000.png'


def prepare():
    room = DATA/'scenes/room65_selected_anchored'
    calibration = read_json(room/'calibration.json')
    landmarks = read_json(room/'landmarks.json')
    labels = Image.new('RGB',(640,550),'white')
    views = {}
    for i,(name,(episode,frame,keys)) in enumerate(VIEWS.items()):
        folder = DATA/'episodes'/episode
        row = rows(folder/'frames.jsonl')[frame]
        original = folder/row['rgb_path']
        mirror = OUT/(name+'_mirror.png')
        with Image.open(original) as im:
            ImageOps.mirror(im).save(mirror)
            label = im.convert('RGB')
        pts = projection(row,calibration,{k:landmarks[k] for k in keys})
        pixels = {k: pts[k]['pixel'] for k in keys}
        assert all(v['center_in_frame'] for v in pts.values())
        draw = ImageDraw.Draw(label)
        for k,(x,y) in pixels.items():
            draw.ellipse((x-4,y-4,x+4,y+4),outline='#ff3232',width=2)
            draw.text((x+5,y-9),k,fill='red',stroke_width=1,stroke_fill='white')
        x, y = (i%2)*320, (i//2)*275
        labels.paste(label,(x,y+28))
        ImageDraw.Draw(labels).text((x+4,y+6),name+'  '+str(frame),fill='black')
        views[name] = {'file':str(original),'mirror':str(mirror),'episode':episode,
                       'frame_id':frame,'native_camera_calibration':str(room/'calibration.json'),
                       'pixels':pixels,'actual_position':row['position_w_m'],'gimbal_deg':row['rig_pitch_deg']}
    labels.save(OUT/'views.jpg',quality=92)
    return views


def run(yaw_gauge=0., primary_only=False, results_dir=OUT):
    results_dir.mkdir(parents=True, exist_ok=True)
    state = [*STATE[:3], math.radians(yaw_gauge)]
    modes = ['fixed_reference'] if primary_only else ['fixed_reference','stage_start','mirrored_stage_start']
    views=prepare()
    records=[]
    for name, v in views.items():
        for ref_mode in modes:
            mirror=ref_mode=='mirrored_stage_start'
            current=v['mirror'] if mirror else v['file']
            reference=str(NEUTRAL) if ref_mode=='fixed_reference' else current
            target_keys = list(v['pixels'])+(list(LEXICAL) if ref_mode=='fixed_reference' else [])
            for goal,seed in itertools.product(target_keys,SEEDS):
                text=(OBJECTS|LEXICAL)[goal]
                if goal in LEXICAL:
                    side=1 if goal=='right' else -1
                    x=None
                else:
                    x=v['pixels'][goal][0]
                    if mirror: x=319-x
                    side=1 if x>160 else -1
                req={'id':f'visual-{yaw_gauge:g}-{name}-{ref_mode}-{goal}-{seed}', 'instruction':text,'rgb_path':current,
                     'ref_rgb_path':reference,'state':state,'seed':seed}
                reply=exchange(DEFAULT_SOCKET,req)
                wps=np.array(reply['waypoints'])
                assert req['id']==reply['id'] and wps.shape==(10,4) and np.isfinite(wps).all()
                turn=math.degrees((wps[-1,3]-state[3]+math.pi)%math.tau-math.pi)
                magnitude=float(np.linalg.norm(wps[-1,:3]-state[:3]))
                records.append({'view':name,'mode':ref_mode,'goal':goal,'target_x':x,'side':side,
                                'yaw_right_deg':turn,'endpoint_translation_m':magnitude,
                                'sign_ok':bool(turn*side>0),'ok_above_deadband':bool(turn*side>=DEADBAND_DEG),
                                'request':req,'response':reply})
    (results_dir/'raw.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    summary={'seeds':SEEDS,'state':state,'two_degree_deadband':DEADBAND_DEG,'views':views,'cells':{},'groups':{},'target_ordering':{}}
    for name in views:
        for mode in modes:
            selected=[r for r in records if r['view']==name and r['mode']==mode]
            for goal in sorted({r['goal'] for r in selected}):
                rs=[r for r in selected if r['goal']==goal]
                summary['cells'][f'{name}/{mode}/{goal}']={'target_x':rs[0]['target_x'],'yaw_deg_each_seed':[round(r['yaw_right_deg'],3) for r in rs],
                    'median_translation_m':round(statistics.median(r['endpoint_translation_m'] for r in rs),3)}
    for mode in modes:
        for group,goals in [('objects',OBJECTS),('explicit_directions',LEXICAL),('laptop_sofa',['laptop','sofa']),('ac',['ac'])]:
            rs=[r for r in records if r['mode']==mode and r['goal'] in goals]
            if rs:summary['groups'][mode+'/'+group]={'n':len(rs),'correct_sign':sum(r['sign_ok'] for r in rs),
                    'correct_above_2deg':sum(r['ok_above_deadband'] for r in rs),'median_abs_yaw_deg':statistics.median(abs(r['yaw_right_deg']) for r in rs),
                    'median_translation_m':statistics.median(r['endpoint_translation_m'] for r in rs)}
        pairs=[]
        for name in list(views)[:2]:
            for seed in SEEDS:
                rs=[r for r in records if r['mode']==mode and r['view']==name and r['request']['seed']==seed and r['goal'] in ['laptop','sofa']]
                pos=next(r for r in rs if r['side']==1)
                neg=next(r for r in rs if r['side']==-1)
                pairs.append({'view':name,'seed':seed,'yaw_for_right_target_minus_left_target_deg':pos['yaw_right_deg']-neg['yaw_right_deg'],
                              'both_in_requested_direction_above_2deg':pos['ok_above_deadband'] and neg['ok_above_deadband']})
        summary['target_ordering'][mode]=pairs
    (results_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['groups'],indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaw-gauge', type=float, default=0.)
    parser.add_argument('--primary-only', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args=parser.parse_args()
    run(args.yaw_gauge, args.primary_only, args.output_dir)
