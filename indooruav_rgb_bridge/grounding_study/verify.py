"""Verify arm pairing, unchanged inputs, held-out separation, and reuse of the historical baseline."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
from PIL import Image
from score import ROOT, OUT, DATA, get, save
sys.path.insert(0,str(ROOT.parent))
from audit_results import audit_inputs
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    scores=get(OUT/'metrics.json')
    details=scores['episodes'];assert len(details)==96
    all_starts={p.parent.name:json.loads(p.open().readline())['position_w_m'] for p in (DATA/'episodes').glob('*/frames.jsonl')}
    prior=get(ROOT.parent/'results/direct_render_vln/direct_pose.json')['episodes']
    cases={};baseline_checks={};groups={}
    for m in ('direct','flight'):
        for sub in (OUT,OUT/'reverse'):
            for fold in (sub/'runs'/m/'baseline').iterdir():
                nm=fold.name;meta=get(fold/'episode.json');src=meta['source'];cohort='reverse' if sub!=OUT else 'new' if meta['new'] else 'old'
                cases[(cohort,nm)]=(meta['spawn'],meta['spawn_yaw'])
                first_image=np.asarray(Image.open(fold/'000.png'));starter=get(fold/'decisions.json')[0]
                for arm in ['hint','guard','both']:
                    other=fold.parent.parent/arm/nm
                    np.testing.assert_array_equal(first_image,np.asarray(Image.open(other/'000.png')))
                    entry=get(other/'decisions.json')[0]
                    assert entry['state']==starter['state']
                    if arm=='guard':assert entry['raw']==starter['raw']
                if src.startswith('corridor'):
                    for one,two in [('baseline','hint'),('guard','both')]:
                        left=fold.parent.parent/one/nm;right=fold.parent.parent/two/nm
                        assert get(left/'frames.json')==get(right/'frames.json')
                        assert get(left/'decisions.json')==get(right/'decisions.json')
                if cohort=='old':
                    if m=='flight':
                        history=get(ROOT.parent/'runs/gimbal/model_fixed'/src/'summary.json')
                        err=math.dist(history['end_w_m'],meta['end'])
                        assert meta['num_decisions']==history['decisions'] and abs(meta['duration']-history['duration_s'])<1e-6
                    else:
                        history=prior[src];err=math.dist(history['steps'][-1]['reached_position'],meta['end'])
                        assert meta['num_decisions']==len(history['steps'])
                    assert err<1e-6
                    baseline_checks[m+'/'+src]=err
    new_info={}
    for (cohort,nm),(p,y) in cases.items():
        if cohort=='old':continue
        nearest=min((math.dist(p,xyz),case) for case,xyz in all_starts.items())
        assert nearest[0]>.05
        new_info[cohort+'/'+nm]=dict(spawn=p,yaw=y,nearest_recorded_start=nearest[1],nearest_recorded_distance_m=nearest[0])
    for file in ('frozen.sha256','reverse_frozen.sha256'):
        for row in (OUT/file).read_text().splitlines():
            ref,name=row.split(maxsplit=1)
            archive='frozen_run.py' if name=='run.py' else name
            assert sha(ROOT/archive)==ref
    assert sha(ROOT/'assets/model.safetensors')=='1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3'
    for d in details:
        assert d['strict'] is None if d['mode']=='direct' else type(d['strict']) is bool
    static_views=get(ROOT.parent/'diagnoses/image/summary.json')['views']
    for name,view in static_views.items():
        np.testing.assert_array_equal(np.asarray(Image.open(view['file'])),np.asarray(Image.open(OUT/'azimuth_views'/f'{name}_0.png')))
    report=dict(zero_rotation_images_equal_original=True,baseline_end_discrepancy_m=baseline_checks,new_starts=new_info,
              all_96_episodes_complete=True,all_arm_start_pixels_and_states_paired=True,unchanged_corridor_hint_negative_control=True,
              detector_repository='IDEA-Research/grounding-dino-tiny',detector_commit='a2bb814dd30d776dcf7e30523b00659f4f141c71',
              detector_sha256=sha(ROOT/'assets/model.safetensors'),simulator_and_expert_files=audit_inputs())
    save(OUT/'verification.json',report)
    print('Verified 96 episodes and 12 unchanged historical endpoints; max deviation',max(baseline_checks.values()))
if __name__=='__main__':main()
