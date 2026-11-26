"""Pair RGB camera variants under the original seeds, instructions, state and poses."""
from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess

out = Path(__file__).resolve().parent
bridge = out.parents[1]
manifest = json.loads((out/'manifest.json').read_text())
paths = {(x['case'], x['view']): x['render'] for x in manifest}
records = [json.loads(l) for l in (out.parent/'image/raw.jsonl').read_text().splitlines()]
original = [r for r in records if r['mode']=='fixed_reference' and r['goal'] in ('laptop','sofa','ac')]
assert len(original)==18
scores = {}
with (out/'worker.log').open('w') as log, (out/'raw.jsonl').open('w') as raw, subprocess.Popen([bridge/'scripts/model_worker.sh'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True) as worker:
    try:
        for case in ['native','wide43','wide169','wide169_hd']:
            cell = []
            for row in original:
                req = dict(row['request'],id=f"{case}_{row['request']['id']}",rgb_path=paths[case,row['view']],ref_rgb_path=paths[case,'reference'])
                worker.stdin.write(json.dumps(req)+'\n')
                worker.stdin.flush()
                ans = json.loads(worker.stdout.readline())
                assert ans['id']==req['id']
                turn = math.degrees((ans['waypoints'][-1][3]-req['state'][3]+math.pi)%math.tau-math.pi)
                signed = turn*row['side']
                rec = dict(case=case,view=row['view'],goal=row['goal'],side=row['side'],yaw_deg=turn,matched_original_deg=row['yaw_right_deg'],correct=signed>0,correct_over_2=signed>=2,request=req,response=ans)
                raw.write(json.dumps(rec)+'\n')
                raw.flush()
                cell.append(rec)
            pairs = defaultdict(list)
            for r in cell:
                pairs[(r['view'],r['goal'])].append(r['correct'])
            scores[case] = dict(n=len(cell),correct=sum(r['correct'] for r in cell),above_two_degrees=sum(r['correct_over_2'] for r in cell),majority_cells=sum(sum(x)>=2 for x in pairs.values()),by_target={k:[sum(r['correct'] for r in cell if r['goal']==k),sum(r['goal']==k for r in cell)] for k in ('ac','laptop','sofa')},max_yaw_delta_from_original_deg=max(abs(r['yaw_deg']-r['matched_original_deg']) for r in cell))
            (out/'summary.json').write_text(json.dumps(dict(fixed_state=[0,0,1.5,0],seeds=[16,42,77],physical_views=4,object_view_pairs=6,conditions=scores),indent=2)+'\n')
            print(case,scores[case],flush=True)
    finally:
        worker.stdin.close()
        try: worker.wait(timeout=20)
        except subprocess.TimeoutExpired: worker.terminate()
