"""Re-render four scored room poses and their reference, changing only the RGB intrinsics."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

out = Path(__file__).resolve().parent
bridge = out.parents[1]
sim = bridge.parent / 'gs_gradnav_simulator'
# Use the installed build of the simulator's CUDA rasterizer on this workstation.
sys.path.insert(0, str(Path.home() / '.cache/torch_extensions/py310_cu128/gsplat_cuda'))
import gsplat_cuda
sys.modules['gsplat.csrc'] = gsplat_cuda
sys.path.insert(0, str(sim / 'scenes/room65_selected_anchored'))
from room import NativeModel

dataset = sim / 'outputs/expert_rgbt_20261123'
prior = json.loads((out.parent / 'image/summary.json').read_text())
views = {k: (x['episode'], x['frame_id']) for k, x in prior['views'].items()}
views['reference'] = ('room_search_laptop_sofa', 0)
f_local = json.loads((dataset/'scenes/room65_selected_anchored/calibration.json').read_text())['rgb_K'][0][0]
model = NativeModel('rgb', 320, 240)
manifest = []
for name, (episode, frame) in views.items():
    folder = dataset / 'episodes' / episode
    row = json.loads((folder/'frames.jsonl').read_text().splitlines()[frame])
    old = np.array(Image.open(folder/row['rgb_path']).convert('RGB'))
    for case, w, h, f in [('native',320,240,f_local),('wide43',320,240,160.),('wide169',320,180,160.),('wide169_hd',1280,720,640.)]:
        K = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]])
        rgb, _, alpha = model.sample(np.array(row['rgb_w2c_native']), K, w, h)
        image = np.rint(255*rgb).astype('uint8')
        dest = out / f'{case}_{name}.png'
        Image.fromarray(image).save(dest)
        mae = float(abs(image.astype(float)-old).mean()) if case=='native' else None
        print(name,case,mae,flush=True)
        if case=='native' and mae > 0.05:
            raise ValueError(f'Native rerender diverged: {mae}')
        manifest.append(dict(view=name,case=case,render=str(dest),episode=episode,frame=frame,size=[w,h],K=K.tolist(),baseline_mae=mae,under_half_alpha=float((alpha<.5).mean())))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
