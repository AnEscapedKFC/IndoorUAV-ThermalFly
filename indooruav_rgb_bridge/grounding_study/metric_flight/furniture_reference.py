"""Independent scoring reference from calibrated reconstruction photographs only."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw

from sim_bridge import SIM
from scenes.room65_selected_anchored.room import NativeModel, intrinsic

SCENE = SIM/'scenes/room65_selected_anchored/inputs'
PHOTO = Path('/home/zr_fan/ThermalGaussian_Surface/data/room_vid65_selected/rgb/c00_0000026.png')
# Image coordinates on the 640 x 480 calibration photograph. Trace the exposed
# upholstery and arms. The white textile, table and chairs mask real surfaces.
PARTS = {
 'backrest': [(140,174),(138,141),(146,111),(158,100),(511,125),(526,132),(530,145),
              (526,177),(519,208),(462,219),(436,218),(423,197),(406,186),(396,183),
              (389,194),(369,194),(345,181),(335,182),(323,196),(321,208),(308,216),(182,207)],
 'left_arm': [(68,214),(109,190),(137,178),(177,183),(180,205),(140,240),(121,242),(72,225)],
 'right_arm': [(455,229),(473,216),(510,211),(526,220),(545,253),(551,278),(539,283),
               (467,288),(460,282)],
 'seat': [(143,241),(182,212),(307,220),(312,231),(331,246),(356,254),(397,260),
          (425,259),(444,252),(452,241),(458,277),(468,295),(387,291),(386,275)],
 'front_panel': [(389,298),(475,304),(467,357),(375,348)],
 'right_side': [(483,295),(544,287),(541,315),(528,351),(512,371),(476,362)],
}
FRONT_CORNERS = np.array([[269,114],[313,135],[245,154],[289,180]]) / 2
BACK_CORNERS = np.array([[483,214],[299,210],[456,289],[299,288]]) / 2


def world_from_native(x, cfg):
    return (x-np.array(cfg['origin'])) @ np.array(cfg['rotation']).T * cfg['meters_per_source_unit']


def reference(output):
    cfg = json.loads((SCENE/'frame.json').read_text())
    rows = {v['sequence']:v for v in json.loads((SCENE/'manifest.json').read_text())['views'] if v['modality']=='rgb'}
    def camera(n):
        return np.array(rows[n]['base_w2c']), intrinsic(rows[n],320,240)
    W,K=camera(26)
    if hashlib.sha256(PHOTO.read_bytes()).hexdigest()!=rows[26]['targets']['lr']['sha256']:
        raise ValueError('Reference photo differs from the calibrated scene input')
    photo=Image.open(PHOTO).convert('RGB').resize((320,240),Image.Resampling.LANCZOS)
    annotated=photo.copy()
    idx=Image.new('L',photo.size); draw=ImageDraw.Draw(idx)
    for i, (name, vertices) in enumerate(PARTS.items(),1):
        poly=[(round(x/2),round(y/2)) for x,y in vertices]
        draw.polygon(poly,fill=i)
        ImageDraw.Draw(annotated).polygon(poly,outline='#ffcc30')
    annotated.save(output/'sofa_reference.png')
    model=NativeModel('rgb',320,240)
    _, z, alpha=model.sample(W,K,320,240)
    y,x=np.indices(z.shape)
    rays=np.stack((x,y,np.ones_like(x)),axis=-1)@np.linalg.inv(K).T
    native=(rays*z[...,None]-W[:3,3])@W[:3,:3]
    labels=np.array(idx)
    # Unsupported labelled pixels stay in the coverage denominator.
    supported=(alpha>=.85)&np.isfinite(z)&(z>0)
    # The point just above the midpoint of the sofa seat (away from the cloth).
    sofa=world_from_native(np.median(native[85:90,150:160],axis=(0,1)),cfg)
    systems=[camera(288),camera(314)]
    corners=[]; residual=[]
    for u,v in zip(FRONT_CORNERS,BACK_CORNERS):
        A=[]; b=[]
        for (ww,kk),uv in zip(systems,(u,v)):
            d=np.linalg.solve(kk,[*uv,1.])@ww[:3,:3]
            d=d/np.linalg.norm(d);H=np.eye(3)-np.outer(d,d)
            A.append(H); b.append(H@np.linalg.inv(ww)[:3,3])
        loc=np.linalg.solve(sum(A),sum(b));corners.append(loc)
        for (ww,kk),uv in zip(systems,(u,v)):
            pix=kk@(ww[:3,:3]@loc+ww[:3,3])
            residual.append(np.linalg.norm(pix[:2]/pix[2]-uv))
    if max(residual)>4.:raise ValueError(f'Laptop annotations disagree: {residual}')
    corners=world_from_native(np.array(corners),cfg)
    laptop=corners.mean(axis=0)
    normal=np.cross(corners[1]-corners[0],corners[2]-corners[0]);normal/=np.linalg.norm(normal)
    camera288=world_from_native(np.linalg.inv(systems[0][0])[:3,3],cfg)
    if (camera288-laptop)@normal<0: normal=-normal
    ref=dict(sofa=sofa.tolist(),laptop=laptop.tolist(),screen_normal=normal.tolist(),
             laptop_max_reprojection_px=float(max(residual)),sofa_sequence=26,laptop_sequences=[288,314],
             scene_scale_m_per_unit=cfg['meters_per_source_unit'],camera_width=320,camera_height=240,
             reference_depth_supported_fraction=float(supported[labels>0].mean()))
    (output/'reference.json').write_text(json.dumps(ref,indent=2)+'\n')
    return ref,dict(native=native,labels=labels,supported=supported,photo=np.array(photo),k=camera(62)[1],model=model,scale=cfg['meters_per_source_unit'])
