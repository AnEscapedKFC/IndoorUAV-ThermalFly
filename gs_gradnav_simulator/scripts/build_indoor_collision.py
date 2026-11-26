"""Create a collision bundle in a new directory using registered RGB training views."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene import GaussianScene
from scenes.room65_selected_anchored.room import NativeModel, intrinsic, tensor
from fit_indoor_frame import fit_frame, read_json, sha256

W, H = 320, 240
SURFACE_STEP, FREE_STEP = .04, .10
MIN_BASELINE, FREE_MARGIN = .25, .16


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def depth_maps(d, confidence, support, mask, z, std, alpha, scale, rays, center, planes, wall_hint):
    valid = mask & np.isfinite(d) & (d*scale > .15) & (d*scale < 22.)
    good = valid & (support >= 3) & (confidence >= .52)
    low = minimum_filter(np.where(good, d, 0), 3)
    high = maximum_filter(np.where(good, d, 0), 3)
    supported = good & (low > 0) & ((high-low)*scale < .13+.02*d*scale)
    agree = (supported & (alpha > .95) & (std*scale < .14+.012*d*scale)
             & (abs(z-d)*scale < .075+.012*d*scale))
    surface = np.where(agree, z, 0).astype('f4')
    best_err = np.full_like(d, np.inf)
    # A fitted wall is considered only in locally observed wall footprints.
    for i, (normal, level) in enumerate(planes):
        den = rays @ normal
        p = np.full_like(d, np.inf)
        np.divide(level-center@normal, den, out=p, where=np.abs(den)>.075)
        err = np.abs(p-d)*scale
        accept = (supported & (alpha>.94) & (p*scale>.15) & (p*scale<22.)
                  & (err<.08+.025*d*scale) & (np.abs(p-z)*scale<.22+.08*d*scale) & (err<best_err))
        if i >= 2:
            accept &= (wall_hint>0) & (np.abs(p-wall_hint)*scale < .14+.03*d*scale)
        surface[accept] = p[accept]
        best_err[accept] = err[accept]
    near = np.where(alpha>.95, np.minimum(low, np.maximum(0, z-2*std)), 0)
    weak = valid & (support>=2) & (confidence>=.25) & (alpha>.99)
    weak_ray = minimum_filter(np.where(weak, np.maximum(0, np.minimum(z-3*std, .93*d)), 0), 3)
    far = np.where(supported, near, np.minimum(weak_ray,15./scale))
    return surface, far.astype('f4')


def render_observations(scene, data, planar, rows, frame):
    scale, rotation = frame['meters_per_source_unit'], np.array(frame['rotation'])
    origin = np.array(frame['origin'])
    lower,upper = np.array(frame['bounds_m'])
    up = rotation[2]
    ceiling = frame['plane_fit']['ceiling']
    planes = [(up,frame['floor_src']), (np.array(ceiling['normal']),ceiling['projection_src'])]
    with np.load(planar/'walls.npz') as walls:
        if str(walls['manifest_sha256']) != frame['source_manifest_sha256']:
            raise ValueError('Wall prior and manifest differ')
        names = {str(n):i for i,n in enumerate(walls['names'])}
        wall_depth = walls['depth']
        planes.extend((p[:3],-p[3]) for p in walls['planes'])
    model = GaussianScene(scene/'models/rgb.ply',W,H)
    y,x = np.indices((H,W),dtype='f4')
    points, colors, views = [], [], []
    masks = {}
    for i,r in enumerate(rows):
        W2C = np.array(r['base_w2c'])
        C2W = np.linalg.inv(W2C)
        K = intrinsic(r,W,H)
        rays = np.stack(((x+.5-K[0,2])/K[0,0],(y+.5-K[1,2])/K[1,1],np.ones_like(x)),axis=-1) @ C2W[:3,:3].T
        target = r['targets']['lr']
        key = target.get('mask_path')
        if key not in masks:
            if key:
                image = cv2.imread(str(data/key),cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(data/key)
                masks[key] = cv2.resize(image,(W,H),interpolation=cv2.INTER_AREA)==255
            else:
                masks[key] = np.ones((H,W),bool)
        prior_path = data/r['dense_prior']['path']
        if sha256(prior_path) != r['dense_prior']['sha256']:
            raise ValueError(f'Changed depth prior: {prior_path}')
        z,std,alpha = NativeModel.sample(model,W2C,K,W,H,moments=True)
        hint = cv2.resize(wall_depth[names[r['source_name']]].astype('f4'),(W,H),interpolation=cv2.INTER_NEAREST)
        with np.load(prior_path) as p:
            solid,far = depth_maps(p['depth'],p['confidence'],p['support'],masks[key],z,std,alpha,
                                   scale,rays,C2W[:3,3],planes,hint)
        views.append((W2C,K,solid,far,C2W[:3,3]))
        yy,xx = np.nonzero(solid[::2,::2]); yy*=2;xx*=2
        if len(xx):
            xyz = ((rays[yy,xx]*solid[yy,xx,None]+C2W[:3,3]-origin) @ rotation.T)*scale
            inside = ((xyz>=lower)&(xyz<upper)).all(1)
            points.append(xyz[inside].astype('f4'))
            rgb = NativeModel.sample(model,W2C,K,W,H)[0]
            colors.append(np.uint8((255*rgb[yy[inside],xx[inside]]).round()))
        if i%100==0:
            print(scene.name,'depths',i+1,'/',len(rows),flush=True)
    if not points:
        raise ValueError('No corroborated surface observations')
    points, colors = np.concatenate(points), np.concatenate(colors)
    dims = np.ceil((upper-lower)/SURFACE_STEP).astype(int)
    keys = np.ravel_multi_index(np.floor((points-lower)/SURFACE_STEP).astype(int).T,dims)
    _,inv,counts = np.unique(keys,return_inverse=True,return_counts=True)
    def pool(values):
        return np.stack([np.bincount(inv,weights=c)/counts for c in values.T],1)
    return pool(points),pool(colors).round().astype('u1'),views


def count_votes(pts, views, scale):
    n = len(views)
    centers = np.array([v[-1] for v in views])
    separate = np.linalg.norm(centers[:,None]-centers,axis=-1)*scale >= MIN_BASELINE
    separate = torch.as_tensor(np.pad(separate,((0,0),(0,1)),constant_values=True),device=pts.device)
    hit = torch.full((len(pts),3),n,dtype=torch.int32,device=pts.device)
    free = hit.clone()
    def vote(bank, use, i):
        ids = torch.where(use & (bank[:,-1]==n))[0]
        prev = bank[ids].long()
        accept = separate[i,prev].all(1)
        bank[ids[accept],(prev[accept]!=n).sum(1)] = i
    for i,(w,K,solid,far,_) in enumerate(views):
        q = pts @ tensor(w[:3,:3]).T + tensor(w[:3,3])
        z = q[:,2];den=z.clamp_min(.001)
        x = (q[:,0]/den*K[0,0]+K[0,2]).floor().long()
        y = (q[:,1]/den*K[1,1]+K[1,2]).floor().long()
        inside = (z*scale>.14)&(x>=1)&(x<W-1)&(y>=1)&(y<H-1)
        x,y=x.clamp(0,W-1),y.clamp(0,H-1)
        obs=tensor(solid)[y,x]
        vote(hit,inside&(obs>0)&((obs-z).abs()*scale<.055),i)
        vote(free,inside&((tensor(far)[y,x]-z)*scale>FREE_MARGIN),i)
    return (hit!=n).sum(1).cpu().numpy(),(free[:,-1]!=n).cpu().numpy()


def triangulated_landmarks(manifest, rows, frame):
    """Use stable triangulated features to guard object edges and weak depth returns."""
    import pycolmap
    sfm = pycolmap.Reconstruction(manifest['preparation']['sfm_source'])
    rotation, origin = np.array(frame['rotation']), np.array(frame['origin'])
    scale = frame['meters_per_source_unit']
    lower, upper = np.array(frame['bounds_m'])
    poses = {r['source_name']: np.array(r['base_w2c']) for r in rows}
    centers = {}
    for image_id, image in sfm.images.items():
        w = poses.get(image.name)
        if w is not None:
            if not np.allclose(image.cam_from_world().matrix(), w[:3], atol=1e-5):
                raise ValueError('Triangulation and RGB poses disagree: ' + image.name)
            centers[image_id] = np.linalg.inv(w)[:3, 3]
    points, colors = [], []
    for pid, p in sfm.points3D.items():
        if pid % 5 == 0 or p.error > 1.0:
            continue  # reserve a reproducible subset for geometric cross-checks
        xyz = ((p.xyz-origin) @ rotation.T)*scale
        if np.any(xyz<lower) or np.any(xyz>=upper):
            continue
        cameras = np.array([centers[t.image_id] for t in p.track.elements if t.image_id in centers])
        if len(cameras) < 3:
            continue
        a = cameras[np.argmax(np.linalg.norm(cameras-cameras[0],axis=1))]
        da = np.linalg.norm(cameras-a,axis=1)
        b = cameras[np.argmax(da)]
        apart = np.minimum(da,np.linalg.norm(cameras-b,axis=1))
        ray_a,ray_b = a-p.xyz, b-p.xyz
        if (apart.max()*scale >= MIN_BASELINE and
                ray_a @ ray_b < np.cos(np.deg2rad(4))*np.linalg.norm(ray_a)*np.linalg.norm(ray_b)):
            points.append(xyz)
            colors.append(p.color)
    return np.asarray(points), np.asarray(colors)


def build(scene,planar,dest,height_m):
    if height_m<=0 or not np.isfinite(height_m):
        raise ValueError('Invalid floor-to-ceiling height')
    cfg = read_json(scene/'inputs/training_config.json')
    manifest = read_json(scene/'inputs/manifest.json')
    data = Path(cfg['source_path'])
    source = read_json(scene/'inputs/source.json')
    if sha256(scene/'models/rgb.ply') != source['gaussian_sha256']['rgb']:
        raise ValueError('RGB model checksum mismatch')
    rows = [r for r in manifest['views'] if r['modality']=='rgb' and r['split']=='train' and 'dense_prior' in r]
    frame = fit_frame(scene,planar,height_m)
    if sha256(data/'plus_manifest.json') != frame['source_manifest_sha256']:
        raise ValueError('Dataset and bundle manifests differ')
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'inputs').mkdir();(dest/'collision').mkdir()
    scale,R,origin = frame['meters_per_source_unit'], np.array(frame['rotation']), np.array(frame['origin'])
    lo,hi=np.array(frame['bounds_m'])
    surfaces, colors, views = render_observations(scene,data,planar,rows,frame)
    dims = np.rint((hi-lo)/FREE_STEP).astype(int)
    cells = lo + (np.indices(dims).reshape(3,-1).T+.5)*FREE_STEP
    coords = np.concatenate((surfaces,cells))
    hit,free = count_votes(tensor((coords/scale)@R+origin),views,scale)
    n = len(surfaces)
    ids = np.flatnonzero((hit[:n]>=3) & ~free[:n])
    neighbors = cKDTree(surfaces[ids]).query_ball_point(surfaces[ids],.11,return_length=True)
    ids = ids[neighbors>=5]
    if not len(ids): raise ValueError('No surfaces passed multi-view checks')
    landmarks, landmark_colors = triangulated_landmarks(manifest,rows,frame)
    # An isolated point may exclude free space, but only coherent groups add obstacles.
    groups = cKDTree(landmarks).query_ball_point(landmarks,.16,return_length=True) >= 4
    xyz = np.concatenate((surfaces[ids],landmarks[groups]))
    rgb = np.concatenate((colors[ids],landmark_colors[groups]))
    _, keep = np.unique(np.floor((xyz-lo)/SURFACE_STEP).astype('i4'),axis=0,return_index=True)
    xyz, rgb = xyz[keep], rgb[keep]
    vertex = np.empty(len(xyz),dtype=[(k,'f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')])
    for key,col in zip(vertex.dtype.names,[*xyz.T,*rgb.T]):
        vertex[key]=col
    PlyData([PlyElement.describe(vertex,'vertex')]).write(dest/'collision/rgb_surface_m.ply')
    free_cells = free[n:] & (hit[n:]<2)
    # A free voxel must lie inside the room and have no retained surface in it.
    free_cells &= (cells[:,2]>.5*FREE_STEP) & (cells[:,2]<height_m-.5*FREE_STEP)
    ceiling = frame['plane_fit']['ceiling']
    distance_below_ceiling = (ceiling['projection_src'] -
                              ((cells/scale)@R+origin) @ np.array(ceiling['normal']))*scale
    free_cells &= distance_below_ceiling > FREE_STEP*np.sqrt(3)/2
    candidates = np.flatnonzero(free_cells)
    guard = FREE_STEP*np.sqrt(3)/2 + SURFACE_STEP
    # Two viewpoints can veto flight space even if three do not certify a surface.
    guards = np.concatenate((surfaces[hit[:n]>=2],landmarks))
    guarded_distance = cKDTree(guards).query(cells[candidates])[0]
    free_cells[candidates[guarded_distance<=guard]] = False
    observed=free_cells.reshape(dims)
    np.savez_compressed(dest/'collision/free_volume.npz',free=observed,origin=lo,step=FREE_STEP)
    write(dest/'inputs/frame.json',frame)
    write(dest/'collision/build.json',dict(rgb_model='models/rgb.ply',rgb_sha256=source['gaussian_sha256']['rgb'],
          manifest_sha256=frame['source_manifest_sha256'],training_rgb_views=len(rows),
          depth_evidence='RGB Gaussian depth moments and translated-view-supported SfM-aligned monocular depth; floor/ceiling and local wall observations',
          candidate_voxels=n, depth_surface_points=len(ids), triangulated_guard_landmarks=len(landmarks),
          triangulated_clustered_landmarks=int(groups.sum()), free_cell_surface_guard_m=guard,
          free_veto_surface_views=2, strong_free_ray_gaussian_sigma_margin=2,
          fused_surface_points=len(xyz), free_cells=int(observed.sum()), total_cells=int(observed.size),
          sparse_voxel_m=SURFACE_STEP,free_voxel_m=FREE_STEP,distinct_views=3,required_view_baseline_m=MIN_BASELINE,
          free_ray_margin_m=FREE_MARGIN,meters_per_source_unit=scale,assumed_ceiling_height_m=height_m,
          unknown_policy='False or out-of-bounds is not certified free. Occupied surface voxels excluded.'))
    print(scene.name,'built',len(xyz),int(observed.sum()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene',type=Path)
    parser.add_argument('planar_scene',type=Path)
    parser.add_argument('output',type=Path,help='New directory for inputs/frame.json and collision assets')
    parser.add_argument('--height-m',type=float,default=2.80,help='Assumed or measured floor-to-ceiling height')
    args=parser.parse_args()
    build(args.scene.resolve(),args.planar_scene.resolve(),args.output.resolve(),args.height_m)
