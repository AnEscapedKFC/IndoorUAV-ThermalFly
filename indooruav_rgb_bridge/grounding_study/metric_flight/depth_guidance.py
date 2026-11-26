"""Live RGB-D measurements in metric world coordinates for furniture observations."""
from collections import deque

import cv2
import numpy as np


def camera(rig):
    # Columns are the optical right, down and forward axes in the flight frame.
    rotation=rig.R @ rig.W[:3,:3].T
    center=rig.s * (rig.R @ (-rig.W[:3,:3].T @ rig.W[:3,3]-rig.origin))
    return center, rotation, rig.ks['rgb']


def backrest_point(rgb,z,box,c,R,K):
    """Recover the upholstered front plane and its physical horizontal midpoint."""
    h,w=z.shape
    x0,y0,x1,y1=box
    ys=np.linspace(y0+.07*(y1-y0),y0+.58*(y1-y0),32)
    xs=np.linspace(x0+.04*(x1-x0),x1-.04*(x1-x0),65)
    u,v=np.meshgrid(np.clip((xs*w).astype(int),0,w-1),np.clip((ys*h).astype(int),0,h-1))
    depth=z[v,u]
    rays=np.stack([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones_like(u)],-1)
    world=(rays*depth[...,None])@R.T+c
    normal=np.cross(np.gradient(world,axis=0),np.gradient(world,axis=1))
    norm=np.linalg.norm(normal,axis=-1)
    luminance=rgb[v,u].mean(axis=-1)
    valid=(np.isfinite(depth)&(depth>.15)&(depth<7.)&(norm>1e-6)
           &(abs(normal[...,2])<.65*norm)&(luminance<=np.percentile(luminance,75)))
    pts=world[valid]
    if len(pts)<60:return None
    normals=normal[valid][::max(1,len(pts)//64),:2]
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
    seeds=pts[::max(1,len(pts)//64)]
    support=abs(pts[:,:2]@normals.T-(seeds[:,:2]*normals).sum(axis=1))<.065
    view=c[:2]-seeds[:,:2];view/=np.maximum(np.linalg.norm(view,axis=1,keepdims=True),1e-8)
    votes=support.sum(axis=0)*(abs((normals*view).sum(axis=1))>.45)
    j=int(votes.argmax())
    if votes[j]<max(45,.30*len(pts)):return None
    surface=pts[support[:,j]]
    axis=np.array([-normals[j,1],normals[j,0]])
    sides=surface[:,:2]@axis
    low,high=np.percentile(sides,[2,98])
    if high-low<.55:return None
    midpoint=axis*(low+high)/2+normals[j]*np.median(surface[:,:2]@normals[j])
    return np.r_[midpoint,np.percentile(surface[:,2],30)]


class ObservationAnchor:
    """Estimate a sofa backrest/seat junction or a laptop screen midpoint."""
    def __init__(self,noun):
        self.noun=noun
        self.points=deque(maxlen=16)
        self.last_camera=None
        self.position=None
        self.error=np.inf
        self.missed=0

    def update(self, rgb, z, box, rig):
        h,w=z.shape
        x0,y0,x1,y1=box
        c,R,K=camera(rig)
        cropped=x0<=.025 or x1>=.975 or y0<=.015
        fresh=False
        source='unavailable'
        if not cropped:
            # Find the horizontal upholstery seam or display bezel from the
            # currently detected instance. Use several columns to reject local objects.
            gray=cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2GRAY)
            l,r=np.clip(np.rint(w*np.array([.65*x0+.35*x1,.35*x0+.65*x1])).astype(int),0,w-1)
            top,bottom=np.clip(np.rint(h*np.array([y0,y1])).astype(int),0,h-1)
            height=bottom-top
            if r-l>=5 and height>=8:
                gradient=np.abs(cv2.Sobel(gray[:,l:r+1],cv2.CV_32F,0,1,ksize=3))
                profile=np.median(gradient,axis=1)
                if self.noun=='sofa':
                    rows=np.arange(top+max(2,round(.18*height)),top+max(3,round(.55*height)))
                    row=rows[np.argmax(profile[rows]*(1.-.55*np.abs((rows-top)/height-.37)))]
                    source='backrest_lower_edge'
                else:
                    rows=np.arange(top+max(2,round(.42*height)),top+max(3,round(.80*height)))
                    bezel=rows[np.argmax(profile[rows])]
                    row=(top+bezel)/2
                    source='screen_midpoint'
                uv=np.array([(l+r)/2,row])
                patch=3 if self.noun=='laptop' else 5
                x,y=np.rint(uv).astype(int)
                region=z[max(0,y-patch):y+patch+1,max(0,x-patch):x+patch+1]
                values=region[np.isfinite(region)&(region>.12)&(region<7.)]
                if values.size>=.75*region.size:
                    d=np.median(values)
                    spread=np.median(abs(values-d))
                    point=c+R@(np.linalg.solve(K,[*uv,1.])*d)
                    if self.noun=='sofa':
                        surface_point=backrest_point(rgb,z,box,c,R,K)
                        if surface_point is not None:
                            point=surface_point
                            source='backrest_plane_midpoint'
                    if spread<.085 and (self.position is None or np.linalg.norm(point-self.position)<.50):
                        if self.last_camera is None or np.linalg.norm(c-self.last_camera)>.055:
                            self.points.append(point)
                            self.last_camera=c.copy()
                        self.position=np.median(self.points,axis=0)
                        scatter=np.linalg.norm(np.asarray(self.points)-self.position,axis=1)
                        self.error=float(.045+np.percentile(scatter,80)+spread)
                        fresh=True
        if self.position is None:
            self.missed+=1
            return None
        opt=R.T@(self.position-c)
        if opt[2]<=.10:
            self.missed+=1
            return None
        pix=K@opt; uv=pix[:2]/pix[2]/[w,h]
        valid=bool(x0-.025<=uv[0]<=x1+.025 and y0-.025<=uv[1]<=y1+.025 and
                   np.all(uv>=0) and np.all(uv<1))
        if valid:
            x,y=np.minimum(np.rint(uv*[w,h]).astype(int),[w-1,h-1])
            observed=z[max(0,y-2):y+3,max(0,x-2):x+3]
            residual=abs(float(np.median(observed))-opt[2])
            valid=np.isfinite(residual) and residual<max(.14,2*self.error)
        self.missed=0 if valid else self.missed+1
        if not fresh:source='reprojected'
        return dict(world=self.position.tolist(),uv=uv.tolist(),error_m=self.error,
                    views=len(self.points),source=source,confirmed=bool(valid),cropped=bool(cropped))
