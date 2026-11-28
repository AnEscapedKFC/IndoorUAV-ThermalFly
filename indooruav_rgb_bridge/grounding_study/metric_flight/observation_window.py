"""Short-lived, image- and depth-verified observations of one RGB instance."""
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from metric_flight.depth_guidance import camera


def hold_seconds(instruction):
    match = re.search(r"(?:hover|hold)[^.]*?\b(?:for\s+)?(one|two|\d+)\s+seconds?\b", instruction.lower())
    if not match:
        return 1.
    word = match.group(1)
    return float({"one": 1, "two": 2}.get(word, word))


def gray(rgb):
    return cv2.cvtColor(np.rint(255 * rgb).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def world_at(uv, depth, rig):
    x, y = np.rint(uv).astype(int).T
    z = depth[y.clip(0, depth.shape[0]-1), x.clip(0, depth.shape[1]-1)]
    c, axes, k = camera(rig)
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(k).T
    return c + (z[:, None] * rays) @ axes.T


class HoldTracker:
    """Fit a fresh box from matched foreground points against the qualified frame."""
    def __init__(self, rgb, depth, rig, box, mask):
        self.gray = gray(rgb)
        self.shape = np.array([rig.width, rig.height])
        l,t,r,b = np.rint(np.asarray(box) * np.tile(self.shape, 2)).astype(int)
        support = np.zeros(depth.shape, np.uint8)
        support[t:b,l:r] = mask[t:b,l:r] & np.isfinite(depth[t:b,l:r]) & (depth[t:b,l:r]>.08)
        pts = cv2.goodFeaturesToTrack(self.gray, 240, .004, 4, mask=support)
        self.points = None if pts is None else pts.reshape(-1,2)
        self.box = np.asarray([[l,t],[r,t],[r,b],[l,b]], np.float32)
        if self.points is not None:
            self.world = world_at(self.points, depth, rig)

    def measure(self, rgb, depth, rig):
        if self.points is None or len(self.points)<8:
            return None
        current = gray(rgb)
        uv, ok, _ = cv2.calcOpticalFlowPyrLK(self.gray, current, self.points, None)
        if uv is None:
            return None
        back, backwards, _ = cv2.calcOpticalFlowPyrLK(current, self.gray, uv, None)
        if back is None:
            return None
        good = (ok.ravel()>0) & (backwards.ravel()>0) & (np.linalg.norm(back-self.points,axis=1)<1.5)
        good &= ((uv>=0)&(uv<self.shape-1)).all(axis=1)
        current_world = world_at(uv,depth,rig)
        good &= np.linalg.norm(current_world-self.world,axis=1)<.11
        if good.sum()<max(8,.45*len(self.points)):
            return None
        transform,inliers = cv2.estimateAffinePartial2D(self.points[good],uv[good],method=cv2.RANSAC,
                                                        ransacReprojThreshold=2.5)
        if transform is None or inliers is None or inliers.sum()<max(8,.65*good.sum()):
            return None
        span = np.ptp(self.points[good][inliers.ravel()>0],axis=0)
        if (span < .25*np.ptp(self.box,axis=0)).any():
            return None
        corners = self.box @ transform[:,:2].T + transform[:,2]
        return (np.r_[corners.min(0),corners.max(0)] / np.tile(self.shape,2)).tolist()


def box_iou(a, b):
    l,t = np.maximum(a[:2],b[:2])
    r,bottom = np.minimum(a[2:],b[2:])
    overlap=max(0,r-l)*max(0,bottom-t)
    return overlap/max(1.e-8, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)


class DoorChair:
    """Bind the chair to a visible doorway once, then associate live RGB-D instances."""
    def __init__(self):
        self.anchor = None

    def detect(self, detector, path, rig, depth):
        result = detector(path,"armchair")
        def location(box):
            l,t,r,b = np.asarray(box["xyxy"]) * [rig.width,rig.height,rig.width,rig.height]
            u,v = np.meshgrid(np.linspace(l+.28*(r-l),r-.28*(r-l),9),
                              np.linspace(t+.26*(b-t),b-.26*(b-t),9))
            uv = np.c_[u.ravel(),v.ravel()]
            points = world_at(uv,depth,rig)
            x,y = np.rint(uv).astype(int).T
            valid = np.isfinite(points).all(1) & (depth[y.clip(0,rig.height-1),x.clip(0,rig.width-1)]>.08)
            return np.median(points[valid],axis=0) if valid.sum()>=35 else None
        reference=self.anchor
        acquiring=reference is None
        if acquiring:
            doors=detector(path,"door")["boxes"]
            reference=location(doors[0]) if doors else None
        matches=[]
        for box in result["boxes"]:
            center=location(box)
            if center is not None and reference is not None:
                distance=np.linalg.norm((center-reference)[:2]) if acquiring else np.linalg.norm(center-reference)
                matches.append((float(distance),box,center))
        matches.sort(key=lambda row:row[0])
        # Reject ambiguous first associations and jumps to a different physical chair.
        matched = bool(matches and matches[0][0] <= (3.0 if acquiring else .70)
                       and (len(matches)==1 or matches[1][0]-matches[0][0]>.12 or
                            box_iou(matches[0][1]["xyxy"],matches[1][1]["xyxy"])>.65))
        if matched:
            _,box,center=matches[0]
            self.anchor=center.copy() if acquiring else .75*self.anchor+.25*center
        return dict(result,boxes=[box] if matched else [],side=box["side"] if matched else None,
                    association="door" if acquiring else "history", matched=matched,
                    candidate_distances=[m[0] for m in matches])


def write_sample(folder, step, rgb, depth, rig):
    name=f"{step:04d}"
    base=Path(folder)/"review_frames"
    base.mkdir(exist_ok=True)
    Image.fromarray(np.rint(255*rgb).astype(np.uint8)).save(base/(name+".png"),compress_level=1)
    c, axes, k = camera(rig)
    z=np.where(np.isfinite(depth)&(depth>0),depth,0)
    np.savez_compressed(base/(name+".npz"), depth_mm=np.rint(1000*np.clip(z,0,65)).astype(np.uint16),
                        center=c,axes=axes,K=k)
