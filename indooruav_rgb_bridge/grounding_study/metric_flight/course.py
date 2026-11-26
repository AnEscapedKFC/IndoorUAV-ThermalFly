"""Horizontal corridor bearing from the vanishing direction of observed RGB lines."""
import cv2
import numpy as np

class VisualCourse:
    def __init__(self,axis):
        self.initial=axis.copy();self.axis=axis.copy();self.confident=False
        self.lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    def update(self,image,rig):
        gray=cv2.cvtColor(np.rint(image*255).astype('uint8'),cv2.COLOR_RGB2GRAY)
        lines=self.lsd.detect(gray)[0]
        if lines is None:return None
        rows=lines[:,0,:];a,b=rows[:,:2],rows[:,2:];dots=b-a;length=np.linalg.norm(dots,axis=1)
        slant=np.abs(dots[:,1])/np.maximum(length,1.e-7)
        idx=(length>=12)&(slant>.13)&(slant<.95)
        a,b,length=a[idx],b[idx],length[idx]
        if len(a)<3:return None
        homogeneous=lambda xy:np.c_[xy,np.ones(len(xy))]
        normal=np.cross(homogeneous(a),homogeneous(b)) @ rig.ks['rgb']
        normal=normal @ (rig.R@rig.W[:3,:3].T).T
        directions=np.stack([-normal[:,1],normal[:,0]],axis=1)
        directions/=np.maximum(np.linalg.norm(directions,axis=1)[:,None],1.e-8)
        directions*=np.where(directions@self.initial>=0,1.,-1.)[:,None]
        cross=self.initial[0]*directions[:,1]-self.initial[1]*directions[:,0]
        angles=np.rad2deg(np.arctan2(cross,directions@self.initial))
        grid=np.arange(-35,36,1)
        votes=np.maximum(0,1-abs(angles[:,None]-grid[None,:])/3.)
        weights=length@votes;peak=grid[np.argmax(weights)]
        keep=np.abs(angles-peak)<3.5
        if keep.sum()<3 or length[keep].sum()<75:return None
        estimate=np.average(directions[keep],axis=0,weights=length[keep])
        # Retain a stable axis through brief occlusions.
        alpha=.12 if self.confident else 1.
        axis=(1-alpha)*self.axis+alpha*estimate
        self.axis=axis/np.linalg.norm(axis);self.confident=True
        return dict(support=int(keep.sum()),yaw=float(np.rad2deg(np.arctan2(self.axis[1],self.axis[0]))))
