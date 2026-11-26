"""Appearance tracking and a front-face normal from the observed appliance's horizontal edges."""
import cv2
import numpy as np

class BoxTracker:
    def __init__(self):
        self.gray=None;self.box=None;self.age=0;self.axis_votes=[]
        self.lines=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    def observe(self,image,det):
        gray=cv2.cvtColor(np.rint(image*255).astype('uint8'),cv2.COLOR_RGB2GRAY)
        h,w=gray.shape;wh=np.array([w,h,w,h]);b=self.box
        candidates=[v for v in det['boxes'] if .002<v['area']<.55]
        if b is not None:
            center=(b[:2]+b[2:])/2;area=np.prod(b[2:]-b[:2])
            def plausible(v):
                z=np.array(v['xyxy']);return np.linalg.norm((z[:2]+z[2:])/2-center)<.24 and .40<v['area']/area<2.50
            candidates=[v for v in candidates if plausible(v)]
        source='detector';selected=candidates[0] if candidates else None
        if selected is None and self.gray is not None and b is not None and self.age<8:
            mask=np.zeros_like(gray);x0,y0,x1,y1=np.rint(b*wh).astype(int)
            mask[y0:y1,x0:x1]=255
            prev=cv2.goodFeaturesToTrack(self.gray,60,.008,3,mask=mask)
            if prev is not None and len(prev)>=5:
                pts,s1,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,prev,None)
                back,s2,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,pts,None)
                ok=(s1[:,0]>0)&(s2[:,0]>0)&(np.linalg.norm(back-prev,axis=2)[:,0]<1.4)
                if ok.sum()>=5:
                    transform,inlier=cv2.estimateAffinePartial2D(prev[ok],pts[ok],method=cv2.RANSAC,ransacReprojThreshold=2.)
                    if transform is not None and inlier.sum()>=5 and .75<np.linalg.norm(transform[:,0])<1.3:
                        cp=np.array([[(x0+x1)/2,(y0+y1)/2,1.]])@transform.T
                        half=.5*np.array([x1-x0,y1-y0])*np.linalg.norm(transform[:,0])
                        bb=np.clip(np.r_[cp[0]-half,cp[0]+half]/wh,0.,1.)
                        if .002<np.prod(bb[2:]-bb[:2])<.55:
                            selected=dict(xyxy=bb.tolist(),area=float(np.prod(bb[2:]-bb[:2])),x=float((bb[0]+bb[2])/2),score=1.)
                            source='tracked_current_rgb'
        self.gray=gray;self.age=0 if source=='detector' and selected else self.age+1
        if selected:self.box=np.array(selected['xyxy'])
        return selected,source if selected else None
    def normal(self,rig,towards_camera):
        if self.box is None:return None
        h,w=self.gray.shape;x0,y0,x1,y1=np.rint(self.box*[w,h,w,h]).astype(int)
        if x1-x0<28 or y1-y0<9:return None
        roi=self.gray[y0:y1,x0:x1];segments=self.lines.detect(roi)[0]
        if segments is not None:
            a=segments[:,0,:2]+[x0,y0];b=segments[:,0,2:]+[x0,y0];diff=b-a;length=np.linalg.norm(diff,axis=1)
            good=(length>max(8.,.20*(x1-x0)))&(abs(diff[:,1])<.5*abs(diff[:,0]))
            for aa,bb,wt in zip(a[good],b[good],length[good]):
                plane=rig.R@rig.W[:3,:3].T @ (rig.ks['rgb'].T@np.cross(np.r_[aa,1.],np.r_[bb,1.]))
                # Horizontal normal to the image plane containing the observed straight edge.
                if np.linalg.norm(plane[:2])/np.linalg.norm(plane)<.08:continue
                theta=np.rad2deg(np.arctan2(plane[1],plane[0]))%180
                self.axis_votes.append((theta,wt))
        if len(self.axis_votes)<5:return None
        votes=np.array(self.axis_votes[-90:]);grid=np.arange(180.)
        gap=abs((votes[:,0,None]-grid[None,:]+90)%180-90)
        peak=np.argmax(votes[:,1]@np.maximum(0,1-gap/9))
        valid=gap[:,peak]<10
        if votes[valid,1].sum()<.32*votes[:,1].sum():return None
        n=np.array([np.cos(np.deg2rad(peak)),np.sin(np.deg2rad(peak))])
        if n@towards_camera<0:n=-n
        return n
