"""Instruction-gated metric checks and a current-RGB visual inspection controller."""
import math
import numpy as np
from metric_flight.tracking import BoxTracker

from metric_flight.instruction import quantities

def gate(text,switch):
    if switch not in ('auto','off'):raise ValueError(switch)
    h,l=quantities(text)
    return switch=='auto' and (h is not None or l is not None)

def angle(x):return (x+180.)%360.-180.
def unit(v):return v/max(1e-8,np.linalg.norm(v))
class Odometer:
    def __init__(self,pos,yaw):
        self.origin=np.array(pos);self.last=self.origin.copy()
        self.axis=np.array([math.cos(math.radians(yaw)),math.sin(math.radians(yaw))])
        self.path=0.;self.forward=0.
    def tick(self,p):
        self.path+=float(np.linalg.norm(p-self.last))
        self.forward=float((p-self.origin)[:2]@self.axis)
        self.last=p.copy()

def metric_waypoint(text,setting,raw,p,yaw,odom,dt,course=None):
    h,length=quantities(text);target=raw.copy()
    if not gate(text,setting):return target
    if h is not None:target[2]=h
    if length is not None:
        # Track the relative straight line requested by the user. Preserve small policy
        # steering and altitude adjustments at cruise, without importing a map or a goal pose.
        axis=odom.axis if course is None else course
        side=np.array([-axis[1],axis[0]])
        remaining=length-odom.path
        turn=math.degrees(math.atan2(axis[1],axis[0]))
        travel=np.clip((raw[:2]-p[:2])@axis,.55*dt,.9*dt)
        travel=np.clip(remaining,-travel,travel)
        cross=(p[:2]-odom.origin[:2])@side
        lateral=np.clip((raw[:2]-p[:2])@side,-.08*dt,.08*dt)-.5*cross
        target[:2]=p[:2]+axis*travel+side*np.clip(lateral,-.20*dt,.20*dt)
        target[3]=turn+np.clip(angle(raw[3]-turn),-6.,6.)
        if h is None:
            # A horizontal "forward" primitive accepts modest model vertical deviations;
            # there is no absolute target or correction to a predetermined altitude.
            target[2]=p[2]+np.clip(raw[2]-p[2],-.045*dt,.045*dt)
    return target

class Inspection:
    def __init__(self,text):
        self.arc='arc' in text.lower().split() or 'along an arc' in text.lower()
        self.climb='climb' in text.lower()
        self.z0=None;self.estimate=None;self.last_box=None;self.last_area=None
        self.rays=[];self.first_az=None;self.span=0.;self.sign=0;self.centered=0
        self.orbiting=False;self.lost=0;self.tracker=BoxTracker();self.arc_goal=28.;self.planned=False
    def step(self,p,yaw,v,raw,det,rig,dt,image):
        if self.z0 is None:self.z0=p[2]
        b,tracking=self.tracker.observe(image,det)
        if b is None:
            self.centered=0;self.first_az=None;self.span=0.;self.lost+=1
            return np.r_[p,yaw],False,dict(phase='observation_lost',lost=self.lost)
        self.lost=0;x0,y0,x1,y1=b['xyxy'];u=(x0+x1)/2;w=(y0+y1)/2
        self.last_box=(u,w);self.last_area=b['area']
        # Calibrated camera origin/orientation, from the currently measured camera pose.
        c=rig.s*(rig.R @ (np.linalg.inv(rig.W)[:3,3]-rig.origin))
        R=rig.R@rig.W[:3,:3].T
        K=rig.ks['rgb'];opt=np.linalg.solve(K,[u*rig.width,w*rig.height,1.])
        ray=unit(R@opt)
        # Initial range from a generic 0.80 m appliance width. Refine by multiview rays
        # after translation; no scene landmark, reference trajectory or goal is read.
        size_range=np.clip(.80*K[0,0]/((x1-x0)*rig.width)*np.linalg.norm(opt),.65,7.)
        measured=c+ray*size_range
        if not self.rays or np.linalg.norm(c-self.rays[-1][0])>.08:
            self.rays.append((c.copy(),ray.copy()))
        obs=self.rays[-35:]
        A=np.zeros((3,3));rhs=np.zeros(3)
        for org,d in obs:
            H=np.eye(3)-np.outer(d,d);A+=H;rhs+=H@org
        ev=np.linalg.eigvalsh(A)[0]
        if len(obs)>=4 and ev>.025:
            loc=np.linalg.solve(A+.025*np.eye(3),rhs+.025*measured)
            if .6<np.dot(loc-c,ray)<6 and np.linalg.norm(loc-measured)<1.8:measured=loc
        self.estimate=measured if self.estimate is None else .65*self.estimate+.35*measured
        rel=self.estimate-p;rad=unit(rel[:2]);ortho=np.array([-rad[1],rad[0]])
        distance=float(np.linalg.norm(rel));xy_range=np.linalg.norm(rel[:2])
        # Maintain the current camera mount, using vertical image displacement for altitude.
        view_yaw=np.rad2deg(np.arctan2(ray[1],ray[0]));aim=yaw+np.clip(angle(view_yaw-yaw),-18.,18.)
        centered=abs(u-.5)<=.13 and abs(w-.5)<=.16
        # Angular width and range are independent cues when the view becomes oblique.
        close=distance<=1.55 or (x1-x0)>=.33
        if close:self.orbiting=True
        phase='approach'
        if self.arc and self.orbiting:phase='arc'
        normal=self.tracker.normal(rig,-rad) if self.arc and not self.planned else None
        if phase=='arc' and not self.planned:
            if normal is not None:
                # Cross the frontal view and sample a complementary oblique angle.
                oblique=np.rad2deg(np.arccos(np.clip(normal@(-rad),-1,1)))
                if oblique>6:
                    self.sign=1 if normal@ortho>=0 else -1
                    self.arc_goal=float(np.clip(2*oblique,28.,82.))
            self.planned=True
        if self.sign==0:
            # Use the model's initial preferred lateral direction; ties use incoming motion.
            lat=(raw[:2]-p[:2])@ortho
            self.sign=1 if (lat if abs(lat)>.025 else v[:2]@ortho)>=0 else -1
        inward=np.clip((xy_range-1.36)*.65,-.16,.36)
        if not close and not self.orbiting:inward=max(inward,.14)
        if abs(u-.5)>.23:inward=min(inward,.04)
        lateral=0.;az=np.rad2deg(np.arctan2(p[1]-self.estimate[1],p[0]-self.estimate[0]))
        if phase=='arc':
            if not centered:self.first_az=None;self.span=0.
            elif self.first_az is None:
                self.first_az=az;self.span=0.;self.arc_anchor=self.estimate.copy()
                self.first_az=np.rad2deg(np.arctan2(p[1]-self.arc_anchor[1],p[0]-self.arc_anchor[0]))
            if self.first_az is not None:
                arc_now=np.rad2deg(np.arctan2(p[1]-self.arc_anchor[1],p[0]-self.arc_anchor[0]))
                self.span=abs(angle(arc_now-self.first_az))
            if self.span<self.arc_goal:lateral=self.sign*.30
        ready=self.orbiting and centered and (not self.arc or self.span>=self.arc_goal)
        self.centered=self.centered+1 if ready else 0
        stop=self.centered>=3 and np.linalg.norm(v)<.12
        if ready:inward=0.;lateral=0.;phase='settle'
        planar=rad*inward+ortho*lateral
        target=p.copy();target[:2]+=dt*planar
        rise=np.clip((.5-w)*.9,-.22,.22)
        if self.climb and p[2]<self.z0+.10:rise=max(rise,.12)
        if ready and abs(.5-w)<.09:rise=0.
        target[2]+=dt*rise
        return np.r_[target,aim],bool(stop),dict(phase=phase,range_est_m=distance,angular_width=x1-x0,cx=u,cy=w,
                     tracking=tracking,arc_goal=self.arc_goal,centered=bool(centered),arc_deg=self.span,sign=self.sign,center_streak=self.centered)
