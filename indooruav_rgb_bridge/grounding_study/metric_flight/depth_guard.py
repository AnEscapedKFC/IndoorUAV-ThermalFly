"""Local projective occupancy and swept-body checks. No scene collision map is read."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from sim_bridge import guided_curve
from metric_flight.depth_guidance import camera


class DepthGuard:
    cell=.065
    def __init__(self, origin, half):
        self.origin=np.asarray(origin)-[6.,6.,3.]
        self.half=np.asarray(half)
        self.evidence=np.zeros((185,185,94),np.int8)
        self.offsets=np.indices((63,63,41),dtype=np.int16).reshape(3,-1).T-[31,31,20]
        self.stalled=0
        self.slide_sign=1

    def observe(self,z,rig):
        center,R,K=camera(rig)
        mid=np.floor((center-self.origin)/self.cell).astype(int)
        ids=self.offsets+mid
        ids=ids[((ids>=0)&(ids<self.evidence.shape)).all(1)]
        opt=(self.origin+(ids+.5)*self.cell-center)@R
        front=opt[:,2]>.08
        ids=ids[front];opt=opt[front]
        h,w=z.shape
        # Conservative nearest surface per image tile; invalid depths cast no free votes.
        scale=8
        clean=np.where(np.isfinite(z)&(z>.08),z,0.).astype(np.float32)
        near=cv2.erode(clean,np.ones((scale,scale),np.uint8))[scale//2::scale,scale//2::scale]
        far=cv2.dilate(clean,np.ones((scale,scale),np.uint8))[scale//2::scale,scale//2::scale]
        u=np.floor((K[0,0]*opt[:,0]/opt[:,2]+K[0,2])/scale).astype(int)
        v=np.floor((K[1,1]*opt[:,1]/opt[:,2]+K[1,2])/scale).astype(int)
        valid=(u>=0)&(v>=0)&(u<near.shape[1])&(v<near.shape[0])
        ids=ids[valid];d=opt[valid,2];u=u[valid];v=v[valid]
        lo=near[v,u];hi=far[v,u]
        free=(lo>.10)&(d<lo-(.10+.025*d))
        surface=(lo>.10)&(hi-lo<.14+.04*d)&(abs(d-lo)<=.065+.02*d)
        for mask,delta in [(free,-3),(surface,12)]:
            key=tuple(ids[mask].T)
            updated=self.evidence[key].astype(int)+delta
            if delta>0:updated=np.maximum(updated,1)
            self.evidence[key]=np.clip(updated,-12,24)

    def body(self,p,R):
        half=self.half+.025
        extent=abs(R)@half
        low=np.floor((p-extent-self.origin)/self.cell).astype(int)
        high=np.floor((p+extent-self.origin)/self.cell).astype(int)
        if np.any(low<0) or np.any(high>=self.evidence.shape):return None
        ids=np.indices(high-low+1).reshape(3,-1).T+low
        delta=(self.origin+(ids+.5)*self.cell-p)@R
        touches=(np.abs(delta)<=half+.5*self.cell*abs(R).sum(axis=0)).all(1)
        return tuple(ids[touches].T)

    def seed_body(self,p,R):
        idx=self.body(p,R)
        if idx is None:raise ValueError('Spawn outside local depth grid')
        self.evidence[idx]=-12

    def check(self,st,goal,seconds,start_v,end_v):
        sample=guided_curve(st,goal,seconds,True,start_velocity=start_v,end_velocity=end_v)
        times=np.linspace(0.,seconds,max(2,int(np.ceil(seconds/.075))+1))
        current=Rotation.from_quat(np.asarray(st['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
        p0=np.asarray(st['position'])
        v0=np.asarray(st['velocity'])
        reference_v=sample(None,0.)[1][:3]
        points=[]
        for t in times:
            pos,vel,acc,_=sample(None,t)
            # Cover actual tilt at entry and predicted attitude from thrust acceleration.
            yaw=np.deg2rad(pos[3]);up=acc[:3]+[0.,0.,9.81];up/=np.linalg.norm(up)
            left=np.cross(up,[np.cos(yaw),np.sin(yaw),0.]);left/=np.linalg.norm(left)
            attitude=np.column_stack((np.cross(left,up),left,up))
            correction=(v0-reference_v)*min(t,.2)
            points.append((pos[:3]+correction,attitude))
        points.append((p0,current))
        last=points[-2][0];vel=sample(None,seconds)[1][:3]
        # Reaction plus controlled braking at 0.8 m/s². Check intermediate positions too.
        brake=vel*(.10+np.linalg.norm(vel)/1.6)
        for fraction in np.linspace(0.,1.,max(2,int(np.ceil(np.linalg.norm(brake)/.04))+1)):
            points.append((last+fraction*brake,attitude))
        for pos,rot in points:
            ids=self.body(pos,rot)
            if ids is None:return 'unknown'
            ev=self.evidence[ids]
            if np.any(ev>0):return 'surface'
            if np.any(ev>=0):return 'unknown'
        return 'free'

    def select(self,st,goal,seconds,passing,prev,rig=None,anchor=None):
        p=np.asarray(st['position']);d=goal[:3]-p
        choices=[('proposed',d)]
        if np.linalg.norm(d[:2])>.005:
            forward=d[:2]/np.linalg.norm(d[:2])
            side=self.slide_sign*np.array([-forward[1],forward[0]])
            choices += [('level',np.r_[d[:2],max(0.,d[2])]),
                        ('short_level',np.r_[.5*d[:2],max(0.,d[2])]),
                        ('slide',np.r_[.08*side+.025*forward,0.]),
                        ('opposite_slide',np.r_[-.08*side+.025*forward,0.]),
                        ('rise_forward',np.r_[.45*d[:2],.055]),
                        ('rise',np.array([0.,0.,.055]))]
        failures={}
        for name,delta in choices:
            target=np.r_[p+delta,goal[3]]
            # During avoidance do not climb or sidestep until the semantic point
            # leaves the fixed camera's field of view. Project with the candidate yaw.
            if name!='proposed' and anchor is not None:
                c,R,K=camera(rig)
                turn=Rotation.from_euler('z',goal[3]-st['euler_deg'][2],degrees=True).as_matrix()
                relative=(turn@R).T@(np.asarray(anchor)-(p+delta+turn@(c-p)))
                uv=K@relative
                if uv[2]<=.10 or np.any(abs(uv[:2]/uv[2]/[rig.width,rig.height]-.5)>.32):
                    failures[name]='target_leaves_view'
                    continue
            status=self.check(st,target,seconds,prev if passing else None,delta/seconds if passing else None)
            if status=='free':
                self.stalled=0
                if name=='opposite_slide':self.slide_sign*=-1
                return target,dict(choice=name,rejected=failures)
            failures[name]=status
        self.stalled+=1
        hold=np.r_[p,st['euler_deg'][2]]
        brake_status=self.check(st,hold,seconds,None,None)
        return hold,dict(choice='brake',rejected=failures,braking_envelope=brake_status)


def survey_rig(base):
    """Survey pan is applied only during the preflight sensor pass."""
    class SurveyRig(base):
        pan_deg=0.
        def render(self,pose):
            if self.pan_deg:
                pose=np.asarray(pose).copy()
                pose[:3,:3]=pose[:3,:3]@Rotation.from_euler('y',self.pan_deg,degrees=True).as_matrix()
            return super().render(pose)
    return SurveyRig


def preflight(sim,guard):
    """Physically hover at launch and scan with the calibrated camera before task reset."""
    from sim_bridge import CONTROL_DT,controller
    rig=sim.scene
    initial=sim.last_receipt['state']
    p=np.asarray(initial['position']);yaw=initial['euler_deg'][2]
    guard.seed_body(p,Rotation.from_euler('z',yaw,degrees=True).as_matrix())
    initial_rgb=sim.last_rgb.copy()
    original_pitch=rig.pitch_deg
    receipt=sim.last_receipt
    hold=guided_curve(initial,np.r_[p,yaw],1.,True)
    samples=[]
    # Include polar caps. The camera sits ahead of the center of rotation,
    # so newly swept top/bottom corners can lie almost directly above/below it.
    pitches=(-68.,-32.,0.,32.,68.)
    turns=60
    for idx in range(turns*len(pitches)):
        band=idx//turns
        rig.pan_deg=(idx%turns)*360./turns*(-1 if band%2 else 1)
        rig.pitch_deg=pitches[band]
        u,_=controller(sim,receipt,None,min((idx+.5)*CONTROL_DT,1.),sample=hold)
        receipt=sim.step(u,format='body_rates_thrust')
        if receipt['terminated'] or receipt['truncated']:
            raise RuntimeError(f'Launch survey stopped: {receipt["reason"]}')
        if idx%2==0:
            guard.observe(sim.last_depth,rig)
            samples.append(dict(t=receipt['sim_time'],position=receipt['state']['position'],
                                w2c=rig.W.tolist(),pitch_deg=rig.pitch_deg,pan_deg=rig.pan_deg))
    rig.pitch_deg=original_pitch
    rig.pan_deg=0.
    sim.reset()
    if not np.allclose(initial_rgb,sim.last_rgb,atol=1e-5):raise RuntimeError('Task reset changed first RGB')
    return dict(duration_s=turns*len(pitches)*CONTROL_DT,samples=samples)
