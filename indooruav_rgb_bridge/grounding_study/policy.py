"""Frozen intervention. No scene identifier, global goal, expert labels or example trajectories."""
import math
import re
import numpy as np

def wrap(rad):
    return (rad+math.pi)%(2*math.pi)-math.pi

def limit(vector, radius):
    return vector * min(1., radius / max(np.linalg.norm(vector),1e-8))

class Policy:
    def __init__(self, instruction):
        self.text=instruction
        t=instruction.lower()
        self.search='laptop' in t and ('search' in t or 'find' in t)
        self.objects=['laptop','sofa'] if self.search and 'sofa' in t else ['ac'] if 'air conditioner' in t else ['laptop'] if 'laptop' in t else ['sofa'] if 'sofa' in t else ['armchair'] if 'armchair' in t else []
        m=re.search(r'about\s+([\d.]+)\s*m',t)
        self.alt=float(m.group(1)) if m else None
        self.profile=('accelerat' in t or 'speed up' in t) and ('descend' in t or 'descent' in t or 'slow' in t)
        self.climb='climb' in t or 'rise' in t
        self.index=self.confirm=0
        self.last_yaw=None
        self.scan=0.
        self.desc=False
        self.z0=None
        self.bearings=BearingMemory()
        self.frame=-1
    def update(self, now, rgb, detector, camera_ray=None):
        self.frame+=1
        if self.z0 is None:self.z0=now[2]
        if self.last_yaw is not None:self.scan+=abs(wrap(now[3]-self.last_yaw))
        self.last_yaw=now[3]
        if not self.objects or detector is None:return self.text,None
        if self.confirm>=2 and self.index<len(self.objects)-1:
            self.index+=1
            self.confirm=0
        obj=self.objects[self.index]
        det=detector(rgb,obj)
        side=det['side']
        self.confirm=self.confirm+1 if side=='center' and det['boxes'][0]['area']>=.03 else 0
        ray=camera_ray(det['boxes'][0]) if det['boxes'] else None
        angle,source=self.bearings.estimate(obj,det,now,ray,self.frame)
        det=dict(det,bearing_right_deg=angle,bearing_source=source)
        if angle is None:return self.text,det
        noun={'ac':'air conditioner','laptop':'laptop','sofa':'sofa','armchair':'black armchair by the door'}[obj]
        hint=f'Turn {side} toward the {noun}.' if side!='center' else f'The {noun} is centered ahead.'
        if source=='memory':
            hint=f'Turn in place to face the {noun}, last seen {SECTORS[sector(angle)]} relative to your current heading.'
        else:
            hint+=f' It is {SECTORS[sector(angle)]} relative to your heading.'
        return self.text+' '+hint,det
    def check(self,now,proposal,stopped,speed):
        # Compute the same predicate on the raw candidate even when corrections are disabled.
        delta=proposal[:3]-now[:3]
        xy=delta[:2].copy()
        dz=float(delta[2])
        yaw=wrap(proposal[3]-now[3])
        why=[]
        max_xy=.65
        desired=dz
        if self.alt is not None:
            if abs(proposal[2]-self.alt)>.20:why.append('cruise_altitude')
            desired=self.alt-now[2]
        if self.profile:
            if now[2]>=self.z0+.15 and dz<-.04:self.desc=True
            if not self.desc and dz<-.04:why.append('descent_before_climb')
            if self.desc and dz>.05:why.append('rise_during_descent')
            desired=np.clip(dz,-.22,0) if self.desc else np.clip(dz,0,.22)
            max_xy=min(max_xy, max(.12,speed-.08) if self.desc else max(.2,speed+.25))
        if self.search and self.scan<math.radians(12):
            max_xy=.12
            if abs(yaw)<math.radians(8): yaw=math.radians(12)
            if dz<-.05:why.append('descent_during_search')
            desired=max(0.,desired)
        if self.climb and not self.profile and now[2]<self.z0+.12 and dz<-.05:
            why.append('descent_before_rise')
            desired=max(0.,desired)
        if np.linalg.norm(xy)>max_xy+1e-5:why.append('horizontal_step')
        if abs(dz)>.25:why.append('vertical_step')
        if abs(wrap(proposal[3]-now[3]))>math.radians(25):why.append('yaw_step')
        incomplete=bool((self.alt is not None and abs(now[2]-self.alt)>.20) or (self.profile and not self.desc) or (self.search and self.scan<math.radians(12)))
        if stopped and incomplete:why.append('stop_before_observable_stage')
        target=now.copy()
        target[:2]+=limit(xy,max_xy)
        target[2]+=np.clip(desired,-.25,.25)
        target[3]+=np.clip(yaw,-math.radians(25),math.radians(25))
        if stopped and self.profile and not self.desc and now[2]<self.z0+.15:
            target[2]=now[2]+.15
        return target,bool(stopped and not incomplete),why

SECTORS=('ahead','ahead and to the right','to the right','behind and to the right','behind','behind and to the left','to the left','ahead and to the left')
def sector(angle):
    return int(math.floor((angle+22.5)/45))%8

class BearingMemory:
    def __init__(self):
        self.last={}
    def estimate(self,obj,det,now,ray_right_deg,step):
        if det['boxes']:
            self.last[obj]=(step, now[:3].copy(), math.degrees(now[3])+ray_right_deg)
        if obj not in self.last:return None,None
        t,xyz,az=self.last[obj]
        if step-t>8 or np.linalg.norm(now[:3]-xyz)>.35:return None,None
        return (az-math.degrees(now[3])+180)%360-180, 'current' if t==step else 'memory'
