"""Brake once per map-coverage incident and recheck with current sensor evidence."""
import numpy as np
from scipy.spatial.transform import Rotation
from sim_bridge import CONTROL_DT, controller, guided_curve


def orientation(state):
    return Rotation.from_quat(np.asarray(state['quaternion_wxyz'])[[1,2,3,0]]).as_matrix()


class CoverageReview:
    def __init__(self):
        self.events = []
        self.reviewed = False
        self.inside_since = None

    def check(self, sim, guard, known, advance):
        r = sim.last_receipt
        st = r['state']
        p = np.asarray(st['position'])
        now = r['sim_time']
        inside = known.contains_box(p, orientation(st), sim.obstacles.safety_half_size)
        if inside:
            if self.inside_since is None:
                self.inside_since = now
            if now-self.inside_since >= .5:
                self.reviewed = False
            return None
        self.inside_since = None
        if self.reviewed:
            return None if self.supported(sim, guard, known) else 'map_coverage_unresolved'
        self.reviewed = True
        event = dict(t=now, position=p.tolist())
        self.events.append(event)
        goal = np.r_[p, st['euler_deg'][2]]
        # Switch to zero terminal velocity at the measured position. Hold for a fresh depth pass.
        curve = guided_curve(st, goal, 1., True)
        for k in range(round(1./CONTROL_DT)):
            action,_ = controller(sim, sim.last_receipt, None, (k+.5)*CONTROL_DT, sample=curve)
            failure = advance(action)
            if failure:
                event.update(result=failure, end_t=sim.last_receipt['sim_time'])
                return failure
        st = sim.last_receipt['state']
        supported = self.supported(sim,guard,known)
        event.update(result='rechecked' if supported and st['speed']<=.10 else 'unresolved',
                     end_t=sim.last_receipt['sim_time'], speed=st['speed'], supported=supported)
        return 'replan_after_brake' if event['result']=='rechecked' else 'map_coverage_unresolved'

    @staticmethod
    def supported(sim, guard, known):
        state = sim.last_receipt['state']
        core_inside = known.contains_box(np.asarray(state['position']), orientation(state), sim.obstacles.half_size)
        entry,_ = guard.stationary(state)
        return bool(core_inside and entry['status']=='free')
