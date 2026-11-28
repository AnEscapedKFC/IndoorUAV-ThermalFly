"""Regress the distinction between measured completion and coverage interruption."""
import unittest
from types import SimpleNamespace
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parents[1])]
from metric_flight.observation_window import window_verdict
from metric_flight.coverage_review import CoverageReview
from metric_flight.depth_guard import DepthGuard


def sample(t,area):
    return dict(t=t,area=area,source='missing' if area is None else 'rgbd_track',
                centered=True,clipped=False,parts_ok=True,view_ok=True)


class ReviewTest(unittest.TestCase):
    def test_median_gap_and_extension(self):
        def ok(areas, drift=.01):
            return window_verdict([sample(.5*j,a) for j,a in enumerate(areas)],1.,drift)['confirmed']
        self.assertTrue(ok([.10,None,.10]))
        self.assertTrue(ok([.10,.09,.10]))
        self.assertFalse(ok([.10,None,None]))
        self.assertFalse(ok([.10,.09,.09]))
        self.assertTrue(ok([.10,.12,None,.13]))
        self.assertFalse(ok([.51,.51,.51]))
        self.assertFalse(ok([.10,None,.10],.13))
        s=[sample(0,.1),sample(.5,.11),sample(1,.10)]
        s[1]['parts_ok']=False
        self.assertFalse(window_verdict(s,1,.01)['confirmed'])

    def test_one_brake_and_hard_stop(self):
        st=dict(position=[0.,0.,0.],velocity=[0.,0.,0.],euler_deg=[0.,0.,0.],angular_velocity=[0.,0.,0.],
                quaternion_wxyz=[1.,0.,0.,0.],speed=0.)
        r=dict(state=st,sim_time=0.)
        sim=SimpleNamespace(last_receipt=r,obstacles=SimpleNamespace(half_size=1,safety_half_size=2))
        grid=SimpleNamespace(contains_box=lambda p,R,h:h==1)
        guard=SimpleNamespace(stationary=lambda s:({'status':'free'},[]))
        from unittest.mock import patch
        c=CoverageReview()
        with patch('metric_flight.coverage_review.controller',return_value=(None,None)):
            def tick(_):
                r['sim_time']+=.05
            self.assertEqual(c.check(sim,guard,grid,tick),'replan_after_brake')
            self.assertIsNone(c.check(sim,guard,grid,tick))
            self.assertEqual(len(c.events),1)
            c=CoverageReview()
            self.assertEqual(c.check(sim,guard,grid,lambda _: 'collision'),'collision')

    def test_fringe_revalidated_at_each_pose(self):
        p=np.array([0.,0.,1.]); R=np.eye(3)
        g=DepthGuard(p,np.array([.35,.35,.11])); g.seed_body(p,R)
        idx=g.body(p,R); core=set(map(tuple,np.column_stack(g.body(p,R,extra=0.))))
        cell=next(tuple(v) for v in np.column_stack(idx) if tuple(v) not in core)
        g.evidence[cell]=12
        state=dict(position=p,quaternion_wxyz=[1,0,0,0])
        self.assertTrue(g.recheck_launch_fringe(state))
        near=g.origin+(np.asarray(cell)+.5)*g.cell
        self.assertIn(cell,set(map(tuple,np.column_stack(g.body(near,R)))))
