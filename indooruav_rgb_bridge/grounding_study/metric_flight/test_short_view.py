from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parents[1])]
from metric_flight.depth_guard import DepthGuard
from metric_flight.short_view import check_step, history_corners, pick_step
from metric_flight.unknown_views import BlockedViews, RayFrame


class ShortViewTests(unittest.TestCase):
    def setUp(self):
        self.k=np.array([[80.,0.,80.],[0.,80.,60.],[0.,0.,1.]])
        self.rig=SimpleNamespace(R=np.eye(3),W=np.eye(4),s=1.,origin=np.zeros(3),ks={'rgb':self.k},
                                 width=160,height=120,pitch_deg=0.)
        self.state=dict(position=[0.,0.,0.],euler_deg=[0.,0.,0.],velocity=[0.,0.,0.],
                        angular_velocity=[0.,0.,0.],quaternion_wxyz=[1.,0.,0.,0.])
        self.recovery=BlockedViews()
        self.guard=DepthGuard(np.zeros(3),[.1,.08,.04],self.recovery)
        self.cell=np.floor((np.array([.45,0.,1.])-self.guard.origin)/self.guard.cell).astype(int)
        self.corners=self.guard.origin+(self.cell[None,None,:]+.5+self.recovery.history.corners*.5)*self.guard.cell

    def test_seen_corner_requires_depth_in_front_of_occluder(self):
        patch_rgb=np.zeros((30,40,3),np.uint8)
        def frame(z):
            depth=np.full((30,40),z)
            return RayFrame(1,np.zeros(3),np.eye(3),self.k,0.,patch_rgb,depth,depth)
        self.assertFalse(history_corners([frame(.3)],self.corners).any())
        self.assertTrue(history_corners([frame(2.)],self.corners).all())

    def test_no_scan_for_repeated_partial_corners(self):
        partial=np.zeros((1,8),bool)
        partial[0,[1,4]]=True
        other=np.zeros((1,8),bool)
        other[0,[0,3]]=True
        for prior,want_scan in ((partial,False),(other,True)):
            with self.subTest(want_scan=want_scan),patch('metric_flight.unknown_views.history_corners',return_value=prior.copy()),patch('metric_flight.unknown_views.visible_corners',return_value=(partial,None,None)):
                plan=self.recovery.target({'move':[self.cell]},self.guard,self.rig,self.state)
                self.assertEqual(plan is not None,want_scan)

    def test_short_move_requires_a_free_path_without_cell_release(self):
        args=(self.guard,self.state,self.rig,self.corners,np.zeros((1,8),bool),[np.zeros(3)],[.2,0.,0.,0.])
        self.guard.seed_body(np.zeros(3),np.eye(3))
        self.guard.evidence[:]=1
        before=self.guard.evidence.copy()
        best,_=pick_step(*args)
        self.assertIsNone(best)
        np.testing.assert_array_equal(self.guard.evidence,before)
        self.guard.evidence[:]=-3
        best,_=pick_step(*args)
        _,goal,seconds,_=best
        self.assertLessEqual(np.linalg.norm(goal[:3]),.081)
        self.assertEqual(goal[3],0.)
        self.assertGreaterEqual(seconds,1.)
        self.assertEqual(self.guard.check(self.state,goal,seconds,None,None),'free')

    def test_intermediate_brake_rejected_even_when_primary_path_is_free(self):
        goal=np.array([.08,0.,0.,0.])
        def check(st,end,seconds,*args):
            if .02<st['position'][0]<.06 and np.linalg.norm(st['velocity'])>.01:
                return 'unknown'
            return 'free'
        with patch.object(self.guard,'check',side_effect=check):
            self.assertEqual(check_step(self.guard,self.state,goal,2.),'braking_unconfirmed')
