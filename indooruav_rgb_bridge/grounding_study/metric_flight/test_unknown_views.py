"""Regressions for historical free-space witnesses and in-place observation."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1])]
from metric_flight.unknown_views import BlockedViews
from metric_flight.depth_guard import DepthGuard


class UnknownViewTests(unittest.TestCase):
    def setUp(self):
        self.views = BlockedViews()
        self.guard = DepthGuard(np.zeros(3), [.1,.08,.04], self.views)
        self.rig = SimpleNamespace(R=np.eye(3), W=np.eye(4), s=1., origin=np.zeros(3),
                                   pitch_deg=-7., ks={'rgb':np.array([[80,0,80],[0,80,60],[0,0,1.]])})
        self.rgb = np.zeros((120,160,3),np.float32)
        self.id = np.floor((np.array([0.,0.,1.])-self.guard.origin)/self.guard.cell).astype(int)

    def add(self, x, depth):
        self.rig.W[0,3] = -x
        self.views.history.add(self.rgb, depth, self.rig)

    def test_calibrated_live_views_only_clear_unknown(self):
        z = np.full((120,160),2.)
        self.add(0.,z)
        self.add(0.,z)
        self.assertEqual(self.views.clear(self.guard,[self.id]),0)
        self.add(.035,z)
        self.assertEqual(self.views.clear(self.guard,[self.id]),1)
        self.assertEqual(self.views.releases[0]['witnesses'],[2,1])
        self.guard.evidence[tuple(self.id)]=8
        self.assertEqual(self.views.clear(self.guard,[self.id]),0)
        self.assertEqual(self.guard.evidence[tuple(self.id)],8)

    def test_occluded_missing_or_surface_veto(self):
        for d in (.0,.5,1.02):
            with self.subTest(d=d):
                self.views=BlockedViews()
                z=np.full((120,160),d)
                self.add(0.,z)
                self.add(.035,z)
                self.assertEqual(self.views.clear(self.guard,[self.id]),0)
        z=np.full((120,160),2.)
        self.add(.07,z)
        self.add(.105,z)
        self.assertEqual(self.views.clear(self.guard,[self.id]),0)

    def test_scan_never_starts_with_translation_speed_or_known_surface(self):
        st=dict(position=[0.,0.,0.],velocity=[.15,0.,0.],quaternion_wxyz=[1.,0.,0.,0.],euler_deg=[0.,0.,0.])
        sim=SimpleNamespace(last_receipt={'state':st,'sim_time':0.},scene=self.rig)
        self.rig.width=160
        self.rig.height=120
        self.views.scan(sim,self.guard,{'waypoint':[self.id]},lambda *_: self.fail('moved'))
        self.assertFalse(self.views.events)
        st['velocity']=[0.,0.,0.]
        self.guard.evidence[self.guard.body(np.zeros(3),np.eye(3))]=12
        self.views.scan(sim,self.guard,{'waypoint':[self.id]},lambda *_: self.fail('moved'))
        self.assertEqual(self.views.events[0]['status'],'scan_veto')

    def test_yaw_exception_propagates_actual_collision_on_first_step(self):
        st=dict(position=[0.,0.,0.],velocity=[0.,0.,0.],quaternion_wxyz=[1.,0.,0.,0.],
                euler_deg=[0.,0.,0.],angular_velocity=[0.,0.,0.])
        sim=SimpleNamespace(last_receipt={'state':st,'sim_time':0.},scene=self.rig)
        self.rig.width=160
        self.rig.height=120
        def advance(_, rotating):
            sim.last_receipt['sim_time']+=.05
            if self.views.events[-1]['phases'][-1]['phase']=='body_yaw':
                self.assertTrue(rotating)
                return 'collision'
        with patch.object(self.views,'target',return_value=(np.array([0.,1.,0.]),-7.,80.)), \
             patch('metric_flight.unknown_views.controller',return_value=(np.zeros(4),None)):
            reason=self.views.scan(sim,self.guard,{'side':[self.id]},advance)
        self.assertEqual(reason,'collision')
        self.assertEqual(self.views.events[-1]['phases'][-1]['duration_s'],.05)
        self.assertEqual(self.guard.evidence[tuple(self.id)],0)

    def test_rotated_camera_mount_covers_side_voxel_and_skips_blind_pivot(self):
        axes=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
        center=np.array([.152,.031,.012])
        self.rig.W[:3,:3]=axes.T
        self.rig.W[:3,3]=-axes.T@center
        self.rig.pitch_deg=0.
        self.rig.width,self.rig.height=160,120
        state={'position':[0.,0.,0.],'euler_deg':[0.,0.,0.]}
        def idx(p):
            return np.floor((p-self.guard.origin)/self.guard.cell).astype(int)
        self.assertIsNone(self.views.target({'pivot':[idx(np.zeros(3))]},self.guard,self.rig,state))
        side=np.array([0.,1.,0.])
        _,pitch,turn=self.views.target({'left':[idx(side)]},self.guard,self.rig,state)
        self.assertEqual(self.views.prediction['full_cells'],1)
        self.assertGreater(turn,0.)
        self.assertTrue(-60<=pitch<=30)
