"""Regressions for unobserved space and the part of a waypoint after the endpoint."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parents[1])]
from metric_flight.depth_guard import DepthGuard
from metric_flight.depth_guidance import ObservationAnchor


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.guard=DepthGuard(np.zeros(3),[.10,.08,.04])
        self.rig=SimpleNamespace(R=np.eye(3),W=np.eye(4),s=1.,origin=np.zeros(3),
                                 ks={'rgb':np.array([[80,0,80],[0,80,60],[0,0,1.]])})
        self.st=dict(position=[0.,0.,0.],velocity=[.48,0.,0.],euler_deg=[0.,0.,0.],
                     quaternion_wxyz=[1.,0.,0.,0.],angular_velocity=[0.,0.,0.])

    def cell(self,xyz):
        return tuple(np.floor((xyz-self.guard.origin)/self.guard.cell).astype(int))

    def test_free_surface_and_occluded_space(self):
        self.guard.observe(np.ones((120,160)),self.rig)
        self.assertLess(self.guard.evidence[self.cell([0.,0.,.6])],0)
        self.assertGreater(self.guard.evidence[self.cell([0.,0.,1.])],0)
        self.assertEqual(self.guard.evidence[self.cell([0.,0.,1.3])],0)

    def test_braking_sweep(self):
        g=np.array([.24,0.,0.,0.]);v=np.array([.48,0.,0.])
        guard=self.guard
        self.assertEqual(guard.check(self.st,g,.5,v,v),'unknown')
        guard.evidence[:]=-9
        self.assertEqual(guard.check(self.st,g,.5,v,v),'free')
        cell=self.cell([.53,0.,0.]);guard.evidence[cell]=8
        self.assertLess(guard.evidence[guard.body(g[:3],np.eye(3))].max(),0)
        self.assertEqual(guard.check(self.st,g,.5,v,v),'surface')

    def test_cropped_box_keeps_physical_anchor(self):
        camera=self.rig;z=np.ones((120,160));rgb=np.zeros((120,160,3),dtype=np.float32)
        rgb[58:60]=1.
        anchor=ObservationAnchor('laptop')
        first=anchor.update(rgb,z,[.2,.2,.8,.8],camera)
        again=anchor.update(rgb,z,[0.,.1,1.,.9],camera)
        self.assertTrue(first['confirmed'] and again['confirmed'])
        self.assertEqual(first['world'],again['world'])
        self.assertEqual(again['views'],1)
        self.assertEqual(again['source'],'reprojected')

if __name__=='__main__':unittest.main()
