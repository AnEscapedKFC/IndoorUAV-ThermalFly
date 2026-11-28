"""Regression tests for evidence-based hover, optical flow, and door association."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import cv2

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parents[1])]
from metric_flight.observation_window import HoldTracker, DoorChair, hold_seconds


class WindowTest(unittest.TestCase):
    def setUp(self):
        self.rig=SimpleNamespace(width=200,height=120,R=np.eye(3),W=np.eye(4),s=1.,origin=np.zeros(3),
                                 ks={'rgb':np.array([[135.,0,100],[0,135.,60],[0,0,1]])})
        self.z=np.ones((120,200),np.float32)
        self.im=np.random.default_rng(21).random((120,200,3)).astype(np.float32)
        self.mask=np.zeros((120,200),bool)
        self.mask[25:95,55:145]=True

    def test_real_image_and_coherent_depth_required(self):
        tracker=HoldTracker(self.im,self.z,self.rig,[.25,.17,.75,.83],self.mask)
        self.assertIsNotNone(tracker.measure(self.im,self.z,self.rig))
        self.assertIsNone(tracker.measure(np.zeros_like(self.im),self.z,self.rig))
        self.assertIsNone(tracker.measure(self.im,2*self.z,self.rig))

    def test_moving_image_moves_box(self):
        tracker=HoldTracker(self.im,self.z,self.rig,[.25,.17,.75,.83],self.mask)
        shift=np.float32([[1,0,4],[0,1,0]])
        moved=cv2.warpAffine(self.im,shift,(200,120))
        fit=tracker.measure(moved,self.z,self.rig)
        self.assertAlmostEqual(fit[0],.27,places=2)

    def test_user_duration(self):
        self.assertEqual(hold_seconds("Observe the whole sofa, then hover."),1)
        self.assertEqual(hold_seconds("Hold the view for two seconds."),2)

    def test_door_identity_requires_context_at_first_sighting(self):
        selector=DoorChair()
        box=dict(xyxy=[.3,.4,.7,.85],side="center")
        det=lambda path,noun: dict(boxes=[box] if noun=="armchair" else [])
        self.assertFalse(selector.detect(det,None,self.rig,self.z)["matched"])
        self.assertTrue(selector.detect(lambda p,n:dict(boxes=[box]),None,self.rig,self.z)["matched"])
        self.assertTrue(selector.detect(det,None,self.rig,self.z)["matched"])

if __name__=="__main__":
    unittest.main()
