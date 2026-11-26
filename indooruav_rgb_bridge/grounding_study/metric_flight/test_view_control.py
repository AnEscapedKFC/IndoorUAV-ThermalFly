"""Acceptance boundaries, instruction-conditioned height and stationary envelope regressions."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parents[1])]
from metric_flight.depth_guard import DepthGuard, survey_rig
from metric_flight.view_control import PITCH_LIMITS, VIEW_BANDS, body_height_delta, framing, pointing_request, view_intent


class CoordinatedViewTest(unittest.TestCase):
    def test_area_requires_full_centered_box(self):
        for box in ([.25,.3,.75,.7],[.1,.125,.9,.875]):
            self.assertTrue(framing(box)["framing_ok"])
        for box in (None,[.25,.31,.75,.69],[.10,.10,.90,.90],[0.,.25,.55,.75]):
            self.assertFalse(framing(box)["framing_ok"])

    def test_semantics_and_geometry_choose_height(self):
        point=np.array([2.,0.,1.9])
        camera=np.array([0.,0.,1.7])
        for noun in ("air conditioner","laptop"):
            band=VIEW_BANDS[view_intent("Look horizontally at the "+noun)]
            self.assertAlmostEqual(body_height_delta(point,camera,1.,band,11.),.2)
            band=VIEW_BANDS[view_intent("Inspect the "+noun+" from below")]
            self.assertLess(body_height_delta(point,camera,1.,band,11.),0)
        self.assertGreater(body_height_delta(point,camera,1.,VIEW_BANDS["above"],-20.),.2)
        self.assertEqual(view_intent("Inspect the air conditioner"),"unconstrained")

    def test_hard_limits_apply_to_every_rig_write(self):
        rig=survey_rig(object,PITCH_LIMITS)()
        for requested,expected in ((-70,-60),(0,0),(90,30)):
            rig.pitch_deg=requested
            self.assertEqual(rig.pitch_deg,expected)

    def test_pitch_tracks_current_pixel_bearing_with_mount_offset(self):
        # Horizontal camera with optical right=-Y, down=-Z and forward=+X.
        axes=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
        rig=SimpleNamespace(R=axes,W=np.eye(4),origin=np.zeros(3),s=1.,
                            ks={"rgb":np.eye(3)},pitch_deg=-8.)
        self.assertAlmostEqual(pointing_request(rig,np.array([2.,0.,0.])), -8.)
        self.assertAlmostEqual(pointing_request(rig,np.array([1.,0.,-1.])), -53.)
        self.assertAlmostEqual(pointing_request(rig,np.array([1.,0.,1.])), 30.)

    def test_static_gate_distinguishes_unknown_and_surface(self):
        guard=DepthGuard(np.zeros(3),[.35,.35,.11])
        st=dict(position=[0.,0.,0.],quaternion_wxyz=[1.,0.,0.,0.])
        self.assertEqual(guard.stationary(st)[0]["status"],"unknown")
        guard.seed_body(np.zeros(3),np.eye(3))
        self.assertEqual(guard.stationary(st)[0]["status"],"free")
        ids=guard.body(np.zeros(3),np.eye(3))
        guard.evidence[tuple(i[0] for i in ids)]=1
        result,cells=guard.stationary(st)
        self.assertEqual((result["status"],len(cells)),("surface",1))


if __name__=="__main__":
    unittest.main()
