"""Check camera-limit and detail decisions without a simulator or pretrained weights."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metric_flight.history_control import HistoryInspection


class FramingTest(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        with patch("metric_flight.history_control.instance_mask", return_value=None):
            self.servo = HistoryInspection(
                folder.name, "laptop", "Inspect its screen and keyboard."
            )
        self.rig = SimpleNamespace(
            R=np.eye(3),
            s=1.0,
            origin=np.zeros(3),
            W=np.eye(4),
            ks={"rgb": np.array([[450, 0, 320], [0, 450, 240], [0, 0, 1]])},
            width=640,
            height=480,
            pitch_deg=-62.0,
        )
        self.info = dict(
            id=0,
            part_pixels=[0, 0, 200, 200],
            matched_edge_pixels=0,
            consistent_pixels=0,
            quality=0.0,
            occluded_cells=0,
            visible_cells=10,
        )
        self.pt = np.array([0.0, 0.45, 2.0])
        self.servo.memory.observe = lambda *args: (self.info, self.pt)

    def step(self, width):
        box = [0.5 - width / 2, 0.29, 0.5 + width / 2, 0.72]
        return self.servo.step(
            np.zeros(3),
            0.0,
            np.zeros(3),
            np.zeros(4),
            {"boxes": [{"xyxy": box}]},
            self.rig,
            0.5,
            None,
            None,
            receipt={"frame_id": 20, "sim_time": 1.0},
        )

    def test_gimbal_at_downward_limit_requests_lower_body(self):
        goal, stop, info = self.step(0.18)
        self.assertFalse(stop)
        self.assertLess(goal[2], 0.0)
        self.assertEqual(info["pitch_request"], -62.0)

    def test_detail_can_be_sufficient_before_nominal_image_width(self):
        self.pt = np.array([0.0, 0.02, 2.0])
        self.rig.pitch_deg = -32.0
        self.info.update(
            part_pixels=[0, 0, 7000, 6500],
            matched_edge_pixels=75,
            consistent_pixels=1500,
        )
        for _ in range(3):
            _, stop, info = self.step(0.34)
        self.assertTrue(stop)
        self.assertEqual(info["phase"], "terminal_hold")


if __name__ == "__main__":
    unittest.main()
