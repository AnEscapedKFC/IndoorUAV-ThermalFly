"""Regression checks for causal RGB-D evidence and keyframe synchronization."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metric_flight.observation_memory import ObservationMemory


class MemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mem = ObservationMemory(
            Path(self.tmp.name), "laptop", lambda rgb, box: rgb.min(axis=2) < 120
        )
        self.k = np.array([[100.0, 0, 60.0], [0, 100.0, 45.0], [0, 0, 1.0]])
        self.image = np.full((90, 120, 3), 0.7, np.float32)
        cv2.rectangle(self.image, (24, 17), (95, 72), (0.1, 0.12, 0.13), -1)
        cv2.rectangle(self.image, (29, 21), (90, 43), (0.25, 0.4, 0.8), -1)
        for x in range(30, 87, 6):
            cv2.line(self.image, (x, 54), (x, 64), (0.6, 0.6, 0.6), 2)

    def observe(self, x, depth=2.0, detect=True):
        i = len(self.mem.records)
        shift = round(-100 * x / 2.0)
        image = cv2.warpAffine(
            self.image,
            np.float32([[1, 0, shift], [0, 1, 0]]),
            (120, 90),
            borderMode=cv2.BORDER_REPLICATE,
        )
        box = [(23 + shift) / 120, 16 / 90, (96 + shift) / 120, 73 / 90]
        w = np.eye(4)
        w[0, 3] = -x
        rig = SimpleNamespace(
            R=np.eye(3),
            s=1.0,
            origin=np.zeros(3),
            W=w,
            ks={"rgb": self.k},
            pitch_deg=-5.0,
        )
        return self.mem.observe(
            image,
            np.full((90, 120), depth, np.float32),
            rig,
            [{"xyxy": box}] if detect else [],
            100 + i,
            0.5 * i,
            [x, 0.0, 0.0],
        )[0]

    def test_parallax_required_and_consistent_view_adds_quality(self):
        first = self.observe(0.0)
        self.assertIn("initial", first["reasons"])
        duplicate = self.observe(0.0)
        self.assertIsNone(duplicate["pair"])
        self.assertEqual(duplicate["reasons"], [])
        moved = self.observe(0.10)
        self.assertEqual(moved["pair"], 0)
        self.assertGreater(moved["matched_edge_pixels"], 0)
        self.assertIn("detail_improvement", moved["reasons"])
        self.mem.finish()
        folder = self.mem.root
        self.assertEqual(
            json.loads((folder / "index.json").read_text())["frames"][-1][
                "receipt_frame"
            ],
            102,
        )
        self.assertFalse((folder / "0001_rgb.png").exists())
        table = np.load(folder / "surfaces.npz")
        self.assertGreater(table["best_quality"].max(), 0)

    def test_nearer_surface_occludes_and_bad_depth_cannot_verify_sharp_rgb(self):
        self.observe(0.0)
        blocked = self.observe(0.10, depth=1.0)
        self.assertEqual(blocked["matched_edge_pixels"], 0)
        self.assertGreater(blocked["occluded_cells"], 0)
        missing = self.observe(0.10, depth=np.nan, detect=False)
        self.assertEqual(missing["occluded_cells"], 0)
        self.assertEqual(missing["reasons"], [])


if __name__ == "__main__":
    unittest.main()
