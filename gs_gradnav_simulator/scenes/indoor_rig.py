"""Render an indoor RGB-T bundle in the metric frame of its collision map."""
from pathlib import Path
import json

import numpy as np
import torch

from scene import GaussianScene
from scenes.room65_selected_anchored.room import NativeModel, RigScene, intrinsic, tensor
from scenes.corridor_native.room import KnownSpace


class BundleModel(NativeModel):
    def __init__(self, path, width, height, device):
        GaussianScene.__init__(self, path, width, height, device=device)


class IndoorRigScene(RigScene):
    def __init__(self, scene_path, width, height, up_axis, scale, device, *, render_thermal=True):
        path = Path(scene_path)
        if path.name != "rgb.ply" or path.parent.name != "models" or up_axis != "z" or scale != 1.:
            raise ValueError("Use models/rgb.ply and the bundle's Z-up collision map at scale 1")
        root = path.parent.parent
        self.name = root.name
        read = lambda name: json.loads((root / "inputs" / name).read_text())
        frame = read("frame.json")
        self.R, self.origin = np.array(frame["rotation"]), np.array(frame["origin"])
        self.s, self.bounds = frame["meters_per_source_unit"], np.array(frame["bounds_m"])
        self.width, self.height = width, round(width * 3 / 4)
        self.device = torch.device(device)
        self.rgb = BundleModel(path, width, self.height, device)
        self.thermal = None
        rows = read("manifest.json")["views"]
        ref = read("source.json")["reference_view_id"]
        rgb = next(r for r in rows if r["view_id"] == ref)
        self.ks = {"rgb": intrinsic(rgb, width, self.height)}
        self.K = tensor(self.ks["rgb"])[None]
        if render_thermal:
            ir = next(r for r in rows if r.get("paired_rgb_view_id") == ref and r["paired"])
            self.thermal = BundleModel(path.with_name("thermal.ply"), width, self.height, device)
            self.ks["thermal"] = intrinsic(ir, width, self.height)
            self.rig = np.array(read("rig.json")["rgb_to_thermal"])

    def metadata(self):
        return {"name": self.name, "meters_per_source_unit": self.s,
                "bounds_m": self.bounds.tolist(), "thermal_enabled": self.thermal is not None}


class IndoorKnownSpace(KnownSpace):
    def __init__(self, scene_dir):
        with np.load(Path(scene_dir) / "collision/free_volume.npz") as grid:
            self.free = grid["free"]
            self.lo = grid["origin"]
            self.step = float(grid["step"])
