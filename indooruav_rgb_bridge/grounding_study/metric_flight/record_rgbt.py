"""Record paired calibrated first-person images from each physical receipt."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sim_bridge import CONTROL_DT, FpvVideoWriter, verify_video

# Forward and left translations in meters and yaw change in degrees, in each
# previous start's body frame. Constants are fixed before checking outcomes.
SHIFTS = {
    'room_inspect_ac': (.55, .30, -8.),
    'new_room_inspect_ac': (.82, -.26, 6.),
    'room_obj_ac_east_to_west': (.72, .29, -7.),
    'room_obj_ac_west_to_east': (.78, -.30, 8.),
    'holdout_room_obj_ac_east_to_west': (.46, -.34, 6.),
    'holdout_room_obj_ac_west_to_east': (.63, .32, -5.),
    'fresh_room_inspect_ac': (.42, -.40, 7.),
    'fresh_room_obj_ac_west_to_east': (.58, .35, -8.),
}


def shifted_spawn(xyz, yaw, shift):
    forward, left, dyaw = shift
    a = np.deg2rad(yaw)
    xyz = xyz.copy()
    xyz[:2] += [forward*np.cos(a)-left*np.sin(a), forward*np.sin(a)+left*np.cos(a)]
    return xyz, yaw + dyaw


class RgbtRecording:
    """Publish a synchronized recording after all encoders close and fully decode."""
    def __init__(self, out, scene_name="room65_selected_anchored"):
        self.scene_name = scene_name
        self.out = Path(out)
        self.stack = ExitStack()
        self.writers = {}
        self.count = 0

    def __enter__(self):
        self.temp = self.stack.enter_context(TemporaryDirectory(dir=self.out))
        self.stage = Path(self.temp)
        self.index = self.stack.enter_context((self.stage/'video_frames.jsonl').open('w'))
        self.encoders = self.stack.enter_context(ExitStack())
        return self

    def frame(self, sim, receipt):
        rgb = np.rint(sim.last_rgb * 255).astype(np.uint8)
        rig = sim.scene
        if not np.allclose(rig.Wir, rig.rig @ rig.W, rtol=0, atol=1e-9):
            raise ValueError('Thermal pose differs from the calibrated synchronized pose')
        ir = np.rint(rig.ir * 255).astype(np.uint8)
        if ir.shape != rgb.shape or receipt['frame_id'] != self.count:
            raise ValueError('Missing or misaligned image pair')
        frames = {'rgb_fpv': rgb, 'thermal_fpv': ir, 'rgb_thermal_fpv': np.concatenate((rgb, ir), axis=1)}
        for name, frame in frames.items():
            if self.count == 0:
                writer = FpvVideoWriter(self.stage/(name+'.mp4'),frame.shape[1],frame.shape[0],1/CONTROL_DT)
                self.encoders.callback(writer.close)
                self.writers[name] = writer
            self.writers[name].write(frame)
        transforms = dict(rgb_w2c=rig.W.tolist(), thermal_w2c=rig.Wir.tolist())
        print(json.dumps(dict(frame=self.count, t=receipt['sim_time'], step=receipt['step_id'],
                              rgb_sha256=hashlib.sha256(rgb).hexdigest(),
                              thermal_sha256=hashlib.sha256(ir).hexdigest(), **transforms)),file=self.index)
        self.count += 1

    def __exit__(self, typ, value, tb):
        try:
            if typ is None:
                self.encoders.close()
                self.index.close()
                checks = {name: verify_video(self.stage/(name+'.mp4'), self.count)
                          for name in self.writers}
                metadata = dict(fps=1/CONTROL_DT, frames=self.count, streams=checks,
                                thermal_source=f'{self.scene_name}/models/thermal.ply',
                                thermal_pose='W_thermal = rgb_to_thermal @ W_rgb',
                                montage='left RGB, right thermal', policy_input='RGB only')
                (self.stage/'recording.json').write_text(json.dumps(metadata, indent=2))
                for path in sorted(self.stage.iterdir(), key=lambda p:p.name=='recording.json'):
                    path.replace(self.out/path.name)
        finally:
            self.stack.close()
        return False
