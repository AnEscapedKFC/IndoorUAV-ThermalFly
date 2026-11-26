"""Transfer a simulator Z-up metric pose to the public IndoorUAV camera convention."""
import math

import numpy as np


def signed_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


class EpisodeCoordinates:
    """Choose an episode-local horizontal origin and heading, preserving scene height."""
    def __init__(self, spawn_xyz, spawn_yaw_deg, native_start_yaw_deg=0.):
        self.origin_xy = np.asarray(spawn_xyz, dtype=float)[:2].copy()
        self.heading = float(spawn_yaw_deg)
        self.native_heading = float(native_start_yaw_deg) % 360.
        angle = math.radians(self.native_heading)
        self.native_rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                         [math.sin(angle), math.cos(angle)]])
        yaw = math.radians(self.heading)
        self.forward = np.asarray([math.cos(yaw), math.sin(yaw)])
        self.left = np.asarray([-math.sin(yaw), math.cos(yaw)])

    def to_native(self, position, yaw_deg):
        xy = np.asarray(position, dtype=float)[:2] - self.origin_xy
        relative_yaw = signed_degrees(self.native_heading + self.heading - yaw_deg)
        # Floating point reconstruction of the identical spawn quaternion can differ by a few ulps.
        yaw = 0. if abs(relative_yaw) < 1e-5 else math.radians(relative_yaw) % math.tau
        planar = self.native_rotation @ np.array([-xy @ self.left, -xy @ self.forward])
        return np.r_[planar, position[2], yaw]

    def to_world(self, native):
        x, y, z, yaw = native
        right, backward = self.native_rotation.T @ np.array([x,y])
        xy = self.origin_xy - right * self.left - backward * self.forward
        return np.r_[xy, z, signed_degrees(self.heading + self.native_heading - math.degrees(yaw))]
