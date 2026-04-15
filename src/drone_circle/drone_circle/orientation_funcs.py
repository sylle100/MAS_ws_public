"""Orientation conversion utilities.

This module provides helper functions for converting between Euler angles
(roll, pitch, yaw) and quaternions.

The conventions follow ROS/TF (ZYX / yaw-pitch-roll) and assume quaternions are
in the form [x, y, z, w].
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple


def euler_from_quaternion(quat: Iterable[float]) -> Tuple[float, float, float]:
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat: A 4-element iterable in the order [x, y, z, w].

    Returns:
        (roll, pitch, yaw) in radians.
    """

    x, y, z, w = quat

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert Euler angles (roll, pitch, yaw) to quaternion.

    Args:
        roll: rotation around x-axis in radians.
        pitch: rotation around y-axis in radians.
        yaw: rotation around z-axis in radians.

    Returns:
        Quaternion as (x, y, z, w).
    """

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w
