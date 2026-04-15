"""Jacobian math utilities for the mobile manipulator.

This module provides utilities for building the end-effector Jacobian and
computing a Moore-Penrose pseudo-inverse.

The kinematic model assumes a planar 2-DOF arm mounted on a 3-DOF mobile base
(x, y, z) with yaw rotation. The end-effector position is assumed to be in the
world frame and the arm is assumed to lie in the world XY plane.

The Jacobian maps joint velocities (base + arm) to end-effector linear velocity.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


def J_mobile(links: Sequence[float], theta0: float, theta1: float, yaw: float) -> np.ndarray:
    """Compute the 3x6 Jacobian for a 2-DOF planar arm mounted on a mobile base.

    Args:
        links: Link lengths [l0, l1].
        theta0: First joint angle (rad).
        theta1: Second joint angle (rad).
        yaw: Yaw of the mobile base (rad).

    Returns:
        A 3x6 Jacobian mapping [vx, vy, vz, omega_z, theta0_dot, theta1_dot]
        to the end-effector linear velocity in the world frame.
    """

    l0, l1 = links

    # Compute end-effector position in the base frame (arm plane (XY)).
    px_body = l0 * np.cos(theta0) + l1 * np.cos(theta0 + theta1)
    py_body = l0 * np.sin(theta0) + l1 * np.sin(theta0 + theta1)

    # Derivatives of body-frame arm position w.r.t joint angles.
    dpx0_body = -l0 * np.sin(theta0) - l1 * np.sin(theta0 + theta1)
    dpx1_body = -l1 * np.sin(theta0 + theta1)
    dpy0_body = l0 * np.cos(theta0) + l1 * np.cos(theta0 + theta1)
    dpy1_body = l1 * np.cos(theta0 + theta1)

    # Rotation from body frame to world frame (yaw)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Derivatives of the body->world rotation w.r.t yaw.
    dpx_dyaw = -sy * px_body - cy * py_body
    dpy_dyaw = cy * px_body - sy * py_body

    # Convert joint derivatives to world frame
    dpx0 = cy * dpx0_body - sy * dpy0_body
    dpx1 = cy * dpx1_body - sy * dpy1_body
    dpy0 = sy * dpx0_body + cy * dpy0_body
    dpy1 = sy * dpx1_body + cy * dpy1_body

    # Build Jacobian (3 rows x 6 columns)
    # Columns correspond to: [vx, vy, vz, omega_z, theta0_dot, theta1_dot]
    J = np.zeros((3, 6), dtype=float)

    # Base translation contributions
    J[0, 0] = 1.0
    J[1, 1] = 1.0
    J[2, 2] = 1.0

    # Yaw contribution
    J[0, 3] = dpx_dyaw
    J[1, 3] = dpy_dyaw

    # Arm joint contributions
    J[0, 4] = dpx0
    J[0, 5] = dpx1
    J[1, 4] = dpy0
    J[1, 5] = dpy1

    return J


def JMoore(J: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    """Compute the Moore-Penrose pseudo-inverse of a matrix."""

    return np.linalg.pinv(J, rcond=rcond)


def Jinv(J: np.ndarray) -> np.ndarray:
    """Compute a (pseudo-)inverse of a square matrix.

    This is a thin wrapper around numpy.linalg.inv with fallback to pseudo-inverse.
    """

    try:
        return np.linalg.inv(J)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(J)
