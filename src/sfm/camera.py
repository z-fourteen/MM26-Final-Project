from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def estimate_simple_pinhole(width: int, height: int, focal_scale: float = 1.2) -> PinholeCamera:
    focal = focal_scale * float(max(width, height))
    return PinholeCamera(
        width=width,
        height=height,
        fx=focal,
        fy=focal,
        cx=width / 2.0,
        cy=height / 2.0,
    )


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    return K @ np.hstack([R.astype(np.float64), t])


def project_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, points3d: np.ndarray) -> np.ndarray:
    points3d = np.asarray(points3d, dtype=np.float64)
    camera_points = (R @ points3d.T + np.asarray(t, dtype=np.float64).reshape(3, 1)).T
    projected = (K @ camera_points.T).T
    return projected[:, :2] / projected[:, 2:3]


def camera_depths(R: np.ndarray, t: np.ndarray, points3d: np.ndarray) -> np.ndarray:
    points3d = np.asarray(points3d, dtype=np.float64)
    camera_points = (R @ points3d.T + np.asarray(t, dtype=np.float64).reshape(3, 1)).T
    return camera_points[:, 2]


def reprojection_errors(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    points3d: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    projected = project_points(K, R, t, points3d)
    return np.linalg.norm(projected - observations, axis=1)
