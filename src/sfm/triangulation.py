from __future__ import annotations

import numpy as np


def triangulate_point_dlt(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    u1, v1 = float(x1[0]), float(x1[1])
    u2, v2 = float(x2[0]), float(x2[1])
    A = np.array(
        [
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1],
        ],
        dtype=np.float64,
    )
    _, _, vh = np.linalg.svd(A)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12:
        return np.full((3,), np.nan, dtype=np.float64)
    return homogeneous[:3] / homogeneous[3]


def triangulate_points_dlt(P1: np.ndarray, P2: np.ndarray, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    if points1.shape != points2.shape:
        raise ValueError(f"Point arrays must have the same shape, got {points1.shape} and {points2.shape}")
    points3d = [
        triangulate_point_dlt(P1, P2, x1, x2)
        for x1, x2 in zip(points1, points2)
    ]
    return np.asarray(points3d, dtype=np.float64)
