#!/usr/bin/env python3
"""Umeyama Sim(3) alignment for 3D point sets.

Computes scale, rotation, and translation to align source points to target points.
Based on: S. Umeyama, "Least-squares estimation of transformation parameters
between two point patterns", PAMI 1991.
"""

import numpy as np


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute Sim(3) alignment (s, R, t) from source to target.

    Minimises ||target - (s * source @ R.T + t)||^2.

    Args:
        source: (N, 3) float array of source points.
        target: (N, 3) float array of target points.

    Returns:
        s: scale factor (float).
        R: rotation matrix (3, 3).
        t: translation vector (3,).
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError(f"Shape mismatch: {source.shape} vs {target.shape}")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {source.shape}")
    if source.shape[0] < 3:
        raise ValueError("Need at least 3 points for alignment")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Alignment points must be finite")

    n = source.shape[0]

    # Centroids
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)

    # Variance of source
    sigma2_s = np.sum((source - mu_s) ** 2) / n
    if sigma2_s <= np.finfo(np.float64).eps:
        raise ValueError("Source points are degenerate")

    # Cross-covariance matrix
    cov = (target - mu_t).T @ (source - mu_s) / n

    # SVD
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    # Scale: s = 1/sigma2_s * trace(D * S)
    s = np.trace(np.diag(D) @ S) / sigma2_s

    # Translation
    t = mu_t - s * (R @ mu_s)

    return float(s), R, t


def transform_points(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply a Sim(3) transform returned by :func:`umeyama`."""
    points = np.asarray(points, dtype=np.float64)
    return scale * (points @ np.asarray(rotation, dtype=np.float64).T) + np.asarray(
        translation, dtype=np.float64
    )
