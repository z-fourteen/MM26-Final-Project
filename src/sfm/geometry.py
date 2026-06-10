from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class GeometricVerificationResult:
    image_name1: str
    image_name2: str
    raw_match_path: Path
    feature_path1: Path
    feature_path2: Path
    fundamental_matrix: np.ndarray
    inlier_matches: np.ndarray
    inlier_mask: np.ndarray
    num_raw_matches: int
    num_inliers: int
    inlier_ratio: float
    status: str


def verify_fundamental_matrix(
    match_path: Path,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    min_num_matches: int,
    min_num_inliers: int,
    ransac_reproj_threshold_px: float,
    ransac_confidence: float,
) -> GeometricVerificationResult:
    match_data = np.load(match_path)
    matches = match_data["matches"].astype(np.int32, copy=False)
    image_name1 = str(match_data["image_name1"])
    image_name2 = str(match_data["image_name2"])
    feature_path1 = Path(str(match_data["feature_path1"]))
    feature_path2 = Path(str(match_data["feature_path2"]))

    num_raw_matches = int(matches.shape[0])
    empty_f = np.full((3, 3), np.nan, dtype=np.float64)
    empty_mask = np.zeros((num_raw_matches,), dtype=bool)

    if num_raw_matches < min_num_matches:
        return GeometricVerificationResult(
            image_name1=image_name1,
            image_name2=image_name2,
            raw_match_path=match_path,
            feature_path1=feature_path1,
            feature_path2=feature_path2,
            fundamental_matrix=empty_f,
            inlier_matches=np.empty((0, 2), dtype=np.int32),
            inlier_mask=empty_mask,
            num_raw_matches=num_raw_matches,
            num_inliers=0,
            inlier_ratio=0.0,
            status="too_few_matches",
        )

    points1 = keypoints1[matches[:, 0], :2].astype(np.float32, copy=False)
    points2 = keypoints2[matches[:, 1], :2].astype(np.float32, copy=False)
    fundamental_matrix, mask = cv2.findFundamentalMat(
        points1,
        points2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold_px,
        confidence=ransac_confidence,
    )

    if fundamental_matrix is None or mask is None or fundamental_matrix.shape != (3, 3):
        return GeometricVerificationResult(
            image_name1=image_name1,
            image_name2=image_name2,
            raw_match_path=match_path,
            feature_path1=feature_path1,
            feature_path2=feature_path2,
            fundamental_matrix=empty_f,
            inlier_matches=np.empty((0, 2), dtype=np.int32),
            inlier_mask=empty_mask,
            num_raw_matches=num_raw_matches,
            num_inliers=0,
            inlier_ratio=0.0,
            status="estimation_failed",
        )

    inlier_mask = mask.ravel().astype(bool)
    inlier_matches = matches[inlier_mask]
    num_inliers = int(inlier_matches.shape[0])
    inlier_ratio = num_inliers / float(max(num_raw_matches, 1))
    status = "verified" if num_inliers >= min_num_inliers else "low_inliers"

    return GeometricVerificationResult(
        image_name1=image_name1,
        image_name2=image_name2,
        raw_match_path=match_path,
        feature_path1=feature_path1,
        feature_path2=feature_path2,
        fundamental_matrix=fundamental_matrix.astype(np.float64, copy=False),
        inlier_matches=inlier_matches.astype(np.int32, copy=False),
        inlier_mask=inlier_mask,
        num_raw_matches=num_raw_matches,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        status=status,
    )


def save_verification_result(result: GeometricVerificationResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image_name1=result.image_name1,
        image_name2=result.image_name2,
        raw_match_path=str(result.raw_match_path),
        feature_path1=str(result.feature_path1),
        feature_path2=str(result.feature_path2),
        F=result.fundamental_matrix,
        inlier_matches=result.inlier_matches,
        inlier_mask=result.inlier_mask,
        num_raw_matches=np.int32(result.num_raw_matches),
        num_inliers=np.int32(result.num_inliers),
        inlier_ratio=np.float32(result.inlier_ratio),
        status=result.status,
    )


def verified_output_path(verified_dir: Path, match_path: Path) -> Path:
    return verified_dir / match_path.name
