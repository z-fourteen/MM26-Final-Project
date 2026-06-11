from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.sfm.camera import PinholeCamera, projection_matrix
from src.sfm.triangulation import triangulate_points_dlt


@dataclass(frozen=True)
class GeometricVerificationResult:
    image_name1: str
    image_name2: str
    raw_match_path: Path
    feature_path1: Path
    feature_path2: Path
    fundamental_matrix: np.ndarray
    homography_matrix: np.ndarray
    essential_matrix: np.ndarray
    inlier_matches: np.ndarray
    inlier_mask: np.ndarray
    homography_inlier_matches: np.ndarray
    homography_inlier_mask: np.ndarray
    essential_inlier_matches: np.ndarray
    essential_inlier_mask: np.ndarray
    similarity_inlier_matches: np.ndarray
    similarity_inlier_mask: np.ndarray
    num_raw_matches: int
    num_inliers: int
    num_h_inliers: int
    num_e_inliers: int
    num_s_inliers: int
    inlier_ratio: float
    h_inlier_ratio: float
    e_inlier_ratio: float
    s_inlier_ratio: float
    homography_ratio: float
    essential_ratio: float
    similarity_ratio: float
    median_triangulation_angle_deg: float
    model_type: str
    calibration_status: str
    status: str


def verify_geometric_models(
    match_path: Path,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    camera1: PinholeCamera,
    camera2: PinholeCamera,
    min_num_matches: int,
    min_num_inliers: int,
    ransac_reproj_threshold_px: float,
    homography_reproj_threshold_px: float,
    essential_reproj_threshold_px: float,
    ransac_confidence: float,
    homography_ratio_threshold: float,
    essential_ratio_threshold: float,
    panoramic_triangulation_angle_deg: float,
    wtf_border_ratio: float,
    wtf_similarity_ratio_threshold: float,
) -> GeometricVerificationResult:
    match_data = np.load(match_path)
    matches = match_data["matches"].astype(np.int32, copy=False)
    image_name1 = str(match_data["image_name1"])
    image_name2 = str(match_data["image_name2"])
    feature_path1 = Path(str(match_data["feature_path1"]))
    feature_path2 = Path(str(match_data["feature_path2"]))

    num_raw_matches = int(matches.shape[0])
    empty_f = np.full((3, 3), np.nan, dtype=np.float64)
    empty_h = np.full((3, 3), np.nan, dtype=np.float64)
    empty_e = np.full((3, 3), np.nan, dtype=np.float64)
    empty_mask = np.zeros((num_raw_matches,), dtype=bool)

    if num_raw_matches < min_num_matches:
        return _make_result(
            image_name1=image_name1,
            image_name2=image_name2,
            raw_match_path=match_path,
            feature_path1=feature_path1,
            feature_path2=feature_path2,
            fundamental_matrix=empty_f,
            num_raw_matches=num_raw_matches,
            empty_mask=empty_mask,
            status="too_few_matches",
            model_type="unverified",
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
        return _make_result(
            image_name1=image_name1,
            image_name2=image_name2,
            raw_match_path=match_path,
            feature_path1=feature_path1,
            feature_path2=feature_path2,
            fundamental_matrix=empty_f,
            num_raw_matches=num_raw_matches,
            empty_mask=empty_mask,
            status="estimation_failed",
            model_type="unverified",
        )

    inlier_mask = mask.ravel().astype(bool)
    inlier_matches = matches[inlier_mask]
    num_inliers = int(inlier_matches.shape[0])
    inlier_ratio = num_inliers / float(max(num_raw_matches, 1))

    homography_matrix, homography_mask = cv2.findHomography(
        points1,
        points2,
        method=cv2.RANSAC,
        ransacReprojThreshold=homography_reproj_threshold_px,
        confidence=ransac_confidence,
    )
    if homography_matrix is None or homography_mask is None:
        homography_matrix = empty_h
        h_inlier_mask = empty_mask.copy()
    else:
        h_inlier_mask = homography_mask.ravel().astype(bool)
    h_inlier_matches = matches[h_inlier_mask]
    num_h_inliers = int(h_inlier_matches.shape[0])
    h_inlier_ratio = num_h_inliers / float(max(num_raw_matches, 1))
    homography_ratio = num_h_inliers / float(max(num_inliers, 1))

    essential_matrix, essential_mask = cv2.findEssentialMat(
        points1,
        points2,
        cameraMatrix=camera1.K,
        method=cv2.RANSAC,
        prob=ransac_confidence,
        threshold=essential_reproj_threshold_px,
    )
    if essential_matrix is None or essential_mask is None or essential_matrix.shape != (3, 3):
        essential_matrix = empty_e
        e_inlier_mask = empty_mask.copy()
    else:
        e_inlier_mask = essential_mask.ravel().astype(bool)
    e_inlier_matches = matches[e_inlier_mask]
    num_e_inliers = int(e_inlier_matches.shape[0])
    e_inlier_ratio = num_e_inliers / float(max(num_raw_matches, 1))
    essential_ratio = num_e_inliers / float(max(num_inliers, 1))
    calibration_status = "calibrated" if essential_ratio >= essential_ratio_threshold else "uncertain"

    s_inlier_mask = estimate_border_similarity_inliers(
        points1=points1,
        points2=points2,
        camera1=camera1,
        camera2=camera2,
        border_ratio=wtf_border_ratio,
        reproj_threshold_px=ransac_reproj_threshold_px,
        confidence=ransac_confidence,
    )
    s_inlier_matches = matches[s_inlier_mask]
    num_s_inliers = int(s_inlier_matches.shape[0])
    s_inlier_ratio = num_s_inliers / float(max(num_raw_matches, 1))
    similarity_ratio = num_s_inliers / float(max(num_inliers, 1))

    median_angle = 0.0
    if num_e_inliers >= min_num_inliers and np.isfinite(essential_matrix).all():
        median_angle = estimate_median_triangulation_angle(
            essential_matrix=essential_matrix,
            camera1=camera1,
            camera2=camera2,
            points1=points1[e_inlier_mask].astype(np.float64),
            points2=points2[e_inlier_mask].astype(np.float64),
        )

    if num_inliers < min_num_inliers:
        status = "low_inliers"
        model_type = "unverified"
    elif similarity_ratio >= wtf_similarity_ratio_threshold:
        status = "rejected_wtf"
        model_type = "rejected_wtf"
    elif calibration_status == "calibrated" and median_angle <= panoramic_triangulation_angle_deg:
        status = "verified_panoramic"
        model_type = "panoramic"
    elif homography_ratio >= homography_ratio_threshold:
        status = "verified_planar"
        model_type = "planar"
    else:
        status = "verified"
        model_type = "general"

    return GeometricVerificationResult(
        image_name1=image_name1,
        image_name2=image_name2,
        raw_match_path=match_path,
        feature_path1=feature_path1,
        feature_path2=feature_path2,
        fundamental_matrix=fundamental_matrix.astype(np.float64, copy=False),
        homography_matrix=homography_matrix.astype(np.float64, copy=False),
        essential_matrix=essential_matrix.astype(np.float64, copy=False),
        inlier_matches=inlier_matches.astype(np.int32, copy=False),
        inlier_mask=inlier_mask,
        homography_inlier_matches=h_inlier_matches.astype(np.int32, copy=False),
        homography_inlier_mask=h_inlier_mask,
        essential_inlier_matches=e_inlier_matches.astype(np.int32, copy=False),
        essential_inlier_mask=e_inlier_mask,
        similarity_inlier_matches=s_inlier_matches.astype(np.int32, copy=False),
        similarity_inlier_mask=s_inlier_mask,
        num_raw_matches=num_raw_matches,
        num_inliers=num_inliers,
        num_h_inliers=num_h_inliers,
        num_e_inliers=num_e_inliers,
        num_s_inliers=num_s_inliers,
        inlier_ratio=inlier_ratio,
        h_inlier_ratio=h_inlier_ratio,
        e_inlier_ratio=e_inlier_ratio,
        s_inlier_ratio=s_inlier_ratio,
        homography_ratio=homography_ratio,
        essential_ratio=essential_ratio,
        similarity_ratio=similarity_ratio,
        median_triangulation_angle_deg=median_angle,
        model_type=model_type,
        calibration_status=calibration_status,
        status=status,
    )


def verify_fundamental_matrix(
    match_path: Path,
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    min_num_matches: int,
    min_num_inliers: int,
    ransac_reproj_threshold_px: float,
    ransac_confidence: float,
) -> GeometricVerificationResult:
    points = np.vstack([keypoints1[:, :2], keypoints2[:, :2]])
    max_x = float(np.nanmax(points[:, 0])) if points.size else 1.0
    max_y = float(np.nanmax(points[:, 1])) if points.size else 1.0
    camera = PinholeCamera(
        width=int(max(max_x + 1.0, 1.0)),
        height=int(max(max_y + 1.0, 1.0)),
        fx=max(max_x, max_y, 1.0),
        fy=max(max_x, max_y, 1.0),
        cx=max_x / 2.0,
        cy=max_y / 2.0,
    )
    return verify_geometric_models(
        match_path=match_path,
        keypoints1=keypoints1,
        keypoints2=keypoints2,
        camera1=camera,
        camera2=camera,
        min_num_matches=min_num_matches,
        min_num_inliers=min_num_inliers,
        ransac_reproj_threshold_px=ransac_reproj_threshold_px,
        homography_reproj_threshold_px=ransac_reproj_threshold_px,
        essential_reproj_threshold_px=ransac_reproj_threshold_px,
        ransac_confidence=ransac_confidence,
        homography_ratio_threshold=0.8,
        essential_ratio_threshold=0.6,
        panoramic_triangulation_angle_deg=1.0,
        wtf_border_ratio=0.1,
        wtf_similarity_ratio_threshold=0.7,
    )


def estimate_border_similarity_inliers(
    points1: np.ndarray,
    points2: np.ndarray,
    camera1: PinholeCamera,
    camera2: PinholeCamera,
    border_ratio: float,
    reproj_threshold_px: float,
    confidence: float,
) -> np.ndarray:
    border_mask = border_point_mask(points1, camera1, border_ratio) & border_point_mask(points2, camera2, border_ratio)
    full_mask = np.zeros((points1.shape[0],), dtype=bool)
    if int(border_mask.sum()) < 4:
        return full_mask
    similarity, mask = cv2.estimateAffinePartial2D(
        points1[border_mask],
        points2[border_mask],
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold_px,
        confidence=confidence,
    )
    if similarity is None or mask is None:
        return full_mask
    border_indices = np.flatnonzero(border_mask)
    full_mask[border_indices[mask.ravel().astype(bool)]] = True
    return full_mask


def border_point_mask(points: np.ndarray, camera: PinholeCamera, border_ratio: float) -> np.ndarray:
    x_margin = max(float(camera.width) * border_ratio, 1.0)
    y_margin = max(float(camera.height) * border_ratio, 1.0)
    xs = points[:, 0]
    ys = points[:, 1]
    return (
        (xs <= x_margin)
        | (xs >= float(camera.width) - x_margin)
        | (ys <= y_margin)
        | (ys >= float(camera.height) - y_margin)
    )


def estimate_median_triangulation_angle(
    essential_matrix: np.ndarray,
    camera1: PinholeCamera,
    camera2: PinholeCamera,
    points1: np.ndarray,
    points2: np.ndarray,
) -> float:
    if points1.shape[0] < 2:
        return 0.0
    try:
        _num_pose_inliers, R, t, pose_mask = cv2.recoverPose(
            essential_matrix,
            points1,
            points2,
            camera1.K,
        )
    except cv2.error:
        return 0.0
    if pose_mask is None:
        pose_inlier_mask = np.ones((points1.shape[0],), dtype=bool)
    else:
        pose_inlier_mask = pose_mask.ravel().astype(bool)
    points1 = points1[pose_inlier_mask]
    points2 = points2[pose_inlier_mask]
    if points1.shape[0] < 2:
        return 0.0

    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros(3, dtype=np.float64)
    R2 = R.astype(np.float64)
    t2 = t.reshape(3).astype(np.float64)
    P1 = projection_matrix(camera1.K, R1, t1)
    P2 = projection_matrix(camera2.K, R2, t2)
    points3d = triangulate_points_dlt(P1, P2, points1, points2)
    finite_mask = np.isfinite(points3d).all(axis=1)
    points3d = points3d[finite_mask]
    if points3d.shape[0] == 0:
        return 0.0
    center1 = np.zeros((3,), dtype=np.float64)
    center2 = -R2.T @ t2
    rays1 = points3d - center1.reshape(1, 3)
    rays2 = points3d - center2.reshape(1, 3)
    norms1 = np.linalg.norm(rays1, axis=1)
    norms2 = np.linalg.norm(rays2, axis=1)
    valid = (norms1 > 1e-12) & (norms2 > 1e-12)
    if not np.any(valid):
        return 0.0
    cosines = np.sum(rays1[valid] * rays2[valid], axis=1) / (norms1[valid] * norms2[valid])
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return float(np.median(angles)) if angles.size else 0.0


def _make_result(
    image_name1: str,
    image_name2: str,
    raw_match_path: Path,
    feature_path1: Path,
    feature_path2: Path,
    fundamental_matrix: np.ndarray,
    num_raw_matches: int,
    empty_mask: np.ndarray,
    status: str,
    model_type: str,
) -> GeometricVerificationResult:
    empty_matches = np.empty((0, 2), dtype=np.int32)
    empty_matrix = np.full((3, 3), np.nan, dtype=np.float64)
    return GeometricVerificationResult(
        image_name1=image_name1,
        image_name2=image_name2,
        raw_match_path=raw_match_path,
        feature_path1=feature_path1,
        feature_path2=feature_path2,
        fundamental_matrix=fundamental_matrix,
        homography_matrix=empty_matrix,
        essential_matrix=empty_matrix,
        inlier_matches=empty_matches,
        inlier_mask=empty_mask,
        homography_inlier_matches=empty_matches,
        homography_inlier_mask=empty_mask,
        essential_inlier_matches=empty_matches,
        essential_inlier_mask=empty_mask,
        similarity_inlier_matches=empty_matches,
        similarity_inlier_mask=empty_mask,
        num_raw_matches=num_raw_matches,
        num_inliers=0,
        num_h_inliers=0,
        num_e_inliers=0,
        num_s_inliers=0,
        inlier_ratio=0.0,
        h_inlier_ratio=0.0,
        e_inlier_ratio=0.0,
        s_inlier_ratio=0.0,
        homography_ratio=0.0,
        essential_ratio=0.0,
        similarity_ratio=0.0,
        median_triangulation_angle_deg=0.0,
        model_type=model_type,
        calibration_status="uncertain",
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
        H=result.homography_matrix,
        E=result.essential_matrix,
        inlier_matches=result.inlier_matches,
        inlier_mask=result.inlier_mask,
        h_inlier_matches=result.homography_inlier_matches,
        h_inlier_mask=result.homography_inlier_mask,
        e_inlier_matches=result.essential_inlier_matches,
        e_inlier_mask=result.essential_inlier_mask,
        s_inlier_matches=result.similarity_inlier_matches,
        s_inlier_mask=result.similarity_inlier_mask,
        num_raw_matches=np.int32(result.num_raw_matches),
        num_inliers=np.int32(result.num_inliers),
        num_h_inliers=np.int32(result.num_h_inliers),
        num_e_inliers=np.int32(result.num_e_inliers),
        num_s_inliers=np.int32(result.num_s_inliers),
        inlier_ratio=np.float32(result.inlier_ratio),
        h_inlier_ratio=np.float32(result.h_inlier_ratio),
        e_inlier_ratio=np.float32(result.e_inlier_ratio),
        s_inlier_ratio=np.float32(result.s_inlier_ratio),
        homography_ratio=np.float32(result.homography_ratio),
        essential_ratio=np.float32(result.essential_ratio),
        similarity_ratio=np.float32(result.similarity_ratio),
        median_triangulation_angle_deg=np.float32(result.median_triangulation_angle_deg),
        model_type=result.model_type,
        calibration_status=result.calibration_status,
        status=result.status,
    )


def verified_output_path(verified_dir: Path, match_path: Path) -> Path:
    return verified_dir / match_path.name
