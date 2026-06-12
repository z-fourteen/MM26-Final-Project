from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.sfm.camera import PinholeCamera, project_points


@dataclass
class RegisteredImage:
    image_name: str
    R: np.ndarray
    t: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t.reshape(3)


@dataclass
class ReconstructionState:
    registered_images: dict[str, RegisteredImage]
    points3d: np.ndarray
    colors: np.ndarray
    observations: list[dict]


@dataclass(frozen=True)
class Candidate2D3D:
    image_name: str
    points2d: np.ndarray
    points3d: np.ndarray
    keypoint_indices: np.ndarray
    point3d_ids: np.ndarray
    match_sources: np.ndarray

    @property
    def num_correspondences(self) -> int:
        return int(self.points2d.shape[0])


@dataclass(frozen=True)
class NextBestViewScore:
    image_name: str
    pyramid_score: float
    num_2d3d: int
    registered_neighbor_count: int
    num_general_edges: int
    num_planar_edges: int
    mean_homography_ratio: float


@dataclass(frozen=True)
class RegistrationResult:
    image_name: str
    success: bool
    R: np.ndarray
    t: np.ndarray
    pnp_inliers: np.ndarray
    mean_reprojection_error: float
    num_2d3d: int
    status: str
    keypoint_indices: np.ndarray
    point3d_ids: np.ndarray


def load_initial_state(sparse_dir: Path) -> ReconstructionState:
    pair = np.load(sparse_dir / "initial_pair.npz")
    points = np.load(sparse_dir / "initial_points.npz")
    image_name1 = str(pair["image_name1"])
    image_name2 = str(pair["image_name2"])
    track_keypoint_indices = points["track_keypoint_indices"].astype(np.int32)
    points3d = points["points3D"].astype(np.float64)
    colors = points["colors"].astype(np.uint8)

    registered_images = {
        image_name1: RegisteredImage(image_name=image_name1, R=pair["R1"].astype(np.float64), t=pair["t1"].astype(np.float64)),
        image_name2: RegisteredImage(image_name=image_name2, R=pair["R2"].astype(np.float64), t=pair["t2"].astype(np.float64)),
    }
    observations = []
    for point3d_id, (keypoint_idx1, keypoint_idx2) in enumerate(track_keypoint_indices):
        observations.append(
            {
                "point3D_id": int(point3d_id),
                "image_name": image_name1,
                "keypoint_idx": int(keypoint_idx1),
            }
        )
        observations.append(
            {
                "point3D_id": int(point3d_id),
                "image_name": image_name2,
                "keypoint_idx": int(keypoint_idx2),
            }
        )

    return ReconstructionState(
        registered_images=registered_images,
        points3d=points3d,
        colors=colors,
        observations=observations,
    )


def load_reconstruction_state(sparse_dir: Path) -> ReconstructionState:
    state_path = sparse_dir / "reconstruction_state.json"
    points_path = sparse_dir / "reconstruction_points.npz"
    if not state_path.exists():
        return load_initial_state(sparse_dir)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    registered_images = {
        image_name: RegisteredImage(
            image_name=image_name,
            R=np.asarray(record["R"], dtype=np.float64),
            t=np.asarray(record["t"], dtype=np.float64).reshape(3),
        )
        for image_name, record in payload["registered_images"].items()
    }
    if points_path.exists():
        point_data = np.load(points_path)
        points3d = point_data["points3D"].astype(np.float64)
        colors = point_data["colors"].astype(np.uint8)
    else:
        initial_points = np.load(sparse_dir / "initial_points.npz")
        points3d = initial_points["points3D"].astype(np.float64)
        colors = initial_points["colors"].astype(np.uint8)

    return ReconstructionState(
        registered_images=registered_images,
        points3d=points3d,
        colors=colors,
        observations=list(payload.get("observations", [])),
    )


def observation_lookup(state: ReconstructionState) -> dict[tuple[str, int], int]:
    return {
        (str(observation["image_name"]), int(observation["keypoint_idx"])): int(observation["point3D_id"])
        for observation in state.observations
    }


def collect_candidate_correspondences(
    image_name: str,
    state: ReconstructionState,
    verified_dir: Path,
    keypoints_by_name: dict[str, np.ndarray],
    allow_planar: bool = False,
    min_general_correspondences: int = 30,
    max_planar_fraction: float = 0.5,
) -> Candidate2D3D:
    lookup = observation_lookup(state)
    general_correspondences: dict[int, tuple[int, int, str]] = {}
    planar_correspondences: dict[int, tuple[int, int, str]] = {}

    for registered_name in state.registered_images:
        verified_path = _find_verified_path(verified_dir, image_name, registered_name)
        if verified_path is None:
            continue
        data = np.load(verified_path)
        status = str(data["status"])
        if status == "verified":
            target = general_correspondences
            source = "general"
        elif allow_planar and status == "verified_planar":
            target = planar_correspondences
            source = "planar"
        else:
            continue
        name1 = str(data["image_name1"])
        name2 = str(data["image_name2"])
        matches = data["inlier_matches"].astype(np.int32)
        for idx1, idx2 in matches:
            if name1 == image_name and name2 == registered_name:
                candidate_kp = int(idx1)
                registered_kp = int(idx2)
            elif name2 == image_name and name1 == registered_name:
                candidate_kp = int(idx2)
                registered_kp = int(idx1)
            else:
                continue
            point3d_id = lookup.get((registered_name, registered_kp))
            if point3d_id is None:
                continue
            target.setdefault(candidate_kp, (point3d_id, registered_kp, source))

    correspondences = dict(general_correspondences)
    if allow_planar:
        num_general = len(general_correspondences)
        planar_budget = max(0, int(max(num_general, min_general_correspondences) * max_planar_fraction))
        for candidate_kp in sorted(planar_correspondences):
            if candidate_kp in correspondences:
                continue
            if planar_budget <= 0:
                break
            correspondences[candidate_kp] = planar_correspondences[candidate_kp]
            planar_budget -= 1

    if not correspondences:
        return Candidate2D3D(
            image_name=image_name,
            points2d=np.empty((0, 2), dtype=np.float64),
            points3d=np.empty((0, 3), dtype=np.float64),
            keypoint_indices=np.empty((0,), dtype=np.int32),
            point3d_ids=np.empty((0,), dtype=np.int32),
            match_sources=np.empty((0,), dtype="<U8"),
        )

    keypoint_indices = np.array(sorted(correspondences), dtype=np.int32)
    point3d_ids = np.array([correspondences[int(kp)][0] for kp in keypoint_indices], dtype=np.int32)
    match_sources = np.array([correspondences[int(kp)][2] for kp in keypoint_indices])
    points2d = keypoints_by_name[image_name][keypoint_indices, :2].astype(np.float64)
    points3d = state.points3d[point3d_ids].astype(np.float64)
    return Candidate2D3D(
        image_name=image_name,
        points2d=points2d,
        points3d=points3d,
        keypoint_indices=keypoint_indices,
        point3d_ids=point3d_ids,
        match_sources=match_sources,
    )


def _find_verified_path(verified_dir: Path, image_name_a: str, image_name_b: str) -> Path | None:
    stem_a = Path(image_name_a).stem
    stem_b = Path(image_name_b).stem
    candidates = [
        verified_dir / f"{stem_a}__{stem_b}.npz",
        verified_dir / f"{stem_b}__{stem_a}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def image_pyramid_visibility_score(
    points2d: np.ndarray,
    image_width: int,
    image_height: int,
    levels: int = 3,
) -> float:
    if points2d.size == 0:
        return 0.0
    width = max(int(image_width), 1)
    height = max(int(image_height), 1)
    clamped = points2d.astype(np.float64, copy=False)
    xs = np.clip(clamped[:, 0], 0.0, np.nextafter(float(width), 0.0))
    ys = np.clip(clamped[:, 1], 0.0, np.nextafter(float(height), 0.0))
    score = 0.0
    for level in range(levels + 1):
        cells_per_axis = 2**level
        x_cells = np.floor(xs / float(width) * cells_per_axis).astype(np.int32)
        y_cells = np.floor(ys / float(height) * cells_per_axis).astype(np.int32)
        x_cells = np.clip(x_cells, 0, cells_per_axis - 1)
        y_cells = np.clip(y_cells, 0, cells_per_axis - 1)
        occupied = set(zip(x_cells.tolist(), y_cells.tolist()))
        weight = float(cells_per_axis**2)
        score += weight * float(len(occupied))
    return score


def score_next_best_view(
    candidate: Candidate2D3D,
    camera: PinholeCamera,
    verified_dir: Path,
    registered_image_names: set[str],
    levels: int = 3,
) -> NextBestViewScore:
    diagnostics = scene_graph_registration_diagnostics(
        image_name=candidate.image_name,
        verified_dir=verified_dir,
        registered_image_names=registered_image_names,
    )
    return NextBestViewScore(
        image_name=candidate.image_name,
        pyramid_score=image_pyramid_visibility_score(
            candidate.points2d,
            image_width=camera.width,
            image_height=camera.height,
            levels=levels,
        ),
        num_2d3d=candidate.num_correspondences,
        registered_neighbor_count=diagnostics["registered_neighbor_count"],
        num_general_edges=diagnostics["num_general_edges"],
        num_planar_edges=diagnostics["num_planar_edges"],
        mean_homography_ratio=diagnostics["mean_homography_ratio"],
    )


def scene_graph_registration_diagnostics(
    image_name: str,
    verified_dir: Path,
    registered_image_names: set[str],
) -> dict[str, int | float]:
    homography_ratios = []
    registered_neighbor_count = 0
    num_general_edges = 0
    num_planar_edges = 0
    for registered_name in registered_image_names:
        verified_path = _find_verified_path(verified_dir, image_name, registered_name)
        if verified_path is None:
            continue
        data = np.load(verified_path)
        status = str(data["status"])
        if status not in {"verified", "verified_planar"}:
            continue
        registered_neighbor_count += 1
        model_type = str(data["model_type"]) if "model_type" in data else "unknown"
        if model_type == "general":
            num_general_edges += 1
        elif model_type == "planar":
            num_planar_edges += 1
        if "homography_ratio" in data:
            homography_ratios.append(float(data["homography_ratio"]))
    return {
        "registered_neighbor_count": registered_neighbor_count,
        "num_general_edges": num_general_edges,
        "num_planar_edges": num_planar_edges,
        "mean_homography_ratio": float(np.mean(homography_ratios)) if homography_ratios else 0.0,
    }


def register_image_pnp(
    candidate: Candidate2D3D,
    camera: PinholeCamera,
    min_2d3d: int,
    min_pnp_inliers: int,
    reproj_error_px: float,
    confidence: float = 0.999,
    min_inlier_ratio: float = 0.0,
    max_mean_error_px: float = 0.0,
    max_median_error_px: float = 0.0,
    min_cheirality_ratio: float = 0.0,
    min_depth_iqr: float = 0.0,
    min_grid_coverage: int = 0,
    grid_size: int = 4,
) -> RegistrationResult:
    if candidate.num_correspondences < min_2d3d:
        return _failed_registration(candidate, "not_enough_2d3d")
    if min_grid_coverage > 0:
        coverage = grid_coverage(candidate.points2d, camera.width, camera.height, grid_size=grid_size)
        if coverage < min_grid_coverage:
            return _failed_registration(candidate, "poor_spatial_coverage")

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=candidate.points3d.astype(np.float64),
        imagePoints=candidate.points2d.astype(np.float64),
        cameraMatrix=camera.K,
        distCoeffs=None,
        iterationsCount=1000,
        reprojectionError=reproj_error_px,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < min_pnp_inliers:
        return _failed_registration(candidate, "pnp_failed")
    if min_inlier_ratio > 0 and float(len(inliers)) / float(max(candidate.num_correspondences, 1)) < min_inlier_ratio:
        return _failed_registration(candidate, "low_pnp_inlier_ratio")

    inlier_indices = inliers.ravel().astype(np.int32)
    inlier_points3d = candidate.points3d[inlier_indices]
    inlier_points2d = candidate.points2d[inlier_indices]
    cv2.solvePnP(
        objectPoints=inlier_points3d.astype(np.float64),
        imagePoints=inlier_points2d.astype(np.float64),
        cameraMatrix=camera.K,
        distCoeffs=None,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3).astype(np.float64)
    projected = project_points(camera.K, R, t, inlier_points3d)
    errors = np.linalg.norm(projected - inlier_points2d, axis=1)
    mean_error = float(np.mean(errors))
    median_error = float(np.median(errors))
    depths = camera_depths_for_pose(R, t, inlier_points3d)
    cheirality_ratio = float(np.mean(depths > 1e-8)) if depths.size else 0.0
    depth_iqr = float(np.percentile(depths, 75) - np.percentile(depths, 25)) if depths.size else 0.0
    if max_mean_error_px > 0 and mean_error > max_mean_error_px:
        return _failed_registration(candidate, "high_pnp_mean_error")
    if max_median_error_px > 0 and median_error > max_median_error_px:
        return _failed_registration(candidate, "high_pnp_median_error")
    if min_cheirality_ratio > 0 and cheirality_ratio < min_cheirality_ratio:
        return _failed_registration(candidate, "low_cheirality")
    if min_depth_iqr > 0 and depth_iqr < min_depth_iqr:
        return _failed_registration(candidate, "low_depth_dispersion")
    return RegistrationResult(
        image_name=candidate.image_name,
        success=True,
        R=R.astype(np.float64),
        t=t,
        pnp_inliers=inlier_indices,
        mean_reprojection_error=mean_error,
        num_2d3d=candidate.num_correspondences,
        status="registered",
        keypoint_indices=candidate.keypoint_indices[inlier_indices],
        point3d_ids=candidate.point3d_ids[inlier_indices],
    )


def _failed_registration(candidate: Candidate2D3D, status: str) -> RegistrationResult:
    return RegistrationResult(
        image_name=candidate.image_name,
        success=False,
        R=np.eye(3, dtype=np.float64),
        t=np.zeros(3, dtype=np.float64),
        pnp_inliers=np.empty((0,), dtype=np.int32),
        mean_reprojection_error=0.0,
        num_2d3d=candidate.num_correspondences,
        status=status,
        keypoint_indices=np.empty((0,), dtype=np.int32),
        point3d_ids=np.empty((0,), dtype=np.int32),
    )


def grid_coverage(points2d: np.ndarray, width: int, height: int, grid_size: int = 4) -> int:
    if points2d.size == 0:
        return 0
    cells = max(int(grid_size), 1)
    xs = np.clip(points2d[:, 0], 0.0, np.nextafter(float(max(width, 1)), 0.0))
    ys = np.clip(points2d[:, 1], 0.0, np.nextafter(float(max(height, 1)), 0.0))
    x_cells = np.floor(xs / float(max(width, 1)) * cells).astype(np.int32)
    y_cells = np.floor(ys / float(max(height, 1)) * cells).astype(np.int32)
    return len(set(zip(x_cells.tolist(), y_cells.tolist())))


def camera_depths_for_pose(R: np.ndarray, t: np.ndarray, points3d: np.ndarray) -> np.ndarray:
    camera_points = (R @ points3d.T + t.reshape(3, 1)).T
    return camera_points[:, 2]


def add_registered_image(state: ReconstructionState, result: RegistrationResult) -> None:
    state.registered_images[result.image_name] = RegisteredImage(
        image_name=result.image_name,
        R=result.R,
        t=result.t,
    )
    existing = set(observation_lookup(state))
    for keypoint_idx, point3d_id in zip(result.keypoint_indices, result.point3d_ids):
        key = (result.image_name, int(keypoint_idx))
        if key in existing:
            continue
        state.observations.append(
            {
                "point3D_id": int(point3d_id),
                "image_name": result.image_name,
                "keypoint_idx": int(keypoint_idx),
            }
        )
        existing.add(key)


def save_reconstruction_state(state: ReconstructionState, sparse_dir: Path) -> tuple[Path, Path]:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    state_path = sparse_dir / "reconstruction_state.json"
    registered_npz_path = sparse_dir / "registered_images.npz"
    points_npz_path = sparse_dir / "reconstruction_points.npz"
    payload = {
        "registered_images": {
            image_name: {
                "R": registered.R.tolist(),
                "t": registered.t.reshape(3).tolist(),
                "center": registered.center.tolist(),
            }
            for image_name, registered in state.registered_images.items()
        },
        "num_points3D": int(state.points3d.shape[0]),
        "num_observations": len(state.observations),
        "observations": state.observations,
    }
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    image_names = np.array(list(state.registered_images.keys()))
    rotations = np.stack([registered.R for registered in state.registered_images.values()])
    translations = np.stack([registered.t.reshape(3) for registered in state.registered_images.values()])
    centers = np.stack([registered.center for registered in state.registered_images.values()])
    np.savez_compressed(
        registered_npz_path,
        image_names=image_names,
        rotations=rotations,
        translations=translations,
        centers=centers,
    )
    np.savez_compressed(
        points_npz_path,
        points3D=state.points3d.astype(np.float64),
        colors=state.colors.astype(np.uint8),
    )
    return state_path, registered_npz_path
