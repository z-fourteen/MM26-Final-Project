from __future__ import annotations

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

    @property
    def num_correspondences(self) -> int:
        return int(self.points2d.shape[0])


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
) -> Candidate2D3D:
    lookup = observation_lookup(state)
    correspondences: dict[int, tuple[int, int]] = {}

    for registered_name in state.registered_images:
        verified_path = _find_verified_path(verified_dir, image_name, registered_name)
        if verified_path is None:
            continue
        data = np.load(verified_path)
        if str(data["status"]) != "verified":
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
            correspondences.setdefault(candidate_kp, (point3d_id, registered_kp))

    if not correspondences:
        return Candidate2D3D(
            image_name=image_name,
            points2d=np.empty((0, 2), dtype=np.float64),
            points3d=np.empty((0, 3), dtype=np.float64),
            keypoint_indices=np.empty((0,), dtype=np.int32),
            point3d_ids=np.empty((0,), dtype=np.int32),
        )

    keypoint_indices = np.array(sorted(correspondences), dtype=np.int32)
    point3d_ids = np.array([correspondences[int(kp)][0] for kp in keypoint_indices], dtype=np.int32)
    points2d = keypoints_by_name[image_name][keypoint_indices, :2].astype(np.float64)
    points3d = state.points3d[point3d_ids].astype(np.float64)
    return Candidate2D3D(
        image_name=image_name,
        points2d=points2d,
        points3d=points3d,
        keypoint_indices=keypoint_indices,
        point3d_ids=point3d_ids,
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


def register_image_pnp(
    candidate: Candidate2D3D,
    camera: PinholeCamera,
    min_2d3d: int,
    min_pnp_inliers: int,
    reproj_error_px: float,
    confidence: float = 0.999,
) -> RegistrationResult:
    if candidate.num_correspondences < min_2d3d:
        return _failed_registration(candidate, "not_enough_2d3d")

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
    mean_error = float(np.mean(np.linalg.norm(projected - inlier_points2d, axis=1)))
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
    import json

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
    return state_path, registered_npz_path
