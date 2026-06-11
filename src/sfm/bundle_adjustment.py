from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.sfm.camera import PinholeCamera, project_points
from src.sfm.reconstruction import ReconstructionState


@dataclass(frozen=True)
class BundleAdjustmentProblem:
    image_names: list[str]
    fixed_image_name: str
    optimizable_image_names: list[str]
    point_ids: np.ndarray
    observations: list[dict]
    observation_points2d: np.ndarray
    camera_param_slices: dict[str, slice]
    point_param_slices: dict[int, slice]
    initial_params: np.ndarray


@dataclass(frozen=True)
class BundleAdjustmentResult:
    optimized_state: ReconstructionState
    problem: BundleAdjustmentProblem
    initial_errors: np.ndarray
    final_errors: np.ndarray
    optimizer_cost_initial: float
    optimizer_cost_final: float
    optimizer_robust_cost_final: float
    num_function_evaluations: int
    success: bool
    message: str


def build_bundle_adjustment_problem(
    state: ReconstructionState,
    keypoints_by_name: dict[str, np.ndarray],
    fixed_image_name: str,
    max_points: int,
    max_observations: int,
    min_track_length: int,
) -> BundleAdjustmentProblem:
    point_observations: dict[int, list[dict]] = {}
    for observation in state.observations:
        point_id = int(observation["point3D_id"])
        if point_id < 0 or point_id >= int(state.points3d.shape[0]):
            continue
        if str(observation["image_name"]) not in state.registered_images:
            continue
        point_observations.setdefault(point_id, []).append(observation)

    eligible_point_ids = [
        point_id
        for point_id, observations in point_observations.items()
        if len(observations) >= min_track_length
    ]
    eligible_point_ids.sort(key=lambda point_id: len(point_observations[point_id]), reverse=True)
    if max_points > 0:
        eligible_point_ids = eligible_point_ids[:max_points]
    selected_point_ids = np.asarray(sorted(eligible_point_ids), dtype=np.int32)
    selected_point_id_set = set(int(point_id) for point_id in selected_point_ids)

    observations = [
        observation
        for observation in state.observations
        if int(observation["point3D_id"]) in selected_point_id_set
        and str(observation["image_name"]) in state.registered_images
    ]
    observations.sort(key=lambda item: (int(item["point3D_id"]), str(item["image_name"]), int(item["keypoint_idx"])))
    if max_observations > 0:
        observations = observations[:max_observations]
        selected_point_ids = np.asarray(
            sorted({int(observation["point3D_id"]) for observation in observations}),
            dtype=np.int32,
        )

    if fixed_image_name not in state.registered_images:
        raise ValueError(f"Fixed image is not registered: {fixed_image_name}")

    image_names = list(state.registered_images)
    optimizable_image_names = [image_name for image_name in image_names if image_name != fixed_image_name]
    params: list[float] = []
    camera_param_slices: dict[str, slice] = {}
    for image_name in optimizable_image_names:
        registered = state.registered_images[image_name]
        rvec, _ = cv2.Rodrigues(registered.R.astype(np.float64))
        start = len(params)
        params.extend(rvec.reshape(3).tolist())
        params.extend(registered.t.reshape(3).astype(np.float64).tolist())
        camera_param_slices[image_name] = slice(start, start + 6)

    point_param_slices: dict[int, slice] = {}
    for point_id in selected_point_ids:
        start = len(params)
        params.extend(state.points3d[int(point_id)].astype(np.float64).tolist())
        point_param_slices[int(point_id)] = slice(start, start + 3)

    observation_points2d = np.asarray(
        [
            keypoints_by_name[str(observation["image_name"])][int(observation["keypoint_idx"]), :2]
            for observation in observations
        ],
        dtype=np.float64,
    )

    return BundleAdjustmentProblem(
        image_names=image_names,
        fixed_image_name=fixed_image_name,
        optimizable_image_names=optimizable_image_names,
        point_ids=selected_point_ids,
        observations=observations,
        observation_points2d=observation_points2d,
        camera_param_slices=camera_param_slices,
        point_param_slices=point_param_slices,
        initial_params=np.asarray(params, dtype=np.float64),
    )


def run_bundle_adjustment(
    state: ReconstructionState,
    cameras: dict[str, PinholeCamera],
    keypoints_by_name: dict[str, np.ndarray],
    fixed_image_name: str,
    max_iterations: int,
    loss: str,
    f_scale: float,
    max_points: int,
    max_observations: int,
    min_track_length: int,
) -> BundleAdjustmentResult:
    problem = build_bundle_adjustment_problem(
        state=state,
        keypoints_by_name=keypoints_by_name,
        fixed_image_name=fixed_image_name,
        max_points=max_points,
        max_observations=max_observations,
        min_track_length=min_track_length,
    )
    if problem.initial_params.size == 0 or not problem.observations:
        raise ValueError("Bundle adjustment problem is empty.")

    initial_residuals = ba_residuals(problem.initial_params, state, problem, cameras)
    result = least_squares(
        fun=lambda params: ba_residuals(params, state, problem, cameras),
        x0=problem.initial_params,
        jac_sparsity=bundle_adjustment_sparsity(problem),
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_iterations,
        verbose=0,
    )
    final_residuals = ba_residuals(result.x, state, problem, cameras)
    optimized_state = apply_bundle_adjustment_result(state, problem, result.x)
    return BundleAdjustmentResult(
        optimized_state=optimized_state,
        problem=problem,
        initial_errors=residuals_to_errors(initial_residuals),
        final_errors=residuals_to_errors(final_residuals),
        optimizer_cost_initial=float(0.5 * np.sum(initial_residuals**2)),
        optimizer_cost_final=float(0.5 * np.sum(final_residuals**2)),
        optimizer_robust_cost_final=float(result.cost),
        num_function_evaluations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )


def bundle_adjustment_sparsity(problem: BundleAdjustmentProblem):
    num_residuals = 2 * len(problem.observations)
    num_params = int(problem.initial_params.size)
    sparsity = lil_matrix((num_residuals, num_params), dtype=np.int8)
    for observation_index, observation in enumerate(problem.observations):
        rows = slice(2 * observation_index, 2 * observation_index + 2)
        camera_slice = problem.camera_param_slices.get(str(observation["image_name"]))
        if camera_slice is not None:
            sparsity[rows, camera_slice] = 1
        point_slice = problem.point_param_slices[int(observation["point3D_id"])]
        sparsity[rows, point_slice] = 1
    return sparsity.tocsr()


def ba_residuals(
    params: np.ndarray,
    state: ReconstructionState,
    problem: BundleAdjustmentProblem,
    cameras: dict[str, PinholeCamera],
) -> np.ndarray:
    residuals = np.empty((len(problem.observations), 2), dtype=np.float64)
    for index, observation in enumerate(problem.observations):
        image_name = str(observation["image_name"])
        point_id = int(observation["point3D_id"])
        R, t = pose_from_params(params, state, problem, image_name)
        point3d = point_from_params(params, state, problem, point_id)
        projected = project_points(cameras[image_name].K, R, t, point3d.reshape(1, 3))[0]
        residuals[index] = projected - problem.observation_points2d[index]
    return residuals.reshape(-1)


def pose_from_params(
    params: np.ndarray,
    state: ReconstructionState,
    problem: BundleAdjustmentProblem,
    image_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    param_slice = problem.camera_param_slices.get(image_name)
    if param_slice is None:
        registered = state.registered_images[image_name]
        return registered.R.astype(np.float64), registered.t.reshape(3).astype(np.float64)
    values = params[param_slice]
    R, _ = cv2.Rodrigues(values[:3].reshape(3, 1))
    t = values[3:6].reshape(3)
    return R.astype(np.float64), t.astype(np.float64)


def point_from_params(
    params: np.ndarray,
    state: ReconstructionState,
    problem: BundleAdjustmentProblem,
    point_id: int,
) -> np.ndarray:
    param_slice = problem.point_param_slices.get(point_id)
    if param_slice is None:
        return state.points3d[point_id].astype(np.float64)
    return params[param_slice].astype(np.float64)


def apply_bundle_adjustment_result(
    state: ReconstructionState,
    problem: BundleAdjustmentProblem,
    params: np.ndarray,
) -> ReconstructionState:
    for image_name in problem.optimizable_image_names:
        R, t = pose_from_params(params, state, problem, image_name)
        state.registered_images[image_name].R = R
        state.registered_images[image_name].t = t
    for point_id in problem.point_ids:
        state.points3d[int(point_id)] = point_from_params(params, state, problem, int(point_id))
    return state


def residuals_to_errors(residuals: np.ndarray) -> np.ndarray:
    if residuals.size == 0:
        return np.empty((0,), dtype=np.float64)
    return np.linalg.norm(residuals.reshape(-1, 2), axis=1)


def per_image_error_summary(observations: list[dict], errors: np.ndarray) -> list[dict]:
    grouped: dict[str, list[float]] = {}
    for observation, error in zip(observations, errors):
        grouped.setdefault(str(observation["image_name"]), []).append(float(error))
    return [
        {
            "image_name": image_name,
            "num_observations": len(values),
            "mean_error": float(np.mean(values)),
            "median_error": float(np.median(values)),
            "max_error": float(np.max(values)),
        }
        for image_name, values in sorted(grouped.items())
    ]
