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
    scope: str
    image_names: list[str]
    fixed_image_name: str
    local_image_names: list[str]
    optimizable_image_names: list[str]
    point_ids: np.ndarray
    observations: list[dict]
    observation_points2d: np.ndarray
    camera_param_slices: dict[str, slice]
    point_param_slices: dict[int, slice]
    shared_focal_slice: slice | None
    initial_params: np.ndarray
    observation_groups_by_image: dict[str, np.ndarray]
    initial_shared_focal: float
    optimize_shared_focal: bool


@dataclass(frozen=True)
class BundleAdjustmentResult:
    optimized_state: ReconstructionState
    optimized_cameras: dict[str, PinholeCamera]
    problem: BundleAdjustmentProblem
    initial_errors: np.ndarray
    final_errors: np.ndarray
    initial_squared_residual_cost: float
    final_squared_residual_cost: float
    initial_robust_cost: float
    final_robust_cost: float
    optimizer_robust_cost_final: float
    num_function_evaluations: int
    success: bool
    message: str
    initial_shared_focal: float
    final_shared_focal: float


def build_bundle_adjustment_problem(
    state: ReconstructionState,
    keypoints_by_name: dict[str, np.ndarray],
    fixed_image_name: str,
    max_points: int,
    max_observations: int,
    min_track_length: int,
    scope: str = "global",
    local_image_names: list[str] | None = None,
    point_priority_errors: dict[int, float] | None = None,
    cameras: dict[str, PinholeCamera] | None = None,
    optimize_shared_focal: bool = False,
) -> BundleAdjustmentProblem:
    if scope not in {"global", "local"}:
        raise ValueError(f"Unsupported BA scope: {scope}")
    if fixed_image_name not in state.registered_images:
        raise ValueError(f"Fixed image is not registered: {fixed_image_name}")

    local_image_set = set(local_image_names or [])
    if scope == "local":
        local_image_set = {image_name for image_name in local_image_set if image_name in state.registered_images}
        if not local_image_set:
            raise ValueError("Local BA requires at least one registered local image.")

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
        and (scope == "global" or any(str(observation["image_name"]) in local_image_set for observation in observations))
    ]
    point_priority_errors = point_priority_errors or {}
    eligible_point_ids.sort(
        key=lambda point_id: (
            float(point_priority_errors.get(point_id, 0.0)),
            len(point_observations[point_id]),
        ),
        reverse=True,
    )
    if max_points > 0:
        eligible_point_ids = eligible_point_ids[:max_points]
    selected_point_ids = np.asarray(sorted(eligible_point_ids), dtype=np.int32)
    selected_point_id_set = set(int(point_id) for point_id in selected_point_ids)

    observations = [
        observation
        for observation in state.observations
        if int(observation["point3D_id"]) in selected_point_id_set
        and str(observation["image_name"]) in state.registered_images
        and (
            scope == "global"
            or str(observation["image_name"]) in local_image_set
            or int(observation["point3D_id"]) in selected_point_id_set
        )
    ]
    observations.sort(key=lambda item: (int(item["point3D_id"]), str(item["image_name"]), int(item["keypoint_idx"])))
    if max_observations > 0:
        observations = select_observations_with_balanced_coverage(observations, max_observations)
        selected_point_ids = np.asarray(
            sorted({int(observation["point3D_id"]) for observation in observations}),
            dtype=np.int32,
        )

    image_names = list(state.registered_images)
    if scope == "global":
        optimizable_image_names = [image_name for image_name in image_names if image_name != fixed_image_name]
    else:
        optimizable_image_names = [
            image_name
            for image_name in image_names
            if image_name in local_image_set and image_name != fixed_image_name
        ]
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

    initial_shared_focal = 0.0
    shared_focal_slice = None
    if optimize_shared_focal:
        if not cameras:
            raise ValueError("Shared focal optimization requires cameras.")
        registered_focals = [
            float(cameras[image_name].fx)
            for image_name in image_names
            if image_name in cameras
        ]
        if not registered_focals:
            raise ValueError("No registered camera intrinsics available for shared focal optimization.")
        initial_shared_focal = float(np.median(registered_focals))
        start = len(params)
        params.append(initial_shared_focal)
        shared_focal_slice = slice(start, start + 1)

    observation_points2d = np.asarray(
        [
            keypoints_by_name[str(observation["image_name"])][int(observation["keypoint_idx"]), :2]
            for observation in observations
        ],
        dtype=np.float64,
    )
    observation_groups_by_image: dict[str, list[int]] = {}
    for observation_index, observation in enumerate(observations):
        observation_groups_by_image.setdefault(str(observation["image_name"]), []).append(observation_index)

    return BundleAdjustmentProblem(
        scope=scope,
        image_names=image_names,
        fixed_image_name=fixed_image_name,
        local_image_names=sorted(local_image_set),
        optimizable_image_names=optimizable_image_names,
        point_ids=selected_point_ids,
        observations=observations,
        observation_points2d=observation_points2d,
        camera_param_slices=camera_param_slices,
        point_param_slices=point_param_slices,
        shared_focal_slice=shared_focal_slice,
        initial_params=np.asarray(params, dtype=np.float64),
        observation_groups_by_image={
            image_name: np.asarray(indices, dtype=np.int32)
            for image_name, indices in observation_groups_by_image.items()
        },
        initial_shared_focal=initial_shared_focal,
        optimize_shared_focal=optimize_shared_focal,
    )


def select_observations_with_balanced_coverage(observations: list[dict], max_observations: int) -> list[dict]:
    if max_observations <= 0 or len(observations) <= max_observations:
        return observations

    grouped_by_image: dict[str, list[dict]] = {}
    for observation in observations:
        grouped_by_image.setdefault(str(observation["image_name"]), []).append(observation)

    image_names = sorted(grouped_by_image)
    selected = []
    selected_keys: set[tuple[int, str, int]] = set()
    cursor = 0
    while len(selected) < max_observations and image_names:
        image_name = image_names[cursor % len(image_names)]
        bucket = grouped_by_image[image_name]
        if bucket:
            observation = bucket.pop(0)
            key = (
                int(observation["point3D_id"]),
                str(observation["image_name"]),
                int(observation["keypoint_idx"]),
            )
            if key not in selected_keys:
                selected.append(observation)
                selected_keys.add(key)
        if not bucket:
            image_names.remove(image_name)
            if not image_names:
                break
            cursor %= len(image_names)
        else:
            cursor += 1

    selected.sort(key=lambda item: (int(item["point3D_id"]), str(item["image_name"]), int(item["keypoint_idx"])))
    return selected


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
    scope: str = "global",
    local_image_names: list[str] | None = None,
    point_priority_errors: dict[int, float] | None = None,
    optimize_shared_focal: bool = False,
    focal_bound_scale: float = 2.0,
) -> BundleAdjustmentResult:
    problem = build_bundle_adjustment_problem(
        state=state,
        keypoints_by_name=keypoints_by_name,
        fixed_image_name=fixed_image_name,
        max_points=max_points,
        max_observations=max_observations,
        min_track_length=min_track_length,
        scope=scope,
        local_image_names=local_image_names,
        point_priority_errors=point_priority_errors,
        cameras=cameras,
        optimize_shared_focal=optimize_shared_focal,
    )
    if problem.initial_params.size == 0 or not problem.observations:
        raise ValueError("Bundle adjustment problem is empty.")

    initial_residuals = ba_residuals(problem.initial_params, state, problem, cameras)
    lower_bounds, upper_bounds = bundle_adjustment_bounds(problem, focal_bound_scale=focal_bound_scale)
    result = least_squares(
        fun=lambda params: ba_residuals(params, state, problem, cameras),
        x0=problem.initial_params,
        jac_sparsity=bundle_adjustment_sparsity(problem),
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_iterations,
        bounds=(lower_bounds, upper_bounds),
        verbose=0,
    )
    final_residuals = ba_residuals(result.x, state, problem, cameras)
    optimized_state = apply_bundle_adjustment_result(state, problem, result.x)
    optimized_cameras = apply_camera_result(cameras, problem, result.x)
    return BundleAdjustmentResult(
        optimized_state=optimized_state,
        optimized_cameras=optimized_cameras,
        problem=problem,
        initial_errors=residuals_to_errors(initial_residuals),
        final_errors=residuals_to_errors(final_residuals),
        initial_squared_residual_cost=squared_residual_cost(initial_residuals),
        final_squared_residual_cost=squared_residual_cost(final_residuals),
        initial_robust_cost=robust_residual_cost(initial_residuals, loss=loss, f_scale=f_scale),
        final_robust_cost=robust_residual_cost(final_residuals, loss=loss, f_scale=f_scale),
        optimizer_robust_cost_final=float(result.cost),
        num_function_evaluations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        initial_shared_focal=problem.initial_shared_focal,
        final_shared_focal=shared_focal_from_params(result.x, problem),
    )


def build_covisibility_graph(state: ReconstructionState) -> dict[tuple[str, str], int]:
    point_images: dict[int, set[str]] = {}
    for observation in state.observations:
        point_id = int(observation["point3D_id"])
        image_name = str(observation["image_name"])
        if point_id < 0 or point_id >= int(state.points3d.shape[0]):
            continue
        if image_name not in state.registered_images:
            continue
        point_images.setdefault(point_id, set()).add(image_name)

    edges: dict[tuple[str, str], int] = {}
    for image_names in point_images.values():
        sorted_names = sorted(image_names)
        for i, image_name1 in enumerate(sorted_names):
            for image_name2 in sorted_names[i + 1 :]:
                key = (image_name1, image_name2)
                edges[key] = edges.get(key, 0) + 1
    return edges


def select_local_ba_images(
    state: ReconstructionState,
    target_image_name: str,
    num_neighbors: int,
) -> list[str]:
    if target_image_name not in state.registered_images:
        raise ValueError(f"Target image is not registered: {target_image_name}")
    covisibility = build_covisibility_graph(state)
    neighbor_weights: dict[str, int] = {}
    for (image_name1, image_name2), weight in covisibility.items():
        if image_name1 == target_image_name:
            neighbor_weights[image_name2] = neighbor_weights.get(image_name2, 0) + weight
        elif image_name2 == target_image_name:
            neighbor_weights[image_name1] = neighbor_weights.get(image_name1, 0) + weight
    neighbors = sorted(neighbor_weights, key=lambda image_name: (-neighbor_weights[image_name], image_name))
    if num_neighbors > 0:
        neighbors = neighbors[:num_neighbors]
    return [target_image_name, *neighbors]


def squared_residual_cost(residuals: np.ndarray) -> float:
    return float(0.5 * np.sum(residuals.astype(np.float64) ** 2))


def robust_residual_cost(residuals: np.ndarray, loss: str, f_scale: float) -> float:
    if residuals.size == 0:
        return 0.0
    scale = float(f_scale)
    if scale <= 0:
        raise ValueError(f"f_scale must be positive, got {f_scale}")
    z = (residuals.astype(np.float64) / scale) ** 2
    if loss == "linear":
        rho = z
    elif loss == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif loss == "huber":
        rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    elif loss == "cauchy":
        rho = np.log1p(z)
    elif loss == "arctan":
        rho = np.arctan(z)
    else:
        raise ValueError(f"Unsupported robust loss: {loss}")
    return float(0.5 * scale**2 * np.sum(rho))


def error_distribution(errors: np.ndarray, thresholds: tuple[float, ...] = (4.0, 8.0, 16.0)) -> dict:
    if errors.size == 0:
        summary = {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    else:
        summary = {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "p90": float(np.percentile(errors, 90)),
            "p95": float(np.percentile(errors, 95)),
            "max": float(np.max(errors)),
        }
    for threshold in thresholds:
        summary[f"observations_above_{int(threshold)}px"] = int(np.sum(errors > threshold))
    return summary


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
        if problem.shared_focal_slice is not None:
            sparsity[rows, problem.shared_focal_slice] = 1
    return sparsity.tocsr()


def bundle_adjustment_bounds(problem: BundleAdjustmentProblem, focal_bound_scale: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(problem.initial_params.shape, -np.inf, dtype=np.float64)
    upper = np.full(problem.initial_params.shape, np.inf, dtype=np.float64)
    if problem.shared_focal_slice is not None:
        scale = max(float(focal_bound_scale), 1.01)
        focal = max(float(problem.initial_shared_focal), 1e-6)
        lower[problem.shared_focal_slice] = focal / scale
        upper[problem.shared_focal_slice] = focal * scale
    return lower, upper


def ba_residuals(
    params: np.ndarray,
    state: ReconstructionState,
    problem: BundleAdjustmentProblem,
    cameras: dict[str, PinholeCamera],
) -> np.ndarray:
    residuals = np.empty((len(problem.observations), 2), dtype=np.float64)
    for image_name, observation_indices in problem.observation_groups_by_image.items():
        R, t = pose_from_params(params, state, problem, image_name)
        points3d = np.stack(
            [
                point_from_params(params, state, problem, int(problem.observations[int(index)]["point3D_id"]))
                for index in observation_indices
            ]
        )
        camera = camera_from_params(params, problem, cameras[image_name])
        projected = project_points(camera.K, R, t, points3d)
        residuals[observation_indices] = projected - problem.observation_points2d[observation_indices]
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


def shared_focal_from_params(params: np.ndarray, problem: BundleAdjustmentProblem) -> float:
    if problem.shared_focal_slice is None:
        return problem.initial_shared_focal
    return float(params[problem.shared_focal_slice][0])


def camera_from_params(params: np.ndarray, problem: BundleAdjustmentProblem, camera: PinholeCamera) -> PinholeCamera:
    if problem.shared_focal_slice is None:
        return camera
    focal = shared_focal_from_params(params, problem)
    return PinholeCamera(
        width=camera.width,
        height=camera.height,
        fx=focal,
        fy=focal,
        cx=camera.cx,
        cy=camera.cy,
        source="bundle_adjustment",
    )


def apply_camera_result(
    cameras: dict[str, PinholeCamera],
    problem: BundleAdjustmentProblem,
    params: np.ndarray,
) -> dict[str, PinholeCamera]:
    if problem.shared_focal_slice is None:
        return cameras
    return {
        image_name: camera_from_params(params, problem, camera)
        for image_name, camera in cameras.items()
    }


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
