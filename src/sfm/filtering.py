from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sfm.reconstruction import ReconstructionState


@dataclass(frozen=True)
class FilteringResult:
    filtered_state: ReconstructionState
    report: dict


def filter_reconstruction_by_ba_errors(
    state: ReconstructionState,
    ba_observations: list[dict],
    ba_errors: np.ndarray,
    max_observation_error: float,
    max_point_median_error: float,
    max_point_max_error: float,
    min_track_length: int,
) -> FilteringResult:
    if len(ba_observations) != int(ba_errors.shape[0]):
        raise ValueError(
            "BA observations and errors have different lengths: "
            f"{len(ba_observations)} vs {ba_errors.shape[0]}"
        )

    observation_errors = _observation_error_lookup(ba_observations, ba_errors)
    per_point_ba_errors = _point_error_groups(ba_observations, ba_errors)

    kept_observations = []
    removed_observation_keys: set[tuple[int, str, int]] = set()
    for observation in state.observations:
        key = observation_key(observation)
        error = observation_errors.get(key)
        if error is not None and error > max_observation_error:
            removed_observation_keys.add(key)
            continue
        kept_observations.append(dict(observation))

    track_lengths_after_observation_filter = _track_lengths(kept_observations)
    removed_points_by_track_length = {
        point_id
        for point_id in range(int(state.points3d.shape[0]))
        if track_lengths_after_observation_filter.get(point_id, 0) < min_track_length
    }
    removed_points_by_error = {
        point_id
        for point_id, errors in per_point_ba_errors.items()
        if float(np.median(errors)) > max_point_median_error or float(np.max(errors)) > max_point_max_error
    }
    removed_point_ids = removed_points_by_track_length | removed_points_by_error

    final_observations = [
        observation
        for observation in kept_observations
        if int(observation["point3D_id"]) not in removed_point_ids
    ]

    filtered_state, point_id_map = compact_reconstruction_points(
        state=state,
        observations=final_observations,
    )

    report = {
        "points_before": int(state.points3d.shape[0]),
        "points_after": int(filtered_state.points3d.shape[0]),
        "observations_before": len(state.observations),
        "observations_after": len(filtered_state.observations),
        "registered_images": len(state.registered_images),
        "max_observation_error": float(max_observation_error),
        "max_point_median_error": float(max_point_median_error),
        "max_point_max_error": float(max_point_max_error),
        "min_track_length": int(min_track_length),
        "ba_observations_scored": len(ba_observations),
        "removed_observations_by_reprojection": len(removed_observation_keys),
        "removed_points_by_track_length": len(removed_points_by_track_length),
        "removed_points_by_error": len(removed_points_by_error),
        "removed_points_total": len(removed_point_ids),
        "per_image_observations_before": per_image_observation_counts(state.observations),
        "per_image_observations_after": per_image_observation_counts(filtered_state.observations),
        "point_id_map_size": len(point_id_map),
    }
    return FilteringResult(filtered_state=filtered_state, report=report)


def compact_reconstruction_points(
    state: ReconstructionState,
    observations: list[dict],
) -> tuple[ReconstructionState, dict[int, int]]:
    used_point_ids = sorted({int(observation["point3D_id"]) for observation in observations})
    point_id_map = {old_id: new_id for new_id, old_id in enumerate(used_point_ids)}

    remapped_observations = []
    for observation in observations:
        old_id = int(observation["point3D_id"])
        new_observation = dict(observation)
        new_observation["point3D_id"] = int(point_id_map[old_id])
        remapped_observations.append(new_observation)

    if used_point_ids:
        point_indices = np.asarray(used_point_ids, dtype=np.int32)
        points3d = state.points3d[point_indices].astype(np.float64)
        colors = state.colors[point_indices].astype(np.uint8)
    else:
        points3d = np.empty((0, 3), dtype=np.float64)
        colors = np.empty((0, 3), dtype=np.uint8)

    return (
        ReconstructionState(
            registered_images=state.registered_images,
            points3d=points3d,
            colors=colors,
            observations=remapped_observations,
        ),
        point_id_map,
    )


def observation_key(observation: dict) -> tuple[int, str, int]:
    return (
        int(observation["point3D_id"]),
        str(observation["image_name"]),
        int(observation["keypoint_idx"]),
    )


def per_image_observation_counts(observations: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for observation in observations:
        image_name = str(observation["image_name"])
        counts[image_name] = counts.get(image_name, 0) + 1
    return dict(sorted(counts.items()))


def _observation_error_lookup(
    observations: list[dict],
    errors: np.ndarray,
) -> dict[tuple[int, str, int], float]:
    return {
        observation_key(observation): float(error)
        for observation, error in zip(observations, errors)
    }


def _point_error_groups(
    observations: list[dict],
    errors: np.ndarray,
) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    for observation, error in zip(observations, errors):
        point_id = int(observation["point3D_id"])
        grouped.setdefault(point_id, []).append(float(error))
    return grouped


def _track_lengths(observations: list[dict]) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for observation in observations:
        point_id = int(observation["point3D_id"])
        lengths[point_id] = lengths.get(point_id, 0) + 1
    return lengths
