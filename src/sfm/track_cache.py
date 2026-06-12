from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackObservation:
    image_name: str
    keypoint_idx: int


@dataclass(frozen=True)
class TrackBuildResult:
    tracks: list[list[TrackObservation]]
    valid_edges: set[tuple[str, str]]
    edge_status_counts: dict[str, int]
    edge_model_type_counts: dict[str, int]
    used_edges: int
    skipped_edges: int
    skipped_matches_by_residual: int


def registered_set_key(registered_names: set[str]) -> str:
    payload = "\n".join(sorted(registered_names)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def track_cache_path(cache_dir: Path, registered_names: set[str]) -> Path:
    return cache_dir / f"registered_tracks_{registered_set_key(registered_names)}.json"


def save_track_cache(cache_path: Path, result: TrackBuildResult, registered_names: set[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "registered_names": sorted(registered_names),
        "tracks": [
            [
                {"image_name": observation.image_name, "keypoint_idx": int(observation.keypoint_idx)}
                for observation in track
            ]
            for track in result.tracks
        ],
        "valid_edges": [list(edge) for edge in sorted(result.valid_edges)],
        "edge_status_counts": result.edge_status_counts,
        "edge_model_type_counts": result.edge_model_type_counts,
        "used_edges": result.used_edges,
        "skipped_edges": result.skipped_edges,
        "skipped_matches_by_residual": result.skipped_matches_by_residual,
    }
    cache_path.write_text(json.dumps(payload), encoding="utf-8")


def load_track_cache(cache_path: Path, registered_names: set[str]) -> TrackBuildResult | None:
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if set(payload.get("registered_names", [])) != set(registered_names):
        return None
    tracks = [
        [
            TrackObservation(
                image_name=str(observation["image_name"]),
                keypoint_idx=int(observation["keypoint_idx"]),
            )
            for observation in track
        ]
        for track in payload["tracks"]
    ]
    valid_edges = {tuple(edge) for edge in payload.get("valid_edges", [])}
    return TrackBuildResult(
        tracks=tracks,
        valid_edges=valid_edges,
        edge_status_counts=dict(payload.get("edge_status_counts", {})),
        edge_model_type_counts=dict(payload.get("edge_model_type_counts", {})),
        used_edges=int(payload.get("used_edges", 0)),
        skipped_edges=int(payload.get("skipped_edges", 0)),
        skipped_matches_by_residual=int(payload.get("skipped_matches_by_residual", 0)),
    )
