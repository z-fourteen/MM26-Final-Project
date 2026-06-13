from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np


_LIGHTGLUE_MATCHER_CACHE: dict[tuple, object] = {}


@dataclass(frozen=True)
class ImageFeatures:
    image_name: str
    feature_path: Path
    feature_type: str
    image_size: tuple[int, int]
    keypoints: np.ndarray
    descriptors: np.ndarray


@dataclass(frozen=True)
class PairMatchResult:
    image_name1: str
    image_name2: str
    feature_path1: Path
    feature_path2: Path
    matches: np.ndarray
    distances: np.ndarray
    ratios: np.ndarray
    feature_type1: str = "rootsift"
    feature_type2: str = "rootsift"
    matcher_backend: str = "rootsift_bf"

    @property
    def num_matches(self) -> int:
        return int(self.matches.shape[0])


def load_features(feature_path: Path) -> ImageFeatures:
    data = np.load(feature_path)
    image_size = data["image_size"].astype(np.int32, copy=False) if "image_size" in data.files else np.array([0, 0])
    return ImageFeatures(
        image_name=str(data["image_name"]),
        feature_path=feature_path,
        feature_type=str(data["feature_type"]) if "feature_type" in data.files else "rootsift",
        image_size=(int(image_size[0]), int(image_size[1])),
        keypoints=data["keypoints"].astype(np.float32, copy=False),
        descriptors=data["descriptors"].astype(np.float32, copy=False),
    )


def match_feature_pair(
    features1: ImageFeatures,
    features2: ImageFeatures,
    ratio_test: float = 0.8,
    mutual_check: bool = True,
) -> PairMatchResult:
    forward_matches, forward_distances, forward_ratios = _ratio_match(
        features1.descriptors,
        features2.descriptors,
        ratio_test=ratio_test,
    )

    if mutual_check and len(forward_matches):
        backward_matches, _backward_distances, _backward_ratios = _ratio_match(
            features2.descriptors,
            features1.descriptors,
            ratio_test=ratio_test,
        )
        backward_lookup = {
            (int(match[1]), int(match[0]))
            for match in backward_matches
        }
        keep_indices = [
            index
            for index, match in enumerate(forward_matches)
            if (int(match[0]), int(match[1])) in backward_lookup
        ]
        forward_matches = forward_matches[keep_indices]
        forward_distances = forward_distances[keep_indices]
        forward_ratios = forward_ratios[keep_indices]

    return PairMatchResult(
        image_name1=features1.image_name,
        image_name2=features2.image_name,
        feature_path1=features1.feature_path,
        feature_path2=features2.feature_path,
        matches=forward_matches.astype(np.int32, copy=False),
        distances=forward_distances.astype(np.float32, copy=False),
        ratios=forward_ratios.astype(np.float32, copy=False),
        feature_type1=features1.feature_type,
        feature_type2=features2.feature_type,
        matcher_backend="rootsift_bf",
    )


def _resolve_torch_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _import_lightglue_symbol(name: str):
    try:
        import lightglue
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        local_lightglue = repo_root / "third_party" / "LightGlue"
        if local_lightglue.exists():
            sys.path.insert(0, str(local_lightglue))
            import lightglue
        else:
            raise ModuleNotFoundError(
                "Official LightGlue is not importable. Install it with "
                "`python -m pip install -e third_party/LightGlue`."
            )
    return getattr(lightglue, name)


def match_feature_pair_lightglue(
    features1: ImageFeatures,
    features2: ImageFeatures,
    lightglue_features: str = "disk",
    filter_threshold: float = 0.1,
    depth_confidence: float = 0.95,
    width_confidence: float = 0.99,
    device: str = "auto",
) -> PairMatchResult:
    import torch

    if features1.feature_type != lightglue_features or features2.feature_type != lightglue_features:
        raise ValueError(
            "LightGlue matcher feature mismatch: "
            f"expected {lightglue_features}, got {features1.feature_type} and {features2.feature_type}"
        )

    torch_device = _resolve_torch_device(device)
    cache_key = (
        str(torch_device),
        lightglue_features,
        float(filter_threshold),
        float(depth_confidence),
        float(width_confidence),
    )
    matcher = _LIGHTGLUE_MATCHER_CACHE.get(cache_key)
    if matcher is None:
        LightGlue = _import_lightglue_symbol("LightGlue")
        matcher = LightGlue(
            features=lightglue_features,
            filter_threshold=float(filter_threshold),
            depth_confidence=float(depth_confidence),
            width_confidence=float(width_confidence),
        ).eval().to(torch_device)
        _LIGHTGLUE_MATCHER_CACHE[cache_key] = matcher

    with torch.no_grad():
        data = {
            "image0": _features_to_lightglue_dict(features1, torch_device),
            "image1": _features_to_lightglue_dict(features2, torch_device),
        }
        output = matcher(data)

    matches_tensor = output["matches"][0] if isinstance(output["matches"], list) else output["matches"][0]
    scores_tensor = output["scores"][0] if isinstance(output["scores"], list) else output["scores"][0]
    matches = matches_tensor.detach().cpu().numpy().astype(np.int32, copy=False)
    scores = scores_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    distances = (1.0 - scores).astype(np.float32, copy=False)

    return PairMatchResult(
        image_name1=features1.image_name,
        image_name2=features2.image_name,
        feature_path1=features1.feature_path,
        feature_path2=features2.feature_path,
        matches=matches,
        distances=distances,
        ratios=scores,
        feature_type1=features1.feature_type,
        feature_type2=features2.feature_type,
        matcher_backend="lightglue",
    )


def match_feature_pair_backend(
    features1: ImageFeatures,
    features2: ImageFeatures,
    matcher_backend: str = "rootsift_bf",
    ratio_test: float = 0.8,
    mutual_check: bool = True,
    lightglue_features: str = "disk",
    lightglue_filter_threshold: float = 0.1,
    lightglue_depth_confidence: float = 0.95,
    lightglue_width_confidence: float = 0.99,
    device: str = "auto",
) -> PairMatchResult:
    normalized_backend = matcher_backend.lower()
    if normalized_backend in {"rootsift_bf", "bf"}:
        return match_feature_pair(
            features1,
            features2,
            ratio_test=ratio_test,
            mutual_check=mutual_check,
        )
    if normalized_backend == "lightglue":
        return match_feature_pair_lightglue(
            features1,
            features2,
            lightglue_features=lightglue_features,
            filter_threshold=lightglue_filter_threshold,
            depth_confidence=lightglue_depth_confidence,
            width_confidence=lightglue_width_confidence,
            device=device,
        )
    raise ValueError(f"Unsupported matcher backend: {matcher_backend}")


def _features_to_lightglue_dict(features: ImageFeatures, device) -> dict:
    import torch

    keypoints = torch.from_numpy(features.keypoints[:, :2].astype(np.float32, copy=False))[None].to(device)
    descriptors = torch.from_numpy(features.descriptors.astype(np.float32, copy=False))[None].to(device)
    image_size = torch.tensor(features.image_size, dtype=torch.float32, device=device)[None]
    return {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "image_size": image_size,
    }


def _ratio_match(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    ratio_test: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if descriptors1.size == 0 or descriptors2.size == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    pairs: list[tuple[int, int]] = []
    distances: list[float] = []
    ratios: list[float] = []

    eps = 1e-12
    for candidates in knn_matches:
        if len(candidates) < 2:
            continue
        best, second_best = candidates
        ratio = best.distance / max(second_best.distance, eps)
        if ratio < ratio_test:
            pairs.append((best.queryIdx, best.trainIdx))
            distances.append(best.distance)
            ratios.append(ratio)

    return (
        np.asarray(pairs, dtype=np.int32),
        np.asarray(distances, dtype=np.float32),
        np.asarray(ratios, dtype=np.float32),
    )


def pair_output_path(match_dir: Path, feature_path1: Path, feature_path2: Path) -> Path:
    return match_dir / f"{feature_path1.stem}__{feature_path2.stem}.npz"


def save_pair_matches(result: PairMatchResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image_name1=result.image_name1,
        image_name2=result.image_name2,
        feature_path1=str(result.feature_path1),
        feature_path2=str(result.feature_path2),
        matches=result.matches,
        distances=result.distances,
        ratios=result.ratios,
        matcher_backend=result.matcher_backend,
        feature_type1=result.feature_type1,
        feature_type2=result.feature_type2,
    )


def draw_match_preview(
    image_path1: Path,
    image_path2: Path,
    features1: ImageFeatures,
    features2: ImageFeatures,
    matches: np.ndarray,
    output_path: Path,
    limit: int = 80,
) -> None:
    image1 = cv2.imread(str(image_path1), cv2.IMREAD_COLOR)
    image2 = cv2.imread(str(image_path2), cv2.IMREAD_COLOR)
    if image1 is None:
        raise ValueError(f"Failed to read image: {image_path1}")
    if image2 is None:
        raise ValueError(f"Failed to read image: {image_path2}")

    keypoints1 = [
        cv2.KeyPoint(float(x), float(y), float(size))
        for x, y, size, _angle, _response, _octave in features1.keypoints
    ]
    keypoints2 = [
        cv2.KeyPoint(float(x), float(y), float(size))
        for x, y, size, _angle, _response, _octave in features2.keypoints
    ]
    cv_matches = [
        cv2.DMatch(_queryIdx=int(query_idx), _trainIdx=int(train_idx), _distance=0.0)
        for query_idx, train_idx in matches[:limit]
    ]
    preview = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        cv_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)
