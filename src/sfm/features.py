from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_DISK_EXTRACTOR_CACHE: dict[tuple, object] = {}


@dataclass(frozen=True)
class FeatureExtractionResult:
    image_name: str
    feature_type: str
    image_size: tuple[int, int]
    resized_size: tuple[int, int]
    scale_factor: float
    num_keypoints: int
    used_alpha_mask: bool
    mask_coverage: float
    output_path: Path


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image_gray(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def read_image_gray_and_alpha_mask(image_path: Path, alpha_threshold: int = 1) -> tuple[np.ndarray, np.ndarray | None, bool]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if image.ndim == 2:
        return image, None, False
    if image.shape[2] == 4:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        alpha = image[:, :, 3]
        mask = np.where(alpha >= int(alpha_threshold), 255, 0).astype(np.uint8)
        return gray, mask, True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray, None, False


def read_image_rgb_and_alpha_mask(image_path: Path, alpha_threshold: int = 1) -> tuple[np.ndarray, np.ndarray | None, bool]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), None, False
    if image.shape[2] == 4:
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = image[:, :, 3]
        mask = np.where(alpha >= int(alpha_threshold), 255, 0).astype(np.uint8)
        return rgb, mask, True
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None, False


def resize_for_sfm(image: np.ndarray, max_image_size: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    max_side = max(height, width)
    if max_image_size <= 0 or max_side <= max_image_size:
        return image, 1.0

    scale = max_image_size / float(max_side)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def rootsift(descriptors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if descriptors is None or descriptors.size == 0:
        return np.empty((0, 128), dtype=np.float32)
    descriptors = descriptors.astype(np.float32, copy=False)
    l1_norm = np.sum(descriptors, axis=1, keepdims=True)
    descriptors = descriptors / np.maximum(l1_norm, eps)
    return np.sqrt(descriptors, out=descriptors).astype(np.float32, copy=False)


def keypoints_to_array(keypoints: list[cv2.KeyPoint], inv_scale: float) -> np.ndarray:
    data = np.empty((len(keypoints), 6), dtype=np.float32)
    for index, keypoint in enumerate(keypoints):
        data[index] = (
            keypoint.pt[0] * inv_scale,
            keypoint.pt[1] * inv_scale,
            keypoint.size * inv_scale,
            keypoint.angle,
            keypoint.response,
            float(keypoint.octave),
        )
    return data


def disk_keypoints_to_array(keypoints: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
    if keypoints.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    keypoints = keypoints.astype(np.float32, copy=False)
    data = np.empty((keypoints.shape[0], 6), dtype=np.float32)
    data[:, 0:2] = keypoints[:, 0:2]
    data[:, 2] = 1.0
    data[:, 3] = -1.0
    if scores is None or scores.size == 0:
        data[:, 4] = 1.0
    else:
        data[:, 4] = scores.astype(np.float32, copy=False)
    data[:, 5] = 0.0
    return data


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


def extract_rootsift(
    image_path: Path,
    feature_dir: Path,
    max_image_size: int = 1600,
    nfeatures: int = 8192,
    use_alpha_mask: bool = True,
    alpha_threshold: int = 1,
) -> FeatureExtractionResult:
    if use_alpha_mask:
        image, mask, has_alpha = read_image_gray_and_alpha_mask(image_path, alpha_threshold=alpha_threshold)
    else:
        image = read_image_gray(image_path)
        mask = None
        has_alpha = False
    original_height, original_width = image.shape[:2]
    resized, scale = resize_for_sfm(image, max_image_size=max_image_size)
    resized_mask = None
    if mask is not None:
        resized_mask, _mask_scale = resize_for_sfm(mask, max_image_size=max_image_size)
        resized_mask = np.where(resized_mask >= 128, 255, 0).astype(np.uint8)
    resized_height, resized_width = resized.shape[:2]
    mask_coverage = float(np.mean(resized_mask > 0)) if resized_mask is not None else 1.0

    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = sift.detectAndCompute(resized, resized_mask)
    descriptors = rootsift(descriptors)
    inv_scale = 1.0 / scale
    keypoint_array = keypoints_to_array(keypoints, inv_scale=inv_scale)

    feature_dir.mkdir(parents=True, exist_ok=True)
    output_path = feature_dir / f"{image_path.stem}.npz"
    np.savez_compressed(
        output_path,
        image_name=image_path.name,
        feature_type="rootsift",
        image_size=np.array([original_width, original_height], dtype=np.int32),
        resized_size=np.array([resized_width, resized_height], dtype=np.int32),
        scale_factor=np.float32(scale),
        keypoints=keypoint_array,
        descriptors=descriptors,
        keypoint_scores=keypoint_array[:, 4].astype(np.float32, copy=False),
        used_alpha_mask=np.bool_(has_alpha and use_alpha_mask),
        alpha_threshold=np.int32(alpha_threshold),
        mask_coverage=np.float32(mask_coverage),
    )

    return FeatureExtractionResult(
        image_name=image_path.name,
        feature_type="rootsift",
        image_size=(original_width, original_height),
        resized_size=(resized_width, resized_height),
        scale_factor=scale,
        num_keypoints=len(keypoint_array),
        used_alpha_mask=bool(has_alpha and use_alpha_mask),
        mask_coverage=mask_coverage,
        output_path=output_path,
    )


def extract_disk(
    image_path: Path,
    feature_dir: Path,
    max_image_size: int = 1600,
    max_num_keypoints: int | None = 4096,
    detection_threshold: float = 0.0,
    nms_window_size: int = 5,
    use_alpha_mask: bool = True,
    alpha_threshold: int = 1,
    device: str = "auto",
) -> FeatureExtractionResult:
    import torch

    DiskExtractor = _import_lightglue_symbol("DISK")
    image, mask, has_alpha = read_image_rgb_and_alpha_mask(image_path, alpha_threshold=alpha_threshold)
    original_height, original_width = image.shape[:2]
    mask_coverage = float(np.mean(mask > 0)) if use_alpha_mask and mask is not None else 1.0
    if use_alpha_mask and mask is not None:
        image = image.copy()
        image[mask == 0] = 0
    torch_image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    torch_device = _resolve_torch_device(device)
    cache_key = (
        str(torch_device),
        max_num_keypoints,
        float(detection_threshold),
        int(nms_window_size),
    )
    extractor = _DISK_EXTRACTOR_CACHE.get(cache_key)
    if extractor is None:
        extractor = DiskExtractor(
            max_num_keypoints=max_num_keypoints,
            detection_threshold=float(detection_threshold),
            nms_window_size=int(nms_window_size),
        ).eval().to(torch_device)
        _DISK_EXTRACTOR_CACHE[cache_key] = extractor

    with torch.no_grad():
        features = extractor.extract(torch_image.to(torch_device), resize=max_image_size)

    keypoints = features["keypoints"][0].detach().cpu().numpy().astype(np.float32, copy=False)
    descriptors = features["descriptors"][0].detach().cpu().numpy().astype(np.float32, copy=False)
    scores = features["keypoint_scores"][0].detach().cpu().numpy().astype(np.float32, copy=False)

    if use_alpha_mask and mask is not None and keypoints.size:
        xs = np.clip(np.rint(keypoints[:, 0]).astype(np.int32), 0, original_width - 1)
        ys = np.clip(np.rint(keypoints[:, 1]).astype(np.int32), 0, original_height - 1)
        keep = mask[ys, xs] > 0
        keypoints = keypoints[keep]
        descriptors = descriptors[keep]
        scores = scores[keep]

    keypoint_array = disk_keypoints_to_array(keypoints, scores=scores)

    feature_dir.mkdir(parents=True, exist_ok=True)
    output_path = feature_dir / f"{image_path.stem}.npz"
    np.savez_compressed(
        output_path,
        image_name=image_path.name,
        feature_type="disk",
        image_size=np.array([original_width, original_height], dtype=np.int32),
        resized_size=np.array([original_width, original_height], dtype=np.int32),
        scale_factor=np.float32(1.0),
        keypoints=keypoint_array,
        descriptors=descriptors,
        keypoint_scores=scores,
        used_alpha_mask=np.bool_(has_alpha and use_alpha_mask),
        alpha_threshold=np.int32(alpha_threshold),
        mask_coverage=np.float32(mask_coverage),
        disk_resize=np.int32(max_image_size),
        disk_max_num_keypoints=np.int32(-1 if max_num_keypoints is None else max_num_keypoints),
        disk_detection_threshold=np.float32(detection_threshold),
        disk_nms_window_size=np.int32(nms_window_size),
    )

    return FeatureExtractionResult(
        image_name=image_path.name,
        feature_type="disk",
        image_size=(original_width, original_height),
        resized_size=(original_width, original_height),
        scale_factor=1.0,
        num_keypoints=len(keypoint_array),
        used_alpha_mask=bool(has_alpha and use_alpha_mask),
        mask_coverage=mask_coverage,
        output_path=output_path,
    )


def extract_image_features(
    image_path: Path,
    feature_dir: Path,
    feature_backend: str = "rootsift",
    max_image_size: int = 1600,
    use_alpha_mask: bool = True,
    alpha_threshold: int = 1,
    disk_max_num_keypoints: int | None = 4096,
    disk_detection_threshold: float = 0.0,
    disk_nms_window_size: int = 5,
    device: str = "auto",
) -> FeatureExtractionResult:
    normalized_backend = feature_backend.lower()
    if normalized_backend == "rootsift":
        return extract_rootsift(
            image_path=image_path,
            feature_dir=feature_dir,
            max_image_size=max_image_size,
            use_alpha_mask=use_alpha_mask,
            alpha_threshold=alpha_threshold,
        )
    if normalized_backend == "disk":
        return extract_disk(
            image_path=image_path,
            feature_dir=feature_dir,
            max_image_size=max_image_size,
            max_num_keypoints=disk_max_num_keypoints,
            detection_threshold=disk_detection_threshold,
            nms_window_size=disk_nms_window_size,
            use_alpha_mask=use_alpha_mask,
            alpha_threshold=alpha_threshold,
            device=device,
        )
    raise ValueError(f"Unsupported feature backend: {feature_backend}")


def draw_keypoints_preview(image_path: Path, feature_path: Path, output_path: Path, limit: int = 500) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    features = np.load(feature_path)
    keypoints = features["keypoints"][:limit]
    cv_keypoints = [
        cv2.KeyPoint(float(x), float(y), float(size))
        for x, y, size, _angle, _response, _octave in keypoints
    ]
    preview = cv2.drawKeypoints(
        image,
        cv_keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)
