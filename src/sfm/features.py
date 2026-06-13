from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class FeatureExtractionResult:
    image_name: str
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
        image_size=np.array([original_width, original_height], dtype=np.int32),
        resized_size=np.array([resized_width, resized_height], dtype=np.int32),
        scale_factor=np.float32(scale),
        keypoints=keypoint_array,
        descriptors=descriptors,
        used_alpha_mask=np.bool_(has_alpha and use_alpha_mask),
        alpha_threshold=np.int32(alpha_threshold),
        mask_coverage=np.float32(mask_coverage),
    )

    return FeatureExtractionResult(
        image_name=image_path.name,
        image_size=(original_width, original_height),
        resized_size=(resized_width, resized_height),
        scale_factor=scale,
        num_keypoints=len(keypoint_array),
        used_alpha_mask=bool(has_alpha and use_alpha_mask),
        mask_coverage=mask_coverage,
        output_path=output_path,
    )


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
