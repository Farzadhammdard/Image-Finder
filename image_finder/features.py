from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_EXTENSIONS_TUPLE = tuple(sorted(IMAGE_EXTENSIONS))
MAX_ANALYSIS_SIDE = 560
EDGE_FEATURE_SIZE = (40, 40)
GRAY_FEATURE_SIZE = (20, 20)
BINARY_FEATURE_SIZE = (16, 16)
ORIENTATION_BINS = 12
RADIAL_BINS = 16
ORIENTATION_ANALYSIS_SIDE = 240
RADIAL_ANALYSIS_SIDE = 96
HASH_SIZE = 8
HU_MOMENTS_COUNT = 7
FEATURE_VECTOR_LENGTH = (
    EDGE_FEATURE_SIZE[0] * EDGE_FEATURE_SIZE[1]
    + GRAY_FEATURE_SIZE[0] * GRAY_FEATURE_SIZE[1]
    + BINARY_FEATURE_SIZE[0] * BINARY_FEATURE_SIZE[1]
    + ORIENTATION_BINS
    + RADIAL_BINS
    + HU_MOMENTS_COUNT
)
HASH_VECTOR_LENGTH = HASH_SIZE * HASH_SIZE * 3


@dataclass
class ExtractedFeatures:
    path: str
    vector: np.ndarray
    ahash: np.ndarray


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def iter_images(folders: Iterable[Path]) -> Iterable[Path]:
    for folder in folders:
        root = Path(folder)
        if not root.exists():
            continue
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                        except OSError:
                            continue

                        if entry.name.lower().endswith(IMAGE_EXTENSIONS_TUPLE):
                            yield Path(entry.path)
            except OSError:
                continue


def _load_gray(image_path: Path) -> np.ndarray | None:
    # Use imdecode(fromfile) so Windows Unicode paths (e.g. Persian names) work reliably.
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    if image_bytes.size == 0:
        return None
    return cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)


def _average_hash(gray: np.ndarray, hash_size: int = HASH_SIZE) -> np.ndarray:
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    threshold = float(resized.mean())
    return (resized > threshold).astype(np.uint8).reshape(-1)


def _difference_hash(gray: np.ndarray, hash_size: int = HASH_SIZE) -> np.ndarray:
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (resized[:, 1:] > resized[:, :-1]).astype(np.uint8).reshape(-1)


def _perceptual_hash(gray: np.ndarray, hash_size: int = HASH_SIZE, highfreq_factor: int = 4) -> np.ndarray:
    side = hash_size * highfreq_factor
    resized = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low_freq = dct[:hash_size, :hash_size]
    flattened = low_freq.reshape(-1)
    threshold = float(np.median(flattened[1:])) if flattened.size > 1 else float(flattened[0])
    return (low_freq > threshold).astype(np.uint8).reshape(-1)


def _resize_longest_side(gray: np.ndarray, max_side: int) -> np.ndarray:
    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return gray

    scale = max_side / float(longest)
    new_width = max(24, int(width * scale))
    new_height = max(24, int(height * scale))
    return cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _prepare_gray(gray: np.ndarray) -> np.ndarray:
    return _resize_longest_side(gray, MAX_ANALYSIS_SIDE)


def _choose_foreground(binary: np.ndarray) -> np.ndarray:
    white_ratio = float(np.count_nonzero(binary)) / float(binary.size)
    if white_ratio > 0.5:
        return cv2.bitwise_not(binary)
    return binary


def _orientation_hist(gray: np.ndarray, bins: int = ORIENTATION_BINS) -> np.ndarray:
    analysis_gray = _resize_longest_side(gray, ORIENTATION_ANALYSIS_SIDE)
    gx = cv2.Sobel(analysis_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(analysis_gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    orientation = cv2.phase(gx, gy, angleInDegrees=True)
    orientation = np.mod(orientation, 180.0)
    hist, _ = np.histogram(orientation, bins=bins, range=(0.0, 180.0), weights=magnitude)
    hist = hist.astype(np.float32)
    norm = float(hist.sum())
    if norm > 0:
        hist /= norm
    return hist


def _radial_density(binary: np.ndarray, bins: int = RADIAL_BINS) -> np.ndarray:
    analysis_binary = _resize_longest_side(binary, RADIAL_ANALYSIS_SIDE)
    mask = (analysis_binary > 0).astype(np.float32)
    height, width = mask.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    distances = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = float(distances.max())
    if max_dist <= 0.0:
        return np.zeros((bins,), dtype=np.float32)
    normalized = distances / max_dist
    hist, _ = np.histogram(normalized, bins=bins, range=(0.0, 1.0), weights=mask)
    hist = hist.astype(np.float32)
    norm = float(hist.sum())
    if norm > 0:
        hist /= norm
    return hist


def _hu_moments(binary: np.ndarray) -> np.ndarray:
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).reshape(-1)
    hu = -np.sign(hu) * np.log10(np.clip(np.abs(hu), 1e-12, None))
    hu = np.nan_to_num(hu, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    norm = float(np.linalg.norm(hu))
    if norm > 0:
        hu /= norm
    return hu


def _build_feature_vector(gray: np.ndarray) -> np.ndarray:
    normalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(normalized, (3, 3), 0)

    edges = cv2.Canny(blurred, 55, 170)

    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = _choose_foreground(otsu)

    edge_small = cv2.resize(edges, EDGE_FEATURE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gray_small = cv2.resize(blurred, GRAY_FEATURE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    binary_small = (
        cv2.resize(foreground, BINARY_FEATURE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    )

    orientation_hist = _orientation_hist(blurred)
    radial_density = _radial_density(foreground)
    hu = _hu_moments(foreground)

    vector = np.concatenate(
        [
            edge_small.reshape(-1) * 0.56,
            gray_small.reshape(-1) * 0.16,
            binary_small.reshape(-1) * 0.16,
            orientation_hist * 0.07,
            radial_density * 0.03,
            hu * 0.02,
        ]
    ).astype(np.float32)

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm

    return vector


def _build_hash_vector(gray: np.ndarray) -> np.ndarray:
    normalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
    ahash = _average_hash(blurred)
    dhash = _difference_hash(blurred)
    phash = _perceptual_hash(blurred)
    return np.concatenate([ahash, dhash, phash]).astype(np.uint8)


def _extract_from_gray(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prepared = _prepare_gray(gray)
    vector = _build_feature_vector(prepared)
    ahash = _build_hash_vector(prepared)
    return vector, ahash


def extract_features(image_path: Path) -> ExtractedFeatures | None:
    gray = _load_gray(image_path)
    if gray is None:
        return None
    vector, ahash = _extract_from_gray(gray)
    return ExtractedFeatures(path=str(image_path), vector=vector, ahash=ahash)


def _query_variant_images(gray: np.ndarray) -> list[np.ndarray]:
    return [
        gray,
        cv2.rotate(gray, cv2.ROTATE_180),
        cv2.bitwise_not(gray),
    ]


def extract_query_feature_variants(
    image_path: Path,
    max_variants: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    gray = _load_gray(image_path)
    if gray is None:
        raise ValueError(f"Cannot read image: {image_path}")

    variants: list[tuple[np.ndarray, np.ndarray]] = []
    for variant in _query_variant_images(gray)[: max(1, int(max_variants))]:
        variants.append(_extract_from_gray(variant))
    return variants


def extract_query_features(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    variants = extract_query_feature_variants(image_path=image_path, max_variants=1)
    return variants[0]


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    distance = int(np.count_nonzero(a != b))
    return 1.0 - (distance / float(a.size))
