from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

try:
    import ezdxf
except Exception:  # pragma: no cover - optional dependency
    ezdxf = None  # type: ignore[assignment]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DXF_EXTENSIONS = {".dxf"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | (DXF_EXTENSIONS if ezdxf is not None else set())
SUPPORTED_EXTENSIONS_TUPLE = tuple(sorted(SUPPORTED_EXTENSIONS))
MAX_ANALYSIS_SIDE = 560
EDGE_FEATURE_SIZE = (40, 40)
GRAY_FEATURE_SIZE = (20, 20)
BINARY_FEATURE_SIZE = (16, 16)
LINE_FEATURE_SIZE = (32, 32)
ORIENTATION_BINS = 12
LINE_ORIENTATION_BINS = 12
RADIAL_BINS = 16
ORIENTATION_ANALYSIS_SIDE = 240
RADIAL_ANALYSIS_SIDE = 96
HASH_SIZE = 8
HU_MOMENTS_COUNT = 7
FEATURE_VECTOR_LENGTH = (
    EDGE_FEATURE_SIZE[0] * EDGE_FEATURE_SIZE[1]
    + GRAY_FEATURE_SIZE[0] * GRAY_FEATURE_SIZE[1]
    + BINARY_FEATURE_SIZE[0] * BINARY_FEATURE_SIZE[1]
    + LINE_FEATURE_SIZE[0] * LINE_FEATURE_SIZE[1]
    + ORIENTATION_BINS
    + LINE_ORIENTATION_BINS
    + RADIAL_BINS
    + HU_MOMENTS_COUNT
)
HASH_VECTOR_LENGTH = HASH_SIZE * HASH_SIZE * 3


@dataclass
class ExtractedFeatures:
    path: str
    vector: np.ndarray
    ahash: np.ndarray


def is_supported_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_image_file(path: Path) -> bool:
    # Kept for backward compatibility with older call sites.
    return is_supported_file(path)


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

                        if entry.name.lower().endswith(SUPPORTED_EXTENSIONS_TUPLE):
                            yield Path(entry.path)
            except OSError:
                continue


def _safe_float(value) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _dxf_polyline_points(entity) -> tuple[list[tuple[float, float]], bool]:
    points: list[tuple[float, float]] = []
    closed = False
    kind = str(entity.dxftype()).upper()

    try:
        if kind == "LWPOLYLINE":
            for row in entity.get_points("xy"):
                px = _safe_float(row[0] if len(row) > 0 else None)
                py = _safe_float(row[1] if len(row) > 1 else None)
                if px is None or py is None:
                    continue
                points.append((px, py))
            closed = bool(getattr(entity, "closed", False))
        elif kind == "POLYLINE":
            for vertex in entity.vertices:
                px = _safe_float(vertex.dxf.x)
                py = _safe_float(vertex.dxf.y)
                if px is None or py is None:
                    continue
                points.append((px, py))
            closed = bool(getattr(entity, "is_closed", False))
    except Exception:
        return [], False

    return points, closed


def _sample_arc_points(
    cx: float,
    cy: float,
    radius: float,
    start_angle: float,
    end_angle: float,
    segments: int,
) -> list[tuple[float, float]]:
    if radius <= 0.0 or segments < 2:
        return []

    start = float(start_angle) % 360.0
    end = float(end_angle) % 360.0
    sweep = end - start
    if sweep <= 0.0:
        sweep += 360.0

    points: list[tuple[float, float]] = []
    for idx in range(segments):
        t = idx / float(segments - 1)
        angle = math.radians(start + sweep * t)
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _render_dxf_to_gray(path: Path, canvas_side: int = 960) -> np.ndarray | None:
    if ezdxf is None:
        return None

    try:
        document = ezdxf.readfile(str(path))
        modelspace = document.modelspace()
    except Exception:
        return None

    primitives: list[tuple] = []
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    def include_point(x: float, y: float) -> None:
        nonlocal min_x, min_y, max_x, max_y
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    for entity in modelspace:
        kind = str(entity.dxftype()).upper()

        if kind == "LINE":
            try:
                sx = _safe_float(entity.dxf.start.x)
                sy = _safe_float(entity.dxf.start.y)
                ex = _safe_float(entity.dxf.end.x)
                ey = _safe_float(entity.dxf.end.y)
            except Exception:
                continue
            if None in (sx, sy, ex, ey):
                continue
            p1 = (float(sx), float(sy))
            p2 = (float(ex), float(ey))
            include_point(*p1)
            include_point(*p2)
            primitives.append(("line", p1, p2))
            continue

        if kind in {"LWPOLYLINE", "POLYLINE"}:
            points, closed = _dxf_polyline_points(entity)
            if len(points) < 2:
                continue
            for point in points:
                include_point(*point)
            primitives.append(("polyline", points, closed))
            continue

        if kind == "CIRCLE":
            try:
                cx = _safe_float(entity.dxf.center.x)
                cy = _safe_float(entity.dxf.center.y)
                radius = _safe_float(entity.dxf.radius)
            except Exception:
                continue
            if None in (cx, cy, radius) or float(radius) <= 0.0:
                continue
            center = (float(cx), float(cy))
            rad = float(radius)
            include_point(center[0] - rad, center[1] - rad)
            include_point(center[0] + rad, center[1] + rad)
            primitives.append(("circle", center, rad))
            continue

        if kind == "ARC":
            try:
                cx = _safe_float(entity.dxf.center.x)
                cy = _safe_float(entity.dxf.center.y)
                radius = _safe_float(entity.dxf.radius)
                start_angle = _safe_float(entity.dxf.start_angle)
                end_angle = _safe_float(entity.dxf.end_angle)
            except Exception:
                continue
            if None in (cx, cy, radius, start_angle, end_angle) or float(radius) <= 0.0:
                continue
            center = (float(cx), float(cy))
            rad = float(radius)
            include_point(center[0] - rad, center[1] - rad)
            include_point(center[0] + rad, center[1] + rad)
            primitives.append(("arc", center, rad, float(start_angle), float(end_angle)))
            continue

        if kind == "SPLINE":
            points: list[tuple[float, float]] = []
            try:
                for row in entity.control_points:
                    px = _safe_float(row[0] if len(row) > 0 else None)
                    py = _safe_float(row[1] if len(row) > 1 else None)
                    if px is None or py is None:
                        continue
                    points.append((px, py))
            except Exception:
                points = []
            if len(points) < 2:
                continue
            for point in points:
                include_point(*point)
            primitives.append(("polyline", points, False))
            continue

    if not primitives:
        return None
    if not all(math.isfinite(v) for v in (min_x, min_y, max_x, max_y)):
        return None

    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    margin = max(16, int(canvas_side * 0.06))
    usable = max(64, canvas_side - 2 * margin)
    scale = min(usable / span_x, usable / span_y)
    if not math.isfinite(scale) or scale <= 0.0:
        return None

    width = max(96, int(round(span_x * scale)) + 2 * margin)
    height = max(96, int(round(span_y * scale)) + 2 * margin)
    canvas = np.zeros((height, width), dtype=np.uint8)
    thickness = max(1, int(round(min(width, height) / 320.0)))

    def to_pixel(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        px = int(round((x - min_x) * scale)) + margin
        py = int(round((max_y - y) * scale)) + margin
        return px, py

    for row in primitives:
        mode = row[0]
        if mode == "line":
            p1 = to_pixel(row[1])
            p2 = to_pixel(row[2])
            cv2.line(canvas, p1, p2, 255, thickness, cv2.LINE_AA)
            continue

        if mode == "polyline":
            points = row[1]
            closed = bool(row[2])
            pixels = [to_pixel(item) for item in points]
            if len(pixels) >= 2:
                poly = np.asarray(pixels, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [poly], closed, 255, thickness, cv2.LINE_AA)
            continue

        if mode == "circle":
            center = row[1]
            radius = float(row[2])
            center_px = to_pixel(center)
            radius_px = max(1, int(round(radius * scale)))
            cv2.circle(canvas, center_px, radius_px, 255, thickness, cv2.LINE_AA)
            continue

        if mode == "arc":
            center = row[1]
            radius = float(row[2])
            start_angle = float(row[3])
            end_angle = float(row[4])
            circumference = max(32, int(round(2.0 * math.pi * radius * scale / 6.0)))
            arc_points = _sample_arc_points(
                cx=center[0],
                cy=center[1],
                radius=radius,
                start_angle=start_angle,
                end_angle=end_angle,
                segments=circumference,
            )
            pixel_points = [to_pixel(item) for item in arc_points]
            if len(pixel_points) >= 2:
                poly = np.asarray(pixel_points, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [poly], False, 255, thickness, cv2.LINE_AA)

    if not np.any(canvas):
        return None
    return canvas


def _load_gray(image_path: Path) -> np.ndarray | None:
    if image_path.suffix.lower() in DXF_EXTENSIONS:
        return _render_dxf_to_gray(image_path)

    # Use imdecode(fromfile) so Windows Unicode paths (e.g. Persian names) work reliably.
    try:
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    except Exception:
        return None
    if image_bytes.size == 0:
        return None
    return cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)


def load_preview_gray(path: Path, max_side: int = 900) -> np.ndarray | None:
    gray = _load_gray(path)
    if gray is None:
        return None
    return _resize_longest_side(gray, max_side)


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


def _line_map_features(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    analysis_gray = _resize_longest_side(gray, ORIENTATION_ANALYSIS_SIDE)
    edges = cv2.Canny(analysis_gray, 45, 155)
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=max(24, int(min(analysis_gray.shape[:2]) * 0.12)),
        minLineLength=max(14, int(min(analysis_gray.shape[:2]) * 0.10)),
        maxLineGap=max(8, int(min(analysis_gray.shape[:2]) * 0.05)),
    )

    line_map = np.zeros_like(analysis_gray, dtype=np.uint8)
    line_orientation = np.zeros((LINE_ORIENTATION_BINS,), dtype=np.float32)

    if lines is not None:
        for row in lines.reshape(-1, 4):
            x1, y1, x2, y2 = [int(value) for value in row]
            cv2.line(line_map, (x1, y1), (x2, y2), 255, 1, cv2.LINE_AA)

            dx = float(x2 - x1)
            dy = float(y2 - y1)
            segment_length = float(np.hypot(dx, dy))
            if segment_length <= 0.0:
                continue
            angle = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
            bin_index = min(LINE_ORIENTATION_BINS - 1, int((angle / 180.0) * LINE_ORIENTATION_BINS))
            line_orientation[bin_index] += segment_length

    resized_line_map = cv2.resize(line_map, LINE_FEATURE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    orientation_norm = float(line_orientation.sum())
    if orientation_norm > 0.0:
        line_orientation /= orientation_norm

    return resized_line_map.reshape(-1), line_orientation


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
    line_map, line_orientation = _line_map_features(blurred)
    radial_density = _radial_density(foreground)
    hu = _hu_moments(foreground)

    vector = np.concatenate(
        [
            edge_small.reshape(-1) * 0.40,
            gray_small.reshape(-1) * 0.10,
            binary_small.reshape(-1) * 0.12,
            line_map * 0.26,
            orientation_hist * 0.05,
            line_orientation * 0.04,
            radial_density * 0.02,
            hu * 0.01,
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
    equalized = cv2.equalizeHist(gray)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    sharpened = cv2.addWeighted(equalized, 1.55, cv2.GaussianBlur(equalized, (0, 0), 1.1), -0.55, 0)
    _, otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [
        gray,
        equalized,
        denoised,
        sharpened,
        otsu,
        cv2.bitwise_not(otsu),
        cv2.rotate(gray, cv2.ROTATE_180),
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
