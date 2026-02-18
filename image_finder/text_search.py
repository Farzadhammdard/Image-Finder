from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover - optional runtime dependency
    RapidOCR = None  # type: ignore[assignment]


OCR_CACHE_FILE = "ocr_text_cache.json"
MIN_OCR_CONFIDENCE = 0.45
MAX_TEXT_LENGTH = 260
MAX_OCR_VARIANTS = 6
TEXT_NORMALIZATION_MAP = str.maketrans(
    {
        "ي": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
    }
)
NON_WORD_PATTERN = re.compile(r"[^\w\s]+", flags=re.UNICODE)


def _normalize_text(value: str) -> str:
    normalized = value.translate(TEXT_NORMALIZATION_MAP)
    normalized = NON_WORD_PATTERN.sub(" ", normalized)
    compact = re.sub(r"\s+", " ", normalized).strip()
    return compact.casefold()


def _clean_text(value: str) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= MAX_TEXT_LENGTH:
        return compact
    return compact[: MAX_TEXT_LENGTH - 3] + "..."


def _read_grayscale(image_path: Path) -> np.ndarray | None:
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    if image_bytes.size == 0:
        return None
    return cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)


def _signature(image_path: Path) -> tuple[int, int] | None:
    try:
        stat = image_path.stat()
    except OSError:
        return None
    return int(stat.st_mtime_ns), int(stat.st_size)


def _token_set(value: str) -> set[str]:
    return {token for token in value.split(" ") if token}


def _char_ngram_set(value: str, n: int = 3) -> set[str]:
    compact = value.replace(" ", "")
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[idx : idx + n] for idx in range(0, len(compact) - n + 1)}


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union_count = len(left | right)
    if union_count == 0:
        return 0.0
    return len(left & right) / float(union_count)


def text_similarity(query_text: str, candidate_text: str) -> float:
    normalized_query = _normalize_text(query_text)
    normalized_candidate = _normalize_text(candidate_text)
    if not normalized_query or not normalized_candidate:
        return 0.0

    ratio_score = fuzz.ratio(normalized_query, normalized_candidate) / 100.0
    partial_score = fuzz.partial_ratio(normalized_query, normalized_candidate) / 100.0
    token_score = fuzz.token_set_ratio(normalized_query, normalized_candidate) / 100.0

    query_tokens = _token_set(normalized_query)
    candidate_tokens = _token_set(normalized_candidate)
    token_overlap = _overlap_ratio(query_tokens, candidate_tokens)

    query_ngrams = _char_ngram_set(normalized_query, n=3)
    candidate_ngrams = _char_ngram_set(normalized_candidate, n=3)
    ngram_overlap = _overlap_ratio(query_ngrams, candidate_ngrams)

    containment = 0.0
    if normalized_query in normalized_candidate or normalized_candidate in normalized_query:
        shorter = min(len(normalized_query), len(normalized_candidate))
        longer = max(len(normalized_query), len(normalized_candidate))
        if longer > 0:
            containment = shorter / float(longer)

    blended = (
        0.34 * ratio_score
        + 0.24 * partial_score
        + 0.24 * token_score
        + 0.10 * token_overlap
        + 0.08 * ngram_overlap
    )
    return max(blended, token_score, partial_score * 0.92, containment)


class TextSearchCache:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.cache_path = self.index_dir / OCR_CACHE_FILE
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, int | str]] = {}
        self._loaded = False
        self._dirty = False
        self._ocr_engine = None

    @property
    def available(self) -> bool:
        return RapidOCR is not None

    def reset(self) -> None:
        with self._lock:
            self._records = {}
            self._loaded = False
            self._dirty = False

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return

            self._records = {}
            if self.cache_path.exists():
                try:
                    payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        entries = payload.get("files", [])
                    elif isinstance(payload, list):
                        entries = payload
                    else:
                        entries = []
                    for item in entries:
                        if not isinstance(item, dict):
                            continue
                        path = item.get("path")
                        if not isinstance(path, str):
                            continue
                        self._records[path] = {
                            "mtime_ns": int(item.get("mtime_ns", 0)),
                            "size": int(item.get("size", -1)),
                            "text": str(item.get("text", "")),
                        }
                except Exception:
                    self._records = {}
            self._loaded = True

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return

            rows: list[dict[str, int | str]] = []
            for path, item in self._records.items():
                rows.append(
                    {
                        "path": path,
                        "mtime_ns": int(item.get("mtime_ns", 0)),
                        "size": int(item.get("size", 0)),
                        "text": str(item.get("text", "")),
                    }
                )

            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps({"version": 1, "files": rows}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._dirty = False

    def _get_ocr_engine(self):
        if RapidOCR is None:
            return None
        with self._lock:
            if self._ocr_engine is None:
                self._ocr_engine = RapidOCR()
            return self._ocr_engine

    def _build_ocr_variants(self, gray: np.ndarray) -> list[np.ndarray]:
        variants: list[np.ndarray] = [gray]

        if min(gray.shape[:2]) < 720:
            upscaled = cv2.resize(gray, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
            variants.append(upscaled)

        sharpen = cv2.addWeighted(gray, 1.6, cv2.GaussianBlur(gray, (0, 0), 1.2), -0.6, 0)
        variants.append(sharpen)

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        variants.extend([otsu, cv2.bitwise_not(otsu), adaptive, cv2.bitwise_not(adaptive)])

        unique: list[np.ndarray] = []
        seen: set[tuple[int, int, int, int]] = set()
        for variant in variants:
            key = (
                int(variant.shape[0]),
                int(variant.shape[1]),
                int(float(variant.mean())),
                int(float(variant.std())),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(variant)

        return unique[:MAX_OCR_VARIANTS]

    def _run_ocr(self, image_path: Path) -> str:
        if RapidOCR is None:
            return ""
        gray = _read_grayscale(image_path)
        if gray is None:
            return ""

        engine = self._get_ocr_engine()
        if engine is None:
            return ""

        variants = self._build_ocr_variants(gray)

        best_text = ""
        best_score = 0.0
        for variant in variants:
            try:
                result, _elapsed = engine(variant)
            except Exception:
                continue
            if not result:
                continue

            chunks: list[str] = []
            confidences: list[float] = []
            for row in result:
                if not isinstance(row, (list, tuple)) or len(row) < 3:
                    continue
                text_raw = str(row[1]).strip()
                if not text_raw:
                    continue
                try:
                    confidence = float(row[2])
                except Exception:
                    confidence = 0.0
                if confidence < MIN_OCR_CONFIDENCE:
                    continue
                chunks.append(text_raw)
                confidences.append(confidence)

            merged = _clean_text(" ".join(chunks))
            if not merged:
                continue

            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            variant_score = avg_conf + min(len(merged), 120) / 360.0
            if variant_score > best_score:
                best_score = variant_score
                best_text = merged
            if best_score >= 1.08 and len(best_text) >= 24:
                break

        return best_text

    def extract_query_text(self, image_path: Path) -> str:
        return self._run_ocr(image_path)

    def get_index_text(self, image_path: Path, build_on_miss: bool = True) -> str:
        self._ensure_loaded()

        signature = _signature(image_path)
        if signature is None:
            return ""
        mtime_ns, size = signature
        key = str(image_path)

        with self._lock:
            row = self._records.get(key)
            if row and int(row.get("mtime_ns", -1)) == mtime_ns and int(row.get("size", -1)) == size:
                return str(row.get("text", ""))

        if not build_on_miss:
            return ""

        extracted = self._run_ocr(image_path)
        with self._lock:
            self._records[key] = {
                "mtime_ns": mtime_ns,
                "size": size,
                "text": extracted,
            }
            self._dirty = True
        return extracted
