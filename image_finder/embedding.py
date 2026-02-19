from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]


DEFAULT_EMBEDDING_MODEL = "clip-ViT-B-32"
EMBEDDING_MODEL_ENV = "IMAGE_FINDER_EMBED_MODEL"

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, "ImageEmbeddingModel"] = {}


def faiss_available() -> bool:
    return faiss is not None


def embedding_available() -> bool:
    return SentenceTransformer is not None and Image is not None


def _resolve_model_name(model_name: str | None = None) -> str:
    if model_name and model_name.strip():
        return model_name.strip()
    configured = os.environ.get(EMBEDDING_MODEL_ENV, "").strip()
    if configured:
        return configured
    return DEFAULT_EMBEDDING_MODEL


class ImageEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed.")
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        raw_dimension = self._model.get_sentence_embedding_dimension()
        if raw_dimension is not None and int(raw_dimension) > 0:
            self.dimension = int(raw_dimension)
        else:
            probe = self._model.encode(
                ["image finder embedding dimension probe"],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            probe_array = np.asarray(probe, dtype=np.float32)
            if probe_array.ndim == 1:
                probe_array = probe_array.reshape(1, -1)
            if probe_array.ndim != 2 or probe_array.shape[1] <= 0:
                raise RuntimeError("Cannot infer embedding dimension for selected model.")
            self.dimension = int(probe_array.shape[1])

    def _load_rgb(self, image_path: Path):
        if Image is None:
            return None
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        except Exception:
            return None
        return image

    def encode_paths(self, paths: list[Path], batch_size: int = 16) -> dict[int, np.ndarray]:
        if not paths:
            return {}

        result: dict[int, np.ndarray] = {}
        chunk_size = max(8, int(batch_size) * 2)
        for chunk_start in range(0, len(paths), chunk_size):
            chunk_paths = paths[chunk_start : chunk_start + chunk_size]
            chunk_images: list[Any] = []
            chunk_positions: list[int] = []
            for offset, image_path in enumerate(chunk_paths):
                image = self._load_rgb(image_path)
                if image is None:
                    continue
                chunk_images.append(image)
                chunk_positions.append(chunk_start + offset)

            if not chunk_images:
                continue

            try:
                embeddings = self._model.encode(
                    chunk_images,
                    batch_size=max(1, int(batch_size)),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            except Exception:
                embeddings = None
            finally:
                for image in chunk_images:
                    try:
                        image.close()
                    except Exception:
                        pass

            if embeddings is None:
                continue

            vectors = np.asarray(embeddings, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            if vectors.ndim != 2:
                continue

            for row_index, absolute_index in enumerate(chunk_positions):
                vector = np.asarray(vectors[row_index], dtype=np.float32, copy=False)
                if vector.ndim != 1 or vector.shape[0] <= 0:
                    continue
                norm = float(np.linalg.norm(vector))
                if norm > 0:
                    vector = vector / norm
                result[absolute_index] = vector
        return result

    def encode_path(self, image_path: Path) -> np.ndarray | None:
        encoded = self.encode_paths([image_path], batch_size=1)
        return encoded.get(0)


def load_embedding_model(model_name: str | None = None) -> ImageEmbeddingModel | None:
    if not embedding_available():
        return None

    resolved = _resolve_model_name(model_name)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(resolved)
        if cached is not None:
            return cached

        try:
            loaded = ImageEmbeddingModel(model_name=resolved)
        except Exception:
            return None

        _MODEL_CACHE[resolved] = loaded
        return loaded


def build_faiss_index(vectors: np.ndarray) -> Any | None:
    if faiss is None:
        return None

    data = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if data.ndim != 2 or data.shape[0] <= 0 or data.shape[1] <= 0:
        return None

    index = faiss.IndexFlatIP(int(data.shape[1]))
    index.add(data)
    return index


def save_faiss_index(index: Any, output_path: Path) -> bool:
    if faiss is None:
        return False
    try:
        faiss.write_index(index, str(output_path))
    except Exception:
        return False
    return True


def load_faiss_index(index_path: Path) -> Any | None:
    if faiss is None or not index_path.exists():
        return None
    try:
        return faiss.read_index(str(index_path))
    except Exception:
        return None
