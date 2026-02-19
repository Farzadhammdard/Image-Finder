from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

from .embedding import build_faiss_index, load_embedding_model, save_faiss_index
from .features import FEATURE_VECTOR_LENGTH, HASH_VECTOR_LENGTH, extract_features, iter_images


INDEX_FEATURES_FILE = "features.npy"
INDEX_HASHES_FILE = "hashes.npy"
INDEX_PATHS_FILE = "paths.json"
INDEX_META_FILE = "meta.json"
INDEX_EMBEDDINGS_FILE = "embeddings.npy"
INDEX_EMBEDDINGS_META_FILE = "embeddings_meta.json"
INDEX_FAISS_FILE = "embeddings.faiss"
PROGRESS_EMIT_INTERVAL_SECONDS = 0.12
PROGRESS_EMIT_MIN_STEP = 16

ProgressCallback = Callable[[int, int, int, int], None]


def _default_worker_count() -> int:
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count))


def _safe_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return int(stat.st_mtime_ns), int(stat.st_size)


def _emit_progress(
    callback: ProgressCallback | None,
    processed: int,
    total: int,
    reused: int,
    failed: int,
) -> None:
    if callback is None:
        return
    callback(processed, total, reused, failed)


def compute_paths_digest(paths: list[str]) -> str:
    joined = "\n".join(paths).encode("utf-8", errors="replace")
    return hashlib.sha256(joined).hexdigest()


def _load_existing_meta(output_dir: Path) -> dict[str, tuple[int, int]]:
    meta_path = output_dir / INDEX_META_FILE
    if not meta_path.exists():
        return {}

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if isinstance(payload, dict):
        records = payload.get("files", [])
    elif isinstance(payload, list):
        records = payload
    else:
        records = []

    result: dict[str, tuple[int, int]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if not isinstance(path, str):
            continue
        mtime_ns = int(item.get("mtime_ns", 0))
        size = int(item.get("size", -1))
        if size < 0:
            continue
        result[path] = (mtime_ns, size)
    return result


def _load_existing_feature_map(output_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    try:
        vectors, hashes, paths = load_index(output_dir)
    except Exception:
        return {}

    if len(paths) != len(vectors) or len(paths) != len(hashes):
        return {}
    if vectors.ndim != 2 or hashes.ndim != 2:
        return {}
    if vectors.shape[1] != FEATURE_VECTOR_LENGTH or hashes.shape[1] != HASH_VECTOR_LENGTH:
        return {}

    feature_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for idx, path in enumerate(paths):
        vector = np.asarray(vectors[idx], dtype=np.float32)
        ahash = np.asarray(hashes[idx], dtype=np.uint8)
        if vector.ndim != 1 or ahash.ndim != 1:
            continue
        if vector.shape[0] != FEATURE_VECTOR_LENGTH or ahash.shape[0] != HASH_VECTOR_LENGTH:
            continue
        feature_map[path] = (vector, ahash)
    return feature_map


def _load_existing_embedding_map(output_dir: Path) -> dict[str, np.ndarray]:
    embeddings_path = output_dir / INDEX_EMBEDDINGS_FILE
    paths_path = output_dir / INDEX_PATHS_FILE
    if not embeddings_path.exists() or not paths_path.exists():
        return {}

    try:
        raw_paths = json.loads(paths_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path, mmap_mode="r")
    except Exception:
        return {}

    if not isinstance(raw_paths, list):
        return {}
    if embeddings.ndim != 2 or embeddings.shape[0] != len(raw_paths):
        return {}

    result: dict[str, np.ndarray] = {}
    for idx, raw_path in enumerate(raw_paths):
        if not isinstance(raw_path, str):
            continue
        vector = np.asarray(embeddings[idx], dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] <= 0:
            continue
        result[raw_path] = vector
    return result


def build_index(
    folders: list[Path],
    output_dir: Path,
    workers: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted({path for path in iter_images(folders)}, key=lambda item: str(item).lower())
    if not image_paths:
        raise RuntimeError("No readable image or DXF files found in provided folders.")

    total = len(image_paths)
    processed = 0
    reused_count = 0
    failed_count = 0
    last_progress_emit_at = 0.0
    last_progress_emit_processed = -1

    def emit_progress(force: bool = False) -> None:
        nonlocal last_progress_emit_at, last_progress_emit_processed
        if progress_callback is None:
            return

        now = time.perf_counter()
        step_delta = processed - last_progress_emit_processed
        interval = now - last_progress_emit_at

        should_emit = force or processed >= total
        if not should_emit and (
            step_delta >= PROGRESS_EMIT_MIN_STEP or interval >= PROGRESS_EMIT_INTERVAL_SECONDS
        ):
            should_emit = True
        if not should_emit:
            return

        _emit_progress(progress_callback, processed, total, reused_count, failed_count)
        last_progress_emit_at = now
        last_progress_emit_processed = processed

    emit_progress(force=True)

    existing_meta = _load_existing_meta(output_dir)
    existing_features = _load_existing_feature_map(output_dir)
    existing_embeddings = _load_existing_embedding_map(output_dir)

    reused: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    pending: list[tuple[Path, str, tuple[int, int]]] = []

    for image_path in image_paths:
        key = str(image_path)
        signature = _safe_signature(image_path)
        if signature is None:
            processed += 1
            failed_count += 1
            emit_progress()
            continue

        cached = existing_features.get(key)
        if cached is not None and existing_meta.get(key) == signature:
            vector, ahash = cached
            reused[key] = (
                np.asarray(vector, dtype=np.float32),
                np.asarray(ahash, dtype=np.uint8),
                signature,
            )
            processed += 1
            reused_count += 1
            emit_progress()
            continue

        pending.append((image_path, key, signature))

    fresh: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    max_workers = workers if workers is not None and workers > 0 else _default_worker_count()
    if pending:
        max_workers = max(1, min(int(max_workers), len(pending)))

    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            tasks = {
                pool.submit(extract_features, image_path): (key, signature)
                for image_path, key, signature in pending
            }
            for task in as_completed(tasks):
                key, signature = tasks[task]
                try:
                    extracted = task.result()
                except Exception:
                    extracted = None

                processed += 1
                if extracted is None:
                    failed_count += 1
                else:
                    vector = extracted.vector.astype(np.float32, copy=False)
                    ahash = extracted.ahash.astype(np.uint8, copy=False)
                    if vector.ndim != 1 or ahash.ndim != 1:
                        failed_count += 1
                    elif vector.shape[0] != FEATURE_VECTOR_LENGTH or ahash.shape[0] != HASH_VECTOR_LENGTH:
                        failed_count += 1
                    else:
                        fresh[key] = (vector, ahash, signature)
                emit_progress()

    vectors: list[np.ndarray] = []
    hashes: list[np.ndarray] = []
    paths: list[str] = []
    meta_records: list[dict[str, int | str]] = []

    for image_path in image_paths:
        key = str(image_path)
        record = reused.get(key) or fresh.get(key)
        if record is None:
            continue

        vector, ahash, signature = record
        if vector.shape[0] != FEATURE_VECTOR_LENGTH or ahash.shape[0] != HASH_VECTOR_LENGTH:
            continue
        mtime_ns, size = signature
        vectors.append(np.asarray(vector, dtype=np.float32))
        hashes.append(np.asarray(ahash, dtype=np.uint8))
        paths.append(key)
        meta_records.append({"path": key, "mtime_ns": mtime_ns, "size": size})

    if not vectors:
        raise RuntimeError("No readable image or DXF files found in provided folders.")

    vector_array = np.stack(vectors).astype(np.float32, copy=False)
    hash_array = np.stack(hashes).astype(np.uint8, copy=False)

    np.save(output_dir / INDEX_FEATURES_FILE, vector_array)
    np.save(output_dir / INDEX_HASHES_FILE, hash_array)
    (output_dir / INDEX_PATHS_FILE).write_text(json.dumps(paths, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / INDEX_META_FILE).write_text(
        json.dumps({"version": 5, "files": meta_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    current_meta = {str(item["path"]): (int(item["mtime_ns"]), int(item["size"])) for item in meta_records}
    embedding_reused_count = 0
    embedding_fresh_count = 0
    embedding_ready = False

    embedding_model = load_embedding_model()
    if embedding_model is not None:
        embeddings_ordered: list[np.ndarray | None] = [None] * len(paths)
        pending_positions: list[int] = []
        pending_paths: list[Path] = []

        for idx, path in enumerate(paths):
            cached = existing_embeddings.get(path)
            signature = current_meta.get(path)
            if cached is not None and signature is not None and existing_meta.get(path) == signature:
                embeddings_ordered[idx] = np.asarray(cached, dtype=np.float32)
                embedding_reused_count += 1
                continue
            pending_positions.append(idx)
            pending_paths.append(Path(path))

        if pending_paths:
            encoded = embedding_model.encode_paths(pending_paths, batch_size=16)
            for local_idx, vector in encoded.items():
                if local_idx < 0 or local_idx >= len(pending_positions):
                    continue
                absolute_idx = pending_positions[local_idx]
                embeddings_ordered[absolute_idx] = np.asarray(vector, dtype=np.float32, copy=False)
                embedding_fresh_count += 1

        if all(vector is not None for vector in embeddings_ordered):
            embedding_array = np.stack(
                [np.asarray(vector, dtype=np.float32, copy=False) for vector in embeddings_ordered]
            ).astype(np.float32, copy=False)
            norms = np.linalg.norm(embedding_array, axis=1, keepdims=True)
            np.divide(embedding_array, np.clip(norms, 1e-12, None), out=embedding_array)

            np.save(output_dir / INDEX_EMBEDDINGS_FILE, embedding_array)
            (output_dir / INDEX_EMBEDDINGS_META_FILE).write_text(
                json.dumps(
                    {
                        "version": 1,
                        "model": embedding_model.model_name,
                        "dimension": int(embedding_array.shape[1]),
                        "count": int(embedding_array.shape[0]),
                        "paths_sha256": compute_paths_digest(paths),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            faiss_index = build_faiss_index(embedding_array)
            if faiss_index is not None:
                save_faiss_index(faiss_index, output_dir / INDEX_FAISS_FILE)
            embedding_ready = True

    emit_progress(force=True)

    return {
        "indexed": len(paths),
        "total": total,
        "processed": processed,
        "reused": reused_count,
        "failed": failed_count,
        "embedding_ready": int(embedding_ready),
        "embedding_reused": embedding_reused_count,
        "embedding_indexed": embedding_fresh_count,
    }


def load_index(index_dir: Path, use_mmap: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features_path = index_dir / INDEX_FEATURES_FILE
    hashes_path = index_dir / INDEX_HASHES_FILE
    paths_path = index_dir / INDEX_PATHS_FILE

    if not features_path.exists() or not hashes_path.exists() or not paths_path.exists():
        raise FileNotFoundError(f"Index files not found in: {index_dir}")

    mmap_mode = "r" if use_mmap else None
    vectors = np.load(features_path, mmap_mode=mmap_mode)
    hashes = np.load(hashes_path, mmap_mode=mmap_mode)
    paths = json.loads(paths_path.read_text(encoding="utf-8"))

    return vectors, hashes, paths


def load_embeddings(index_dir: Path, paths: list[str], use_mmap: bool = False) -> tuple[np.ndarray | None, str]:
    embeddings_path = index_dir / INDEX_EMBEDDINGS_FILE
    embeddings_meta_path = index_dir / INDEX_EMBEDDINGS_META_FILE

    if not embeddings_path.exists() or not embeddings_meta_path.exists():
        return None, ""

    try:
        payload = json.loads(embeddings_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None, ""

    if not isinstance(payload, dict):
        return None, ""
    model_name = str(payload.get("model", ""))
    expected_digest = str(payload.get("paths_sha256", ""))
    if expected_digest != compute_paths_digest(paths):
        return None, model_name

    mmap_mode = "r" if use_mmap else None
    try:
        vectors = np.load(embeddings_path, mmap_mode=mmap_mode)
    except Exception:
        return None, model_name

    if vectors.ndim != 2 or vectors.shape[0] != len(paths):
        return None, model_name

    return vectors, model_name
