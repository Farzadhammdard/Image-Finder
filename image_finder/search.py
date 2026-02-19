from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .embedding import load_embedding_model, load_faiss_index
from .features import extract_query_feature_variants
from .indexer import INDEX_FAISS_FILE, load_embeddings, load_index
from .text_search import TextSearchCache, text_similarity


@dataclass
class SearchResult:
    path: str
    score: float
    vector_score: float
    hash_score: float
    embedding_score: float = 0.0
    text_score: float = 0.0
    text_excerpt: str = ""


@dataclass
class LoadedIndex:
    vectors: np.ndarray
    hashes: np.ndarray
    paths: list[str]
    embedding_vectors: np.ndarray | None = None
    faiss_index: Any | None = None
    embedding_model_name: str = ""


@dataclass
class SearchContext:
    query_text: str
    used_text_rerank: bool
    used_ai_embedding: bool = False


def load_runtime_index(index_dir: Path) -> LoadedIndex:
    vectors, hashes, paths = load_index(index_dir=index_dir, use_mmap=False)
    vectors_array = np.ascontiguousarray(vectors, dtype=np.float32)
    hashes_array = np.ascontiguousarray(hashes, dtype=np.uint8)

    embedding_vectors_raw, embedding_model_name = load_embeddings(index_dir=index_dir, paths=paths, use_mmap=False)
    embedding_vectors = None
    faiss_index = None
    if embedding_vectors_raw is not None:
        candidate = np.ascontiguousarray(embedding_vectors_raw, dtype=np.float32)
        if candidate.ndim == 2 and candidate.shape[0] == vectors_array.shape[0]:
            embedding_vectors = candidate
            loaded_faiss = load_faiss_index(index_dir / INDEX_FAISS_FILE)
            if loaded_faiss is not None:
                try:
                    if int(getattr(loaded_faiss, "ntotal", -1)) == int(candidate.shape[0]):
                        faiss_index = loaded_faiss
                except Exception:
                    faiss_index = None

    return LoadedIndex(
        vectors=vectors_array,
        hashes=hashes_array,
        paths=paths,
        embedding_vectors=embedding_vectors,
        faiss_index=faiss_index,
        embedding_model_name=embedding_model_name,
    )


def _top_sorted_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if scores.ndim != 1:
        raise ValueError("Score vector must be one-dimensional.")
    if scores.size == 0:
        return np.array([], dtype=np.int64)

    limited_k = max(1, min(int(top_k), int(scores.size)))
    partitioned = np.argpartition(scores, -limited_k)[-limited_k:]
    return partitioned[np.argsort(scores[partitioned])[::-1]]


def _compute_local_scores(
    vectors: np.ndarray,
    hashes: np.ndarray,
    variants: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    vector_scores = np.full((vectors.shape[0],), -1.0, dtype=np.float32)
    hash_scores = np.full((hashes.shape[0],), 0.0, dtype=np.float32)
    matched_variant = False

    for variant_vector, variant_hash in variants:
        local_vector = np.asarray(variant_vector, dtype=np.float32, copy=False)
        local_hash = np.asarray(variant_hash, dtype=np.uint8, copy=False)
        if local_vector.shape[0] != vectors.shape[1] or local_hash.shape[0] != hashes.shape[1]:
            continue
        matched_variant = True
        current_vector_scores = vectors @ local_vector
        hash_mismatch = np.count_nonzero(hashes != local_hash, axis=1)
        current_hash_scores = 1.0 - (hash_mismatch.astype(np.float32) / float(hashes.shape[1]))
        vector_scores = np.maximum(vector_scores, current_vector_scores)
        hash_scores = np.maximum(hash_scores, current_hash_scores)

    if not matched_variant:
        raise ValueError("Index format changed. Click Rebuild Desktop Index once.")
    return vector_scores, hash_scores


def _extract_query_embedding(query_image: Path, expected_dim: int) -> np.ndarray | None:
    model = load_embedding_model()
    if model is None:
        return None
    if int(model.dimension) != int(expected_dim):
        return None

    vector = model.encode_path(query_image)
    if vector is None:
        return None

    query_vector = np.asarray(vector, dtype=np.float32, copy=False).reshape(-1)
    if query_vector.ndim != 1 or query_vector.shape[0] != expected_dim:
        return None

    norm = float(np.linalg.norm(query_vector))
    if norm > 0:
        query_vector = query_vector / norm
    return query_vector


def _search_with_faiss(
    faiss_index: Any,
    query_embedding: np.ndarray,
    total_items: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    pool_size = max(int(top_k) * 24, 512)
    pool_size = max(1, min(pool_size, int(total_items)))
    try:
        distances, indices = faiss_index.search(
            np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32),
            pool_size,
        )
    except Exception:
        return None

    if indices.ndim != 2 or distances.ndim != 2 or indices.shape[0] == 0:
        return None

    merged: dict[int, float] = {}
    for raw_idx, raw_score in zip(indices[0], distances[0]):
        item = int(raw_idx)
        if item < 0 or item >= total_items:
            continue
        score = float(raw_score)
        existing = merged.get(item)
        if existing is None or score > existing:
            merged[item] = score

    if not merged:
        return None

    ordered = sorted(merged.items(), key=lambda row: row[1], reverse=True)
    ordered_indices = np.fromiter((item[0] for item in ordered), dtype=np.int64, count=len(ordered))
    ordered_scores = np.fromiter((item[1] for item in ordered), dtype=np.float32, count=len(ordered))
    return ordered_indices, ordered_scores


def find_similar_in_index_with_context(
    index: LoadedIndex,
    query_image: Path,
    top_k: int = 10,
    text_cache: TextSearchCache | None = None,
    text_rerank_pool: int = 120,
    text_weight: float = 0.42,
    enable_text_rerank: bool = True,
    enable_ai_embedding: bool = True,
    query_variants: int = 5,
) -> tuple[list[SearchResult], SearchContext]:
    variants = extract_query_feature_variants(
        image_path=query_image,
        max_variants=max(1, int(query_variants)),
    )
    q_vector, q_hash = variants[0]
    q_vector = np.asarray(q_vector, dtype=np.float32, copy=False)
    q_hash = np.asarray(q_hash, dtype=np.uint8, copy=False)

    if index.vectors.ndim != 2 or index.hashes.ndim != 2:
        raise ValueError("Invalid index format. Rebuild index and try again.")
    if index.vectors.shape[1] != q_vector.shape[0] or index.hashes.shape[1] != q_hash.shape[0]:
        raise ValueError("Index format changed. Click Rebuild Desktop Index once.")

    total = int(index.vectors.shape[0])
    vector_scores = np.full((total,), -1.0, dtype=np.float32)
    hash_scores = np.full((total,), 0.0, dtype=np.float32)
    embedding_scores = np.full((total,), 0.0, dtype=np.float32)
    final_score = np.full((total,), -2.0, dtype=np.float32)
    used_ai_embedding = False

    if enable_ai_embedding and index.embedding_vectors is not None and index.faiss_index is not None:
        query_embedding = _extract_query_embedding(query_image, expected_dim=index.embedding_vectors.shape[1])
        if query_embedding is not None:
            faiss_hits = _search_with_faiss(index.faiss_index, query_embedding, total_items=total, top_k=top_k)
            if faiss_hits is not None:
                candidate_indices, candidate_scores = faiss_hits
                min_required = max(1, min(int(top_k), total))
                if candidate_indices.size >= min_required:
                    local_vectors = index.vectors[candidate_indices]
                    local_hashes = index.hashes[candidate_indices]
                    local_vector_scores, local_hash_scores = _compute_local_scores(local_vectors, local_hashes, variants)

                    normalized_embedding = np.clip((candidate_scores + 1.0) * 0.5, 0.0, 1.0)
                    vector_scores[candidate_indices] = local_vector_scores
                    hash_scores[candidate_indices] = local_hash_scores
                    embedding_scores[candidate_indices] = normalized_embedding

                    visual_score = (
                        0.56 * local_vector_scores + 0.20 * local_hash_scores + 0.24 * normalized_embedding
                    )
                    structure_boost = (
                        np.clip(local_vector_scores - 0.42, 0.0, 1.0)
                        * np.clip(local_hash_scores - 0.58, 0.0, 1.0)
                        * 0.18
                    )
                    final_score[candidate_indices] = visual_score + structure_boost
                    used_ai_embedding = True

    if not used_ai_embedding:
        vector_scores, hash_scores = _compute_local_scores(index.vectors, index.hashes, variants)
        visual_score = 0.74 * vector_scores + 0.26 * hash_scores
        structure_boost = np.clip(vector_scores - 0.42, 0.0, 1.0) * np.clip(hash_scores - 0.58, 0.0, 1.0) * 0.24
        final_score = visual_score + structure_boost

    query_text = ""
    used_text_rerank = False
    text_score_lookup: dict[int, float] = {}
    text_excerpt_lookup: dict[int, str] = {}

    if enable_text_rerank and text_cache is not None and text_cache.available:
        query_text = text_cache.extract_query_text(query_image)
        if query_text:
            pool_size = max(top_k, min(int(text_rerank_pool), int(final_score.size)))
            pool_indices = _top_sorted_indices(final_score, pool_size)
            clipped_text_weight = max(0.0, min(float(text_weight), 0.72))

            for idx in pool_indices:
                item = int(idx)
                path = Path(index.paths[item])
                # Keep search latency stable: do not run OCR for uncached index items here.
                candidate_text = text_cache.get_index_text(path, build_on_miss=False)
                if not candidate_text:
                    continue
                match_score = text_similarity(query_text, candidate_text)
                if match_score <= 0.0:
                    continue

                text_score_lookup[item] = match_score
                text_excerpt_lookup[item] = candidate_text
                text_bonus = clipped_text_weight * (match_score ** 1.18)
                final_score[item] = final_score[item] + text_bonus

            text_cache.flush()
            used_text_rerank = bool(text_score_lookup)

    top_indices = _top_sorted_indices(final_score, top_k=top_k)

    results: list[SearchResult] = []
    for idx in top_indices:
        item = int(idx)
        results.append(
            SearchResult(
                path=index.paths[item],
                score=float(final_score[item]),
                vector_score=float(vector_scores[item]),
                hash_score=float(hash_scores[item]),
                embedding_score=float(embedding_scores[item]),
                text_score=float(text_score_lookup.get(item, 0.0)),
                text_excerpt=text_excerpt_lookup.get(item, ""),
            )
        )

    return results, SearchContext(
        query_text=query_text,
        used_text_rerank=used_text_rerank,
        used_ai_embedding=used_ai_embedding,
    )


def find_similar_in_index(
    index: LoadedIndex,
    query_image: Path,
    top_k: int = 10,
    text_cache: TextSearchCache | None = None,
    enable_text_rerank: bool = True,
    enable_ai_embedding: bool = True,
    query_variants: int = 5,
) -> list[SearchResult]:
    results, _context = find_similar_in_index_with_context(
        index=index,
        query_image=query_image,
        top_k=top_k,
        text_cache=text_cache,
        enable_text_rerank=enable_text_rerank,
        enable_ai_embedding=enable_ai_embedding,
        query_variants=query_variants,
    )
    return results


def find_similar(index_dir: Path, query_image: Path, top_k: int = 10) -> list[SearchResult]:
    runtime_index = load_runtime_index(index_dir=index_dir)
    return find_similar_in_index(index=runtime_index, query_image=query_image, top_k=top_k)
