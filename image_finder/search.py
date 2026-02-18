from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import extract_query_feature_variants
from .indexer import load_index
from .text_search import TextSearchCache, text_similarity


@dataclass
class SearchResult:
    path: str
    score: float
    vector_score: float
    hash_score: float
    text_score: float = 0.0
    text_excerpt: str = ""


@dataclass
class LoadedIndex:
    vectors: np.ndarray
    hashes: np.ndarray
    paths: list[str]


@dataclass
class SearchContext:
    query_text: str
    used_text_rerank: bool


def load_runtime_index(index_dir: Path) -> LoadedIndex:
    vectors, hashes, paths = load_index(index_dir=index_dir, use_mmap=False)
    vectors_array = np.ascontiguousarray(vectors, dtype=np.float32)
    hashes_array = np.ascontiguousarray(hashes, dtype=np.uint8)
    return LoadedIndex(vectors=vectors_array, hashes=hashes_array, paths=paths)


def _top_sorted_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if scores.ndim != 1:
        raise ValueError("Score vector must be one-dimensional.")
    if scores.size == 0:
        return np.array([], dtype=np.int64)

    limited_k = max(1, min(int(top_k), int(scores.size)))
    partitioned = np.argpartition(scores, -limited_k)[-limited_k:]
    return partitioned[np.argsort(scores[partitioned])[::-1]]


def find_similar_in_index_with_context(
    index: LoadedIndex,
    query_image: Path,
    top_k: int = 10,
    text_cache: TextSearchCache | None = None,
    text_rerank_pool: int = 120,
    text_weight: float = 0.42,
) -> tuple[list[SearchResult], SearchContext]:
    variants = extract_query_feature_variants(image_path=query_image, max_variants=3)
    q_vector, q_hash = variants[0]
    q_vector = np.asarray(q_vector, dtype=np.float32, copy=False)
    q_hash = np.asarray(q_hash, dtype=np.uint8, copy=False)

    if index.vectors.ndim != 2 or index.hashes.ndim != 2:
        raise ValueError("Invalid index format. Rebuild index and try again.")
    if index.vectors.shape[1] != q_vector.shape[0] or index.hashes.shape[1] != q_hash.shape[0]:
        raise ValueError("Index format changed. Click Rebuild Desktop Index once.")

    vector_scores = np.full((index.vectors.shape[0],), -1.0, dtype=np.float32)
    hash_scores = np.full((index.hashes.shape[0],), 0.0, dtype=np.float32)
    matched_variant = False
    for variant_vector, variant_hash in variants:
        local_vector = np.asarray(variant_vector, dtype=np.float32, copy=False)
        local_hash = np.asarray(variant_hash, dtype=np.uint8, copy=False)
        if local_vector.shape[0] != index.vectors.shape[1] or local_hash.shape[0] != index.hashes.shape[1]:
            continue
        matched_variant = True
        current_vector_scores = index.vectors @ local_vector
        hash_mismatch = np.count_nonzero(index.hashes != local_hash, axis=1)
        current_hash_scores = 1.0 - (hash_mismatch.astype(np.float32) / float(index.hashes.shape[1]))
        vector_scores = np.maximum(vector_scores, current_vector_scores)
        hash_scores = np.maximum(hash_scores, current_hash_scores)
    if not matched_variant:
        raise ValueError("Index format changed. Click Rebuild Desktop Index once.")

    visual_score = 0.74 * vector_scores + 0.26 * hash_scores
    structure_boost = np.clip(vector_scores - 0.42, 0.0, 1.0) * np.clip(hash_scores - 0.58, 0.0, 1.0) * 0.24
    final_score = visual_score + structure_boost

    query_text = ""
    used_text_rerank = False
    text_score_lookup: dict[int, float] = {}
    text_excerpt_lookup: dict[int, str] = {}

    if text_cache is not None and text_cache.available:
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
                text_score=float(text_score_lookup.get(item, 0.0)),
                text_excerpt=text_excerpt_lookup.get(item, ""),
            )
        )

    return results, SearchContext(query_text=query_text, used_text_rerank=used_text_rerank)


def find_similar_in_index(
    index: LoadedIndex,
    query_image: Path,
    top_k: int = 10,
    text_cache: TextSearchCache | None = None,
) -> list[SearchResult]:
    results, _context = find_similar_in_index_with_context(
        index=index,
        query_image=query_image,
        top_k=top_k,
        text_cache=text_cache,
    )
    return results


def find_similar(index_dir: Path, query_image: Path, top_k: int = 10) -> list[SearchResult]:
    runtime_index = load_runtime_index(index_dir=index_dir)
    return find_similar_in_index(index=runtime_index, query_image=query_image, top_k=top_k)
