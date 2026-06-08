"""Shared local vector + BM25 index (no external DB required)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.task4_chunking_indexing import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
)

PROJECT_DIR = Path(__file__).parent.parent
INDEX_PATH = PROJECT_DIR / "data" / "index" / "vector_index.pkl"

_model: SentenceTransformer | None = None
_chunks: list[dict] | None = None
_embeddings: np.ndarray | None = None
_bm25: BM25Okapi | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed theo batch để tránh treo và hiện tiến trình."""
    model = get_model()
    total = len(texts)
    if total == 0:
        return np.array([], dtype=np.float32)

    batches = []
    num_batches = (total + batch_size - 1) // batch_size
    print(f"  Embedding {total} chunks in {num_batches} batches (batch_size={batch_size})...")

    for i in range(0, total, batch_size):
        batch_num = i // batch_size + 1
        end = min(i + batch_size, total)
        print(f"    [{batch_num}/{num_batches}] chunks {i + 1}-{end}/{total}", flush=True)
        vectors = model.encode(
            texts[i:end],
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=batch_size,
        )
        batches.append(np.asarray(vectors, dtype=np.float32))

    return np.vstack(batches)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]


def save_index(chunks: list[dict], embeddings: np.ndarray) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunks": chunks,
        "embeddings": embeddings,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }
    INDEX_PATH.write_bytes(pickle.dumps(payload))
    _reset_cache()


def _reset_cache() -> None:
    global _chunks, _embeddings, _bm25
    _chunks = None
    _embeddings = None
    _bm25 = None


def load_index() -> tuple[list[dict], np.ndarray]:
    global _chunks, _embeddings, _bm25
    if _chunks is not None and _embeddings is not None:
        return _chunks, _embeddings

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index not found at {INDEX_PATH}. Run task4_chunking_indexing.py first."
        )

    payload = pickle.loads(INDEX_PATH.read_bytes())
    _chunks = payload["chunks"]
    _embeddings = payload["embeddings"]
    return _chunks, _embeddings


def get_bm25() -> tuple[BM25Okapi, list[dict]]:
    global _bm25
    chunks, _ = load_index()
    if _bm25 is None:
        tokenized = [c["content"].lower().split() for c in chunks]
        _bm25 = BM25Okapi(tokenized)
    return _bm25, chunks


def cosine_search(query: str, top_k: int) -> list[dict]:
    chunks, embeddings = load_index()
    query_vec = embed_query(query)
    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append(
            {
                "content": chunks[idx]["content"],
                "score": float(scores[idx]),
                "metadata": chunks[idx]["metadata"],
            }
        )
    return results
