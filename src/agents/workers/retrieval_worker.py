"""
Retrieval Worker — bọc Task 9 (hybrid search + PageIndex fallback).
"""

from __future__ import annotations

import time

from ..contracts import WorkerRequest, WorkerResponse
from ..state import PipelineState
from ...task9_retrieval_pipeline import retrieve


class RetrievalWorker:
    worker_id = "retrieval_worker"

    def handle(self, request: WorkerRequest, state: PipelineState) -> WorkerResponse:
        t0 = time.perf_counter()
        payload = request.payload
        query = payload["query"]
        top_k = payload.get("top_k", 5)
        use_reranking = payload.get("use_reranking", True)
        score_threshold = payload.get("score_threshold", 0.3)

        chunks = retrieve(
            query,
            top_k=top_k,
            use_reranking=use_reranking,
            score_threshold=score_threshold,
        )

        source = chunks[0].get("source", "hybrid") if chunks else "none"
        best_score = chunks[0]["score"] if chunks else 0.0
        duration_ms = (time.perf_counter() - t0) * 1000

        state.chunks = chunks
        state.retrieval_source = source
        state.append_trace(
            agent="retrieval_worker",
            action="retrieve",
            input_summary=f"query={query[:60]!r}, top_k={top_k}",
            output_summary=f"{len(chunks)} chunks, source={source}, best_score={best_score:.3f}",
            duration_ms=duration_ms,
            metadata={"use_reranking": use_reranking, "score_threshold": score_threshold},
        )

        return WorkerResponse(
            correlation_id=request.correlation_id,
            worker_id=self.worker_id,
            task="retrieve",
            success=True,
            payload={
                "chunks": chunks,
                "retrieval_source": source,
                "best_score": best_score,
                "needs_enrichment": best_score < score_threshold or source == "pageindex",
            },
        )
