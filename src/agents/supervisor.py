"""
RAG Supervisor — điều phối retrieval → MCP enrichment (nếu cần) → generation.

Luồng:
    1. Khởi tạo PipelineState + trace
    2. Gửi WorkerRequest → RetrievalWorker
    3. Nếu news query hoặc score thấp → MCPEnrichmentWorker
    4. Gửi WorkerRequest → GenerationWorker
    5. Trả PipelineState hoàn chỉnh kèm trace
"""

from __future__ import annotations

import time

from .contracts import WorkerRequest
from .state import PipelineState
from .workers import GenerationWorker, MCPEnrichmentWorker, RetrievalWorker


class RAGSupervisor:
    def __init__(self) -> None:
        self.retrieval_worker = RetrievalWorker()
        self.mcp_worker = MCPEnrichmentWorker()
        self.generation_worker = GenerationWorker()

    def run(
        self,
        query: str,
        *,
        top_k: int = 5,
        use_reranking: bool = True,
        score_threshold: float = 0.3,
        chat_history: list[dict] | None = None,
    ) -> PipelineState:
        t0 = time.perf_counter()
        state = PipelineState(
            query=query,
            config={
                "top_k": top_k,
                "use_reranking": use_reranking,
                "score_threshold": score_threshold,
            },
            chat_history=chat_history or [],
            status="pending",
        )

        state.append_trace(
            agent="supervisor",
            action="start_pipeline",
            input_summary=f"query={query[:80]!r}",
            output_summary=f"run_id={state.run_id}",
            duration_ms=0,
            metadata=state.config,
        )

        # --- Step 1: Retrieval ---
        state.status = "retrieving"
        retrieve_req = WorkerRequest(
            correlation_id=state.run_id,
            worker_id="retrieval_worker",
            task="retrieve",
            payload={
                "query": query,
                "top_k": top_k,
                "use_reranking": use_reranking,
                "score_threshold": score_threshold,
            },
        )
        retrieve_resp = self.retrieval_worker.handle(retrieve_req, state)
        if not retrieve_resp.success:
            state.status = "error"
            state.error = retrieve_resp.error or "retrieval failed"
            return state

        best_score = retrieve_resp.payload.get("best_score", 0.0)

        # --- Step 2: MCP enrichment (conditional) ---
        should_enrich, reason = self.mcp_worker.should_enrich(query, best_score, score_threshold)
        if should_enrich:
            state.status = "enriching"
            enrich_req = WorkerRequest(
                correlation_id=state.run_id,
                worker_id="mcp_worker",
                task="enrich",
                payload={
                    "query": query,
                    "existing_chunks": state.chunks,
                    "reason": reason,
                },
            )
            self.mcp_worker.handle(enrich_req, state)
            state.append_trace(
                agent="supervisor",
                action="route_to_mcp",
                input_summary=f"reason={reason}, best_score={best_score:.3f}",
                output_summary=f"chunks_after_enrich={len(state.chunks)}",
                duration_ms=0,
            )
        else:
            state.append_trace(
                agent="supervisor",
                action="skip_mcp",
                input_summary=f"best_score={best_score:.3f}, threshold={score_threshold}",
                output_summary="hybrid sufficient, skip MCP",
                duration_ms=0,
            )

        # --- Step 3: Generation ---
        state.status = "generating"
        gen_req = WorkerRequest(
            correlation_id=state.run_id,
            worker_id="generation_worker",
            task="generate",
            payload={
                "query": query,
                "chunks": state.chunks,
                "chat_history": state.chat_history,
            },
        )
        gen_resp = self.generation_worker.handle(gen_req, state)
        if not gen_resp.success:
            state.status = "error"
            state.error = gen_resp.error or "generation failed"
            return state

        total_ms = (time.perf_counter() - t0) * 1000
        state.status = "done"
        state.append_trace(
            agent="supervisor",
            action="complete_pipeline",
            input_summary=f"run_id={state.run_id}",
            output_summary=f"status=done, total_ms={total_ms:.0f}",
            duration_ms=total_ms,
            metadata={"agents_invoked": self._agents_invoked(state)},
        )
        return state

    @staticmethod
    def _agents_invoked(state: PipelineState) -> list[str]:
        return list(dict.fromkeys(t.agent for t in state.trace))


def run_supervised_rag(
    query: str,
    top_k: int = 5,
    use_reranking: bool = True,
    score_threshold: float = 0.3,
    chat_history: list[dict] | None = None,
) -> dict:
    """API tương thích với generate_with_citation + thêm trace."""
    supervisor = RAGSupervisor()
    state = supervisor.run(
        query,
        top_k=top_k,
        use_reranking=use_reranking,
        score_threshold=score_threshold,
        chat_history=chat_history,
    )
    result = state.to_result()
    result["answer"] = state.answer or ""
    result["sources"] = state.sources
    return result
