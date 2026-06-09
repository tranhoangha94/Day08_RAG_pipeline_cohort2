"""
MCP Enrichment Worker — external capability qua MCP tool `search_news_context`.

Kích hoạt khi supervisor phát hiện câu hỏi tin tức hoặc hybrid score thấp.
"""

from __future__ import annotations

import re
import time

from ..contracts import WorkerRequest, WorkerResponse
from ..mcp_client import call_tool
from ..state import PipelineState

NEWS_KEYWORDS = re.compile(
    r"nghệ sĩ|ca sĩ|diễn viên|tin tức|bắt|bị bắt|scandal|showbiz|2024|2025|2026",
    re.IGNORECASE,
)


class MCPEnrichmentWorker:
    worker_id = "mcp_worker"

    def should_enrich(self, query: str, best_score: float, threshold: float) -> tuple[bool, str]:
        if NEWS_KEYWORDS.search(query):
            return True, "news_query_detected"
        if best_score < threshold:
            return True, "low_hybrid_score"
        return False, ""

    def handle(self, request: WorkerRequest, state: PipelineState) -> WorkerResponse:
        t0 = time.perf_counter()
        payload = request.payload
        query = payload["query"]
        existing = payload.get("existing_chunks", [])
        reason = payload.get("reason", "supervisor_dispatch")

        mcp_result = call_tool(
            "search_news_context",
            {"query": query, "top_k": 3},
        )
        new_chunks = mcp_result.get("chunks", [])
        via = mcp_result.get("via", "unknown")

        merged = list(existing)
        seen_content = {c.get("content", "")[:200] for c in existing}
        for chunk in new_chunks:
            key = chunk.get("content", "")[:200]
            if key not in seen_content:
                merged.append(chunk)
                seen_content.add(key)

        state.chunks = merged
        if new_chunks:
            base = state.retrieval_source
            state.retrieval_source = f"{base}+mcp" if base != "none" else "mcp"

        duration_ms = (time.perf_counter() - t0) * 1000
        state.append_trace(
            agent="mcp_worker",
            action="enrich_via_mcp",
            input_summary=f"query={query[:60]!r}, reason={reason}",
            output_summary=f"+{len(new_chunks)} MCP chunks via={via}, total={len(merged)}",
            duration_ms=duration_ms,
            metadata={"mcp_tool": "search_news_context", "mcp_via": via, "reason": reason},
        )

        return WorkerResponse(
            correlation_id=request.correlation_id,
            worker_id=self.worker_id,
            task="enrich",
            success=mcp_result.get("success", False) or bool(existing),
            payload={"chunks": merged, "mcp_chunks_added": len(new_chunks), "mcp_via": via},
        )
