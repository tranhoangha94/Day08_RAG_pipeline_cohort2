"""
Shared state schema cho multi-agent RAG pipeline.

Mọi agent đọc/ghi qua PipelineState; mỗi bước append vào `trace`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

AgentRole = Literal["supervisor", "retrieval_worker", "generation_worker", "mcp_worker"]
PipelineStatus = Literal["pending", "retrieving", "enriching", "generating", "done", "error"]


@dataclass
class TraceEntry:
    """Một bước trong luồng xử lý — quan sát được agent nào làm gì."""

    trace_id: str
    agent: AgentRole
    action: str
    timestamp: float
    duration_ms: float
    input_summary: str
    output_summary: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent": self.agent,
            "action": self.action,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "metadata": self.metadata,
        }


@dataclass
class PipelineState:
    """
    Shared state giữa supervisor và workers.

    Fields:
        query          — câu hỏi người dùng
        chunks         — kết quả retrieval (+ enrichment từ MCP)
        answer         — câu trả lời cuối
        sources        — chunks dùng cho generation
        retrieval_source — hybrid | pageindex | hybrid+mcp
        trace          — danh sách TraceEntry (audit trail)
        config         — top_k, threshold, ...
        status         — trạng thái pipeline
        error          — lỗi nếu có
        run_id         — correlation id cho toàn bộ run
    """

    query: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    chunks: list[dict] = field(default_factory=list)
    answer: str | None = None
    sources: list[dict] = field(default_factory=list)
    retrieval_source: str = "none"
    trace: list[TraceEntry] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    status: PipelineStatus = "pending"
    error: str | None = None
    chat_history: list[dict] = field(default_factory=list)

    def append_trace(
        self,
        agent: AgentRole,
        action: str,
        input_summary: str,
        output_summary: str,
        duration_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> TraceEntry:
        entry = TraceEntry(
            trace_id=f"{self.run_id}-{len(self.trace) + 1:02d}",
            agent=agent,
            action=action,
            timestamp=time.time(),
            duration_ms=duration_ms,
            input_summary=input_summary,
            output_summary=output_summary,
            metadata=metadata or {},
        )
        self.trace.append(entry)
        return entry

    def to_result(self) -> dict[str, Any]:
        return {
            "answer": self.answer or "",
            "sources": self.sources,
            "retrieval_source": self.retrieval_source,
            "trace": [t.to_dict() for t in self.trace],
            "run_id": self.run_id,
            "status": self.status,
            "error": self.error,
        }
