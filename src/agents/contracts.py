"""
Message contract tối thiểu giữa Supervisor và Workers.

Supervisor gửi WorkerRequest → Worker trả WorkerResponse.
Chỉ truyền các field cần thiết từ PipelineState (không share toàn bộ object).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

WorkerId = Literal["retrieval_worker", "generation_worker", "mcp_worker"]
TaskName = Literal["retrieve", "generate", "enrich"]
MessageType = Literal["task", "result", "error"]


@dataclass(frozen=True)
class WorkerRequest:
    """Supervisor → Worker."""

    type: MessageType = "task"
    correlation_id: str = ""
    worker_id: WorkerId = "retrieval_worker"
    task: TaskName = "retrieve"
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "correlation_id": self.correlation_id,
            "worker_id": self.worker_id,
            "task": self.task,
            "payload": self.payload,
        }


@dataclass
class WorkerResponse:
    """Worker → Supervisor."""

    type: MessageType = "result"
    correlation_id: str = ""
    worker_id: WorkerId = "retrieval_worker"
    task: TaskName = "retrieve"
    success: bool = True
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "correlation_id": self.correlation_id,
            "worker_id": self.worker_id,
            "task": self.task,
            "success": self.success,
            "payload": self.payload,
            "error": self.error,
        }


# --- Payload schemas (tối thiểu) ---

# retrieve:  { query, top_k, use_reranking, score_threshold }
# enrich:    { query, existing_chunks, reason }
# generate:  { query, chunks, chat_history }

RETRIEVE_PAYLOAD_KEYS = ("query", "top_k", "use_reranking", "score_threshold")
ENRICH_PAYLOAD_KEYS = ("query", "existing_chunks", "reason")
GENERATE_PAYLOAD_KEYS = ("query", "chunks", "chat_history")
