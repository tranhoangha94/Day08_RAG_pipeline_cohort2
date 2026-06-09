"""
Dry-run demo — kiểm tra supervisor routing + trace KHÔNG cần index hay OpenAI.

Chạy:
    python -m src.agents.demo_dry_run
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.contracts import WorkerRequest  # noqa: E402
from src.agents.supervisor import RAGSupervisor  # noqa: E402
from src.agents.workers.generation_worker import GenerationWorker  # noqa: E402


MOCK_CHUNKS_LEGAL = [
    {
        "content": "Điều 249 BLTTHS: Tàng trữ trái phép chất ma tuý bị phạt tù từ 1-5 năm.",
        "score": 0.72,
        "metadata": {"source": "BLTTHS_2015.md", "type": "legal"},
        "source": "hybrid",
    }
]

MOCK_CHUNKS_NEWS = [
    {
        "content": "Ca sĩ X bị bắt vì tàng trữ ma tuý tại TP.HCM năm 2024.",
        "score": 0.55,
        "metadata": {"source": "news_2024.md", "type": "news", "via": "mcp_local_fallback"},
        "source": "mcp",
    }
]


def _mock_retrieve(query, **kwargs):
    if "nghệ sĩ" in query.lower() or "ca sĩ" in query.lower():
        return [{"content": "Tin showbiz...", "score": 0.25, "metadata": {}, "source": "pageindex"}]
    return MOCK_CHUNKS_LEGAL


def _mock_mcp(tool_name, arguments):
    return {"success": True, "chunks": MOCK_CHUNKS_NEWS, "via": "mcp_local_fallback"}


def _mock_generate(self, request: WorkerRequest, state):
    t0 = time.perf_counter()
    answer = "Theo [BLTTHS 2015, Điều 249], tàng trữ ma tuý bị phạt tù 1-5 năm."
    state.answer = answer
    state.sources = state.chunks
    state.append_trace(
        agent="generation_worker",
        action="generate",
        input_summary=f"query={request.payload['query'][:40]!r}",
        output_summary="answer_len=65 (mock)",
        duration_ms=(time.perf_counter() - t0) * 1000,
    )
    from src.agents.contracts import WorkerResponse

    return WorkerResponse(
        correlation_id=request.correlation_id,
        worker_id="generation_worker",
        task="generate",
        success=True,
        payload={"answer": answer, "sources": state.chunks},
    )


def run_case(query: str) -> None:
    print(f"\n{'='*60}\nQUERY: {query}\n{'='*60}")
    with (
        patch("src.agents.workers.retrieval_worker.retrieve", side_effect=_mock_retrieve),
        patch("src.agents.mcp_client.call_tool", side_effect=_mock_mcp),
        patch.object(GenerationWorker, "handle", _mock_generate),
    ):
        state = RAGSupervisor().run(query, top_k=3, score_threshold=0.3)

    for entry in state.trace:
        print(f"  [{entry.trace_id}] {entry.agent:20s} | {entry.action:22s} | {entry.duration_ms:6.1f}ms")
        print(f"       {entry.input_summary} → {entry.output_summary}")

    print(f"\nSOURCE: {state.retrieval_source}")
    print(f"ANSWER: {state.answer}")


if __name__ == "__main__":
    run_case("Hình phạt tàng trữ ma tuý?")
    run_case("Những nghệ sĩ nào bị bắt vì ma tuý?")
