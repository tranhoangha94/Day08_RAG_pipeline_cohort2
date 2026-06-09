"""
Demo multi-agent RAG pipeline với reasoning flow quan sát được.

Chạy:
    python -m src.agents.demo
    python -m src.agents.demo --query "Hình phạt tàng trữ ma tuý?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.agents.supervisor import RAGSupervisor  # noqa: E402


DEMO_QUERIES = [
  "Hình phạt cho tội tàng trữ trái phép chất ma tuý?",
  "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
]


def print_trace(state) -> None:
    print("\n" + "=" * 72)
    print("REASONING FLOW (trace)")
    print("=" * 72)
    for i, entry in enumerate(state.trace, 1):
        meta = f" | {entry.metadata}" if entry.metadata else ""
        print(
            f"  [{i:02d}] {entry.trace_id} | {entry.agent:20s} | {entry.action:22s} "
            f"| {entry.duration_ms:7.1f}ms"
        )
        print(f"       IN : {entry.input_summary}")
        print(f"       OUT: {entry.output_summary}{meta}")
    print("=" * 72)


def print_message_contract(state) -> None:
    print("\n--- Message Contract (sample) ---")
    sample = {
        "supervisor_to_retrieval": {
            "type": "task",
            "worker_id": "retrieval_worker",
            "task": "retrieve",
            "payload": {"query": state.query[:50], "top_k": 5},
        },
        "retrieval_to_supervisor": {
            "type": "result",
            "worker_id": "retrieval_worker",
            "payload": {"chunks_count": len(state.chunks), "retrieval_source": state.retrieval_source},
        },
    }
    print(json.dumps(sample, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo Supervisor + Workers RAG")
    parser.add_argument("--query", "-q", help="Câu hỏi tùy chỉnh")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    queries = [args.query] if args.query else DEMO_QUERIES
    supervisor = RAGSupervisor()

    for query in queries:
        print("\n" + "#" * 72)
        print(f"QUERY: {query}")
        print("#" * 72)

        state = supervisor.run(
            query,
            top_k=args.top_k,
            use_reranking=not args.no_rerank,
            score_threshold=args.threshold,
        )

        print_trace(state)
        print_message_contract(state)

        print(f"\nRETRIEVAL SOURCE: {state.retrieval_source}")
        print(f"CHUNKS USED    : {len(state.sources)}")
        print(f"RUN ID         : {state.run_id}")
        print(f"STATUS         : {state.status}")

        if state.error:
            print(f"ERROR          : {state.error}")
        else:
            print("\n--- ANSWER ---")
            print(state.answer or "(empty)")
            print("--- END ANSWER ---")


if __name__ == "__main__":
    main()
