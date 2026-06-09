"""
Generation Worker — bọc Task 10 (reorder + format + LLM citation).
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv

from ..contracts import WorkerRequest, WorkerResponse
from ..state import PipelineState
from ...task10_generation import (
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    format_context,
    reorder_for_llm,
)

load_dotenv()


class GenerationWorker:
    worker_id = "generation_worker"

    def handle(self, request: WorkerRequest, state: PipelineState) -> WorkerResponse:
        t0 = time.perf_counter()
        payload = request.payload
        query = payload["query"]
        chunks = payload.get("chunks", state.chunks)
        chat_history = payload.get("chat_history", [])

        reordered = reorder_for_llm(chunks)
        context = format_context(reordered)
        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

        from openai import OpenAI

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in chat_history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content or ""

        duration_ms = (time.perf_counter() - t0) * 1000
        state.answer = answer
        state.sources = chunks
        state.append_trace(
            agent="generation_worker",
            action="generate",
            input_summary=f"query={query[:60]!r}, {len(chunks)} chunks",
            output_summary=f"answer_len={len(answer)}, model=gpt-4o-mini",
            duration_ms=duration_ms,
            metadata={"chunks_reordered": len(reordered)},
        )

        return WorkerResponse(
            correlation_id=request.correlation_id,
            worker_id=self.worker_id,
            task="generate",
            success=True,
            payload={"answer": answer, "sources": chunks},
        )
