"""
RAG Multi-Agent Demo — Streamlit UI
Supervisor → Retrieval Worker → MCP Worker (conditional) → Generation Worker

Chạy:
    streamlit run app.py
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.agents.supervisor import run_supervised_rag  # noqa: E402

st.set_page_config(
    page_title="Multi-Agent RAG Demo",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_QUESTIONS = [
    "Hình phạt cho tội tàng trữ trái phép chất ma tuý?",
    "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
    "Luật Phòng chống ma tuý 2021 quy định những hình thức cai nghiện nào?",
    "Khung hình phạt cho tội vận chuyển trái phép chất ma tuý?",
]

AGENT_META = {
    "supervisor": {"label": "Supervisor", "role": "Orchestrator", "color": "#6c63ff"},
    "retrieval_worker": {"label": "Retrieval Worker", "role": "Hybrid + PageIndex", "color": "#f59e0b"},
    "mcp_worker": {"label": "MCP Worker", "role": "News enrichment", "color": "#f97316"},
    "generation_worker": {"label": "Generation Worker", "role": "LLM + citation", "color": "#22c55e"},
}

CSS = """
<style>
    .stApp { background: #0f1117; }
    .block-container { padding-top: 1.5rem; max-width: 1280px; }
    div[data-testid="stSidebar"] { background: #1a1d2e; }

    .demo-header {
        text-align: center; padding: 28px 16px 24px;
        background: linear-gradient(135deg, #1a1d2e 0%, #0f1117 100%);
        border: 1px solid #2e3250; border-radius: 12px; margin-bottom: 20px;
    }
    .demo-header h1 {
        font-size: clamp(22px, 4vw, 34px); font-weight: 700; margin: 0 0 8px;
        background: linear-gradient(90deg, #6c63ff, #00d2ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .demo-header p { color: #8892b0; margin: 0; line-height: 1.6; font-size: 15px; }

    .card {
        background: #1a1d2e; border: 1px solid #2e3250;
        border-radius: 12px; padding: 18px 20px; margin-bottom: 16px;
    }
    .card-title {
        font-size: 13px; text-transform: uppercase; letter-spacing: .06em;
        color: #8892b0; font-weight: 700; margin-bottom: 14px;
    }

    .svc-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
    @media (max-width: 700px) { .svc-grid { grid-template-columns: repeat(2, 1fr); } }
    .svc-card {
        background: #22253a; border: 1px solid #2e3250; border-radius: 10px;
        padding: 12px; text-align: center; transition: border-color .3s;
    }
    .svc-card.active { border-color: #22c55e; }
    .svc-card.idle { border-color: #2e3250; opacity: .55; }
    .svc-card.skipped { border-color: #3a3f5c; opacity: .4; }
    .svc-dot { width: 10px; height: 10px; border-radius: 50%; margin: 0 auto 6px; background: #8892b0; }
    .svc-card.active .svc-dot { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
    .svc-card.idle .svc-dot { background: #8892b0; }
    .svc-card.skipped .svc-dot { background: #3a3f5c; }
    .svc-name { font-size: 12px; font-weight: 600; color: #e2e8f0; }
    .svc-role { font-size: 10px; color: #8892b0; margin-top: 2px; }

    .flow-wrap {
        background: #22253a; border: 1px solid #2e3250; border-radius: 12px;
        padding: 20px; overflow-x: auto;
    }
    .flow { display: flex; align-items: center; gap: 0; flex-wrap: nowrap; min-width: 620px; }
    .node { display: flex; flex-direction: column; align-items: center; gap: 4px; min-width: 88px; }
    .node-box {
        border: 2px solid #2e3250; border-radius: 10px; padding: 10px 12px;
        font-size: 11px; font-weight: 600; text-align: center; min-width: 76px; color: #e2e8f0;
    }
    .node-box.done { border-color: #22c55e; background: rgba(34,197,94,.1); }
    .node-box.active { border-color: #6c63ff; background: rgba(108,99,255,.15); box-shadow: 0 0 14px rgba(108,99,255,.25); }
    .node-box.skipped { border-color: #3a3f5c; opacity: .35; border-style: dashed; }
    .node-depth { font-size: 9px; color: #8892b0; }
    .arrow { color: #8892b0; font-size: 16px; padding: 0 3px; flex-shrink: 0; }
    .arrow.done { color: #00d2ff; }

    .progress-wrap { height: 6px; background: #22253a; border-radius: 3px; overflow: hidden; margin-bottom: 12px; }
    .progress-bar { height: 100%; background: linear-gradient(90deg, #6c63ff, #00d2ff); border-radius: 3px; }

    .tl-row {
        display: grid; grid-template-columns: 130px 1fr 90px 72px; gap: 8px;
        align-items: center; padding: 8px 12px; border-radius: 8px;
        background: #22253a; border: 1px solid #2e3250; font-size: 12px; margin-bottom: 6px;
    }
    .tl-agent { font-weight: 600; }
    .tl-action { color: #8892b0; }
    .tl-dur { text-align: right; color: #8892b0; font-family: monospace; font-size: 11px; }
    .badge {
        padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: 600; text-align: center;
    }
    .badge-done { background: rgba(34,197,94,.2); color: #22c55e; }
    .badge-run { background: rgba(108,99,255,.25); color: #a78bfa; }
    .badge-err { background: rgba(239,68,68,.2); color: #ef4444; }

    .result-box {
        background: #22253a; border: 1px solid #22c55e; border-radius: 10px;
        padding: 16px; line-height: 1.7; font-size: 14px; color: #e2e8f0;
    }
    .meta-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }
    .meta-tag {
        background: #1a1d2e; border: 1px solid #2e3250; border-radius: 6px;
        padding: 4px 10px; font-size: 12px; color: #8892b0;
    }
    .meta-tag span { color: #e2e8f0; font-family: monospace; }

    .msg-contract {
        background: #0a0c13; border: 1px solid #2e3250; border-radius: 8px;
        padding: 14px; font-size: 11px; font-family: 'Courier New', monospace;
        color: #8be9fd; overflow: auto; max-height: 280px; white-space: pre;
    }

    .sample-chip {
        display: inline-block; background: #22253a; border: 1px solid #2e3250;
        border-radius: 20px; padding: 5px 12px; font-size: 11px; color: #8892b0;
        margin: 0 6px 6px 0; cursor: default;
    }
</style>
"""


def init_session():
    defaults = {
        "messages": [],
        "chat_history": [],
        "last_result": None,
        "query_input": SAMPLE_QUESTIONS[0],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _select_sample(question: str) -> None:
    """Callback chạy trước khi widget render — được phép ghi session_state."""
    st.session_state.query_input = question


def _agents_from_trace(trace: list[dict]) -> set[str]:
    return {e["agent"] for e in trace}


def _used_mcp(trace: list[dict]) -> bool:
    return any(e["agent"] == "mcp_worker" for e in trace) or any(
        e.get("action") == "route_to_mcp" for e in trace
    )


def _total_ms(trace: list[dict]) -> float:
    for e in reversed(trace):
        if e.get("action") == "complete_pipeline":
            return e.get("duration_ms", 0)
    return sum(e.get("duration_ms", 0) for e in trace if e["agent"] != "supervisor")


def render_header():
    st.markdown(
        """
        <div class="demo-header">
            <h1>⚖️ Multi-Agent RAG Interaction Demo</h1>
            <p>
                Supervisor điều phối các Worker qua message contract —
                Retrieval → MCP enrichment (nếu cần) → Generation.
                Mọi bước được ghi vào trace để quan sát luồng agent.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _mcp_skip_reason(trace: list[dict] | None) -> str:
    if not trace:
        return ""
    for entry in trace:
        if entry.get("action") == "skip_mcp":
            return entry.get("output_summary", "skip MCP")
    return ""


def _agent_card_state(agent_id: str, trace: list[dict] | None) -> tuple[str, str, str]:
    """Return (border_color, dot_color, status_label)."""
    if not trace:
        return "#2e3250", "#8892b0", "Ready"

    invoked = _agents_from_trace(trace)
    mcp_used = _used_mcp(trace)

    if agent_id == "mcp_worker" and not mcp_used:
        reason = _mcp_skip_reason(trace) or "not needed"
        return "#3a3f5c", "#3a3f5c", f"Skipped · {reason}"
    if agent_id in invoked:
        return "#22c55e", "#22c55e", "Invoked"
    return "#2e3250", "#8892b0", "Idle"


def render_agent_status(trace: list[dict] | None):
    """Mỗi card render trong 1 column riêng — tránh Streamlit escape HTML."""
    with st.container(border=True):
        st.markdown(
            '<div class="card-title" style="margin-bottom:12px">Agent Status</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(4)
        for col, (agent_id, meta) in zip(cols, AGENT_META.items()):
            border, dot, status = _agent_card_state(agent_id, trace)
            with col:
                st.markdown(
                    f'<div style="background:#22253a;border:1px solid {border};border-radius:10px;'
                    f'padding:12px;text-align:center">'
                    f'<div style="width:10px;height:10px;border-radius:50%;background:{dot};'
                    f'margin:0 auto 6px"></div>'
                    f'<div style="font-size:12px;font-weight:600;color:#e2e8f0">{meta["label"]}</div>'
                    f'<div style="font-size:10px;color:#8892b0;margin-top:2px">{meta["role"]}</div>'
                    f'<div style="font-size:10px;color:#8892b0">{status}</div></div>',
                    unsafe_allow_html=True,
                )


def render_flow_diagram(trace: list[dict] | None):
    if not trace:
        nodes = {
            "user": "active",
            "supervisor": "",
            "retrieval": "",
            "mcp": "",
            "generation": "",
        }
        arrows = [""] * 5
    else:
        mcp = _used_mcp(trace)
        invoked = _agents_from_trace(trace)
        nodes = {
            "user": "done",
            "supervisor": "done" if "supervisor" in invoked else "",
            "retrieval": "done" if "retrieval_worker" in invoked else "",
            "mcp": "done" if mcp else "skipped",
            "generation": "done" if "generation_worker" in invoked else "",
        }
        arrows = ["done"] * 5

    def cls(name: str) -> str:
        state = nodes[name]
        return f"node-box {state}".strip()

    def arr(i: int) -> str:
        return f"arrow {arrows[i]}".strip()

    flow_html = (
        f'<div class="card-title">Agent Topology</div><div class="flow-wrap"><div class="flow">'
        f'<div class="node"><div class="{cls("user")}">👤 User</div><div class="node-depth">Browser</div></div>'
        f'<div class="{arr(0)}">→</div>'
        f'<div class="node"><div class="{cls("supervisor")}">Supervisor</div><div class="node-depth">orchestrator</div></div>'
        f'<div class="{arr(1)}">→</div>'
        f'<div class="node"><div class="{cls("retrieval")}">Retrieval<br>Worker</div><div class="node-depth">hybrid / pageindex</div></div>'
        f'<div class="{arr(2)}">→</div>'
        f'<div class="node"><div class="{cls("mcp")}">MCP<br>Worker</div><div class="node-depth">conditional</div></div>'
        f'<div class="{arr(3)}">→</div>'
        f'<div class="node"><div class="{cls("generation")}">Generation<br>Worker</div><div class="node-depth">gpt-4o-mini</div></div>'
        f'<div class="{arr(4)}">→</div>'
        f'<div class="node"><div class="node-box done">Answer</div><div class="node-depth">citation</div></div>'
        f"</div></div>"
    )
    with st.container(border=True):
        st.markdown(flow_html, unsafe_allow_html=True)


def render_timeline(trace: list[dict]):
    if not trace:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title">Execution Timeline</div>'
                '<p style="color:#8892b0;font-size:13px;text-align:center;padding:16px">'
                "Chạy demo để xem timeline tương tác agent</p>",
                unsafe_allow_html=True,
            )
        return

    total = _total_ms(trace)
    with st.container(border=True):
        st.markdown(
            f'<div class="card-title">Execution Timeline · '
            f'<span style="color:#22c55e">⏱ {total/1000:.2f}s total</span></div>'
            f'<div class="progress-wrap"><div class="progress-bar" style="width:100%"></div>',
            unsafe_allow_html=True,
        )
        for entry in trace:
            meta = AGENT_META.get(entry["agent"], {"label": entry["agent"], "color": "#8892b0"})
            badge_cls = "badge-err" if entry.get("action") == "error" else "badge-done"
            inp = html.escape(entry["input_summary"][:70])
            st.markdown(
                f'<div class="tl-row"><div class="tl-agent" style="color:{meta["color"]}">'
                f'{meta["label"]}</div><div class="tl-action">{entry["action"]} — IN: {inp}</div>'
                f'<div class="badge {badge_cls}">Done</div>'
                f'<div class="tl-dur">{entry["duration_ms"]:.0f}ms</div></div>',
                unsafe_allow_html=True,
            )


def build_message_contract(query: str, result: dict) -> dict:
    trace = result.get("trace", [])
    mcp_used = _used_mcp(trace)
    contract = {
        "user_to_supervisor": {
            "type": "query",
            "payload": {"query": query[:80], "top_k": result.get("config", {}).get("top_k", 5)},
        },
        "supervisor_to_retrieval": {
            "type": "task",
            "worker_id": "retrieval_worker",
            "task": "retrieve",
            "payload": {"query": query[:60], "top_k": 5},
        },
        "retrieval_to_supervisor": {
            "type": "result",
            "worker_id": "retrieval_worker",
            "payload": {
                "chunks_count": len(result.get("sources", [])),
                "retrieval_source": result.get("retrieval_source", "hybrid"),
            },
        },
    }
    if mcp_used:
        contract["supervisor_to_mcp"] = {
            "type": "task",
            "worker_id": "mcp_worker",
            "task": "enrich",
            "payload": {"query": query[:60], "reason": "news_query_or_low_score"},
        }
        contract["mcp_to_supervisor"] = {
            "type": "result",
            "worker_id": "mcp_worker",
            "payload": {"chunks_after_enrich": len(result.get("sources", []))},
        }
    else:
        contract["supervisor_decision"] = {
            "action": "skip_mcp",
            "reason": "hybrid retrieval sufficient",
        }
    contract["supervisor_to_generation"] = {
        "type": "task",
        "worker_id": "generation_worker",
        "task": "generate",
        "payload": {"query": query[:60], "chunks": len(result.get("sources", []))},
    }
    contract["generation_to_supervisor"] = {
        "type": "result",
        "worker_id": "generation_worker",
        "payload": {"answer_len": len(result.get("answer", "")), "model": "gpt-4o-mini"},
    }
    return contract


def render_message_contract(query: str, result: dict | None):
    if not result:
        return
    contract = build_message_contract(query, result)
    contract_text = html.escape(json.dumps(contract, ensure_ascii=False, indent=2))
    st.markdown(
        f'<div class="card"><div class="card-title">Message Contract (Supervisor ↔ Workers)</div>'
        f'<div class="msg-contract">{contract_text}</div></div>',
        unsafe_allow_html=True,
    )


def render_answer(result: dict):
    answer = result.get("answer", "")
    run_id = result.get("run_id", "—")
    source = result.get("retrieval_source", "—")
    status = result.get("status", "—")
    total = _total_ms(result.get("trace", []))
    asked = html.escape(result.get("query", ""))

    st.markdown(
        f'<div class="meta-row">'
        f'<div class="meta-tag">question: <span>{asked[:80]}</span></div>'
        f'<div class="meta-tag">run_id: <span>{run_id}</span></div>'
        f'<div class="meta-tag">status: <span>{status}</span></div>'
        f'<div class="meta-tag">retrieval: <span>{source}</span></div>'
        f'<div class="meta-tag">latency: <span>{total/1000:.2f}s</span></div></div>'
        f'<div class="result-box">{html.escape(answer)}</div>',
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict], retrieval_source: str):
    if not sources:
        return
    st.markdown('<div class="card-title">Retrieved Sources</div>', unsafe_allow_html=True)
    st.caption(f"{retrieval_source} · {len(sources)} chunks")
    for i, src in enumerate(sources, 1):
        meta = src.get("metadata", {})
        title = meta.get("source", meta.get("filename", f"Source {i}"))
        score = src.get("score", 0)
        with st.expander(f"📄 {i}. {title} — score {score:.3f}"):
            st.markdown(src.get("content", ""))


def run_pipeline(query: str, top_k: int, use_reranking: bool, score_threshold: float) -> dict:
    return run_supervised_rag(
        query,
        top_k=top_k,
        use_reranking=use_reranking,
        score_threshold=score_threshold,
        chat_history=st.session_state.chat_history,
    )


def main():
    init_session()
    st.markdown(CSS, unsafe_allow_html=True)
    render_header()

    with st.sidebar:
        st.header("⚙️ Cài đặt")
        top_k = st.slider("Top K chunks", 3, 10, 5)
        use_reranking = st.checkbox("Bật reranking", value=True)
        score_threshold = st.slider("Score threshold (fallback PageIndex)", 0.0, 0.9, 0.3, 0.05)
        st.divider()
        st.markdown("**Pipeline**")
        st.code(
            "Supervisor\n  → Retrieval Worker\n  → MCP Worker (conditional)\n  → Generation Worker",
            language=None,
        )
        st.divider()
        if st.button("🗑️ Xóa lịch sử", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.last_result = None
            st.rerun()

    last = st.session_state.last_result
    trace = last.get("trace") if last else None

    # 1. User Prompt + Final Answer
    with st.container(border=True):
        st.markdown('<div class="card-title">Legal Question</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="card-title" style="margin-top:4px">Sample questions</div>',
            unsafe_allow_html=True,
        )
        scols = st.columns(2)
        for i, sq in enumerate(SAMPLE_QUESTIONS):
            label = sq[:42] + "…" if len(sq) > 42 else sq
            with scols[i % 2]:
                st.button(
                    label,
                    key=f"sample_{i}",
                    use_container_width=True,
                    on_click=_select_sample,
                    args=(sq,),
                )

        query = st.text_area(
            "Câu hỏi",
            key="query_input",
            height=100,
            label_visibility="collapsed",
            placeholder="Nhập câu hỏi pháp luật ma tuý...",
        )

        run_clicked = st.button("▶ Run Multi-Agent Demo", type="primary", use_container_width=True)

        if run_clicked and query.strip():
            with st.status("🔄 Agents đang tương tác...", expanded=True) as status:
                st.write("**Supervisor** → khởi tạo pipeline, gửi task tới Retrieval Worker")
                try:
                    result = run_pipeline(query.strip(), top_k, use_reranking, score_threshold)
                    result["query"] = query.strip()
                    st.write(
                        f"**Retrieval Worker** → {result.get('retrieval_source', '?')} · "
                        f"{len(result.get('sources', []))} chunks"
                    )
                    if _used_mcp(result.get("trace", [])):
                        st.write("**Supervisor** → route_to_mcp → **MCP Worker** enrichment")
                    else:
                        st.write("**Supervisor** → skip_mcp (retrieval đủ tốt)")
                    st.write("**Generation Worker** → sinh câu trả lời có citation")
                    status.update(label="✅ Pipeline hoàn tất", state="complete")
                except Exception as exc:
                    result = {
                        "answer": f"Lỗi: {exc}",
                        "sources": [],
                        "retrieval_source": "error",
                        "trace": [],
                        "run_id": "error",
                        "status": "error",
                        "query": query.strip(),
                    }
                    status.update(label=f"❌ Lỗi: {exc}", state="error")

            st.session_state.last_result = result
            st.session_state.messages.append({"role": "user", "content": query.strip()})
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "retrieval_source": result.get("retrieval_source", ""),
                    "trace": result.get("trace", []),
                }
            )
            st.session_state.chat_history.append({"role": "user", "content": query.strip()})
            st.session_state.chat_history.append({"role": "assistant", "content": result.get("answer", "")})
            st.rerun()

        if last:
            st.markdown(
                '<div class="card-title" style="margin-top:16px">Final Answer</div>',
                unsafe_allow_html=True,
            )
            render_answer(last)

    # refresh trace after potential run
    last = st.session_state.last_result
    trace = last.get("trace") if last else None

    # 2. Agent Status
    render_agent_status(trace)

    # 3. Agent Topology
    render_flow_diagram(trace)

    # 4. Execution Timeline
    render_timeline(trace or [])

    if last:
        render_message_contract(last.get("query", ""), last)
        render_sources(last.get("sources", []), last.get("retrieval_source", ""))

        with st.expander("🔍 Raw trace JSON"):
            st.json(last.get("trace", []))


if __name__ == "__main__":
    main()
