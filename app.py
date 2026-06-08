"""
RAG Chatbot — Bài tập nhóm
Streamlit UI → Task 9 (retrieve) → Task 10 (generate_with_citation)

Chạy:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.task10_generation import generate_with_citation

st.set_page_config(
    page_title="RAG Pháp Luật Ma Tuý",
    page_icon="⚖️",
    layout="wide",
)

SAMPLE_QUESTIONS = [
    "Hình phạt cho tội tàng trữ trái phép chất ma tuý?",
    "Luật Phòng chống ma tuý 2021 quy định những hình thức cai nghiện nào?",
    "Điều kiện áp dụng cai nghiện bắt buộc là gì?",
    "Khung hình phạt cho tội vận chuyển trái phép chất ma tuý?",
]


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def render_sources(sources: list[dict], retrieval_source: str):
    st.caption(f"Retrieval: **{retrieval_source}** · {len(sources)} nguồn")
    for i, src in enumerate(sources, 1):
        meta = src.get("metadata", {})
        title = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        score = src.get("score", 0)
        with st.expander(f"📄 {i}. {title} ({doc_type}) — score {score:.3f}"):
            st.markdown(src.get("content", ""))


def main():
    init_session()

    st.title("⚖️ RAG Chatbot — Pháp Luật Ma Tuý")
    st.markdown(
        "Chatbot trả lời câu hỏi về **pháp luật ma tuý** và **tin tức liên quan**, "
        "có citation và hiển thị nguồn tham khảo."
    )

    with st.sidebar:
        st.header("Cài đặt")
        top_k = st.slider("Top K chunks", 3, 10, 5)
        use_reranking = st.checkbox("Bật reranking", value=True)
        score_threshold = st.slider("Score threshold (fallback PageIndex)", 0.0, 0.9, 0.3, 0.05)

        st.divider()
        st.markdown("**Câu hỏi mẫu**")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, key=f"sample_{q[:20]}"):
                st.session_state.pending_query = q

        st.divider()
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.markdown(
            "**Pipeline:**\n"
            "Streamlit → Task 9 (retrieve) → Task 10 (generation)"
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Nguồn tham khảo"):
                    render_sources(msg["sources"], msg.get("retrieval_source", "hybrid"))

    query = st.chat_input("Nhập câu hỏi của bạn...")
    if hasattr(st.session_state, "pending_query"):
        query = st.session_state.pending_query
        del st.session_state.pending_query

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và tạo câu trả lời..."):
                try:
                    result = generate_with_citation(
                        query,
                        top_k=top_k,
                        use_reranking=use_reranking,
                        score_threshold=score_threshold,
                        chat_history=st.session_state.chat_history,
                    )
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    retrieval_source = result.get("retrieval_source", "hybrid")
                except Exception as exc:
                    answer = f"Lỗi khi xử lý: {exc}"
                    sources = []
                    retrieval_source = "error"

            st.markdown(answer)
            if sources:
                with st.expander("📚 Nguồn tham khảo"):
                    render_sources(sources, retrieval_source)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "retrieval_source": retrieval_source,
            }
        )
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
