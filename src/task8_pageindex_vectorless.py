"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

_sections_cache: list[dict] | None = None


def _load_sections() -> list[dict]:
    """Fallback local: chia markdown theo heading (structural retrieval)."""
    global _sections_cache
    if _sections_cache is not None:
        return _sections_cache

    sections = []
    if not STANDARDIZED_DIR.exists():
        _sections_cache = sections
        return sections

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        parts = re.split(r"(?m)^#{1,3}\s+", content)
        headers = re.findall(r"(?m)^#{1,3}\s+(.+)$", content)

        if not headers:
            sections.append(
                {
                    "content": content[:2000],
                    "metadata": {"source": md_file.name, "type": md_file.parent.name},
                }
            )
            continue

        for i, body in enumerate(parts[1:], 0):
            title = headers[i] if i < len(headers) else md_file.stem
            section_text = f"## {title}\n{body.strip()}"
            if len(section_text) < 50:
                continue
            sections.append(
                {
                    "content": section_text[:3000],
                    "metadata": {
                        "source": md_file.name,
                        "type": md_file.parent.name,
                        "section": title,
                    },
                }
            )

    _sections_cache = sections
    return sections


def _local_structural_search(query: str, top_k: int) -> list[dict]:
    sections = _load_sections()
    query_tokens = set(re.findall(r"\w+", query.lower()))

    scored = []
    for sec in sections:
        content_tokens = set(re.findall(r"\w+", sec["content"].lower()))
        overlap = len(query_tokens & content_tokens)
        if overlap == 0:
            continue
        score = overlap / max(len(query_tokens), 1)
        scored.append((score, sec))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "content": s["content"],
            "score": score,
            "metadata": s["metadata"],
            "source": "pageindex",
        }
        for score, s in scored[:top_k]
    ]


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    # TODO: Implement upload
    #
    # Tham khảo: https://github.com/VectifyAI/PageIndex
    #
    # from pageindex import PageIndex
    #
    # pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    #
    # for md_file in STANDARDIZED_DIR.rglob("*.md"):
    #     content = md_file.read_text(encoding="utf-8")
    #     pi.upload(
    #         content=content,
    #         metadata={"filename": md_file.name, "type": md_file.parent.name}
    #     )
    #     print(f"  ✓ Uploaded: {md_file.name}")

    if PAGEINDEX_API_KEY and not PAGEINDEX_API_KEY.endswith("xxx"):
        try:
            from pageindex import PageIndex

            pi = PageIndex(api_key=PAGEINDEX_API_KEY)
            for md_file in STANDARDIZED_DIR.rglob("*.md"):
                content = md_file.read_text(encoding="utf-8")
                pi.upload(
                    content=content,
                    metadata={"filename": md_file.name, "type": md_file.parent.name},
                )
                print(f"  ✓ Uploaded: {md_file.name}")
            return
        except Exception as exc:
            print(f"  ⚠ PageIndex upload failed: {exc}")

    _load_sections()
    print(f"  ✓ Indexed {len(_sections_cache or [])} sections locally (fallback)")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    # TODO: Implement PageIndex query
    #
    # from pageindex import PageIndex
    #
    # pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    # results = pi.query(query=query, top_k=top_k)
    #
    # return [
    #     {
    #         "content": r.text,
    #         "score": r.score,
    #         "metadata": r.metadata,
    #         "source": "pageindex"
    #     }
    #     for r in results
    # ]

    if PAGEINDEX_API_KEY and not PAGEINDEX_API_KEY.endswith("xxx"):
        try:
            from pageindex import PageIndex

            pi = PageIndex(api_key=PAGEINDEX_API_KEY)
            results = pi.query(query=query, top_k=top_k)
            return [
                {
                    "content": r.text,
                    "score": r.score,
                    "metadata": r.metadata,
                    "source": "pageindex",
                }
                for r in results
            ]
        except Exception:
            pass

    return _local_structural_search(query, top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
