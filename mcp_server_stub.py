"""
MCP stdio server stub — demo external capability cho MCP Worker.

Cấu hình trong .env:
    MCP_SERVER_COMMAND=python mcp_server_stub.py

Protocol: JSON-RPC 2.0, một request/response trên stdin/stdout.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NEWS_DIR = Path(__file__).parent / "data" / "standardized" / "news"


def search_news(query: str, top_k: int = 3) -> list[dict]:
    query_tokens = set(re.findall(r"\w+", query.lower()))
    results = []
    if not NEWS_DIR.exists():
        return results
    for md in NEWS_DIR.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        tokens = set(re.findall(r"\w+", text.lower()))
        overlap = len(query_tokens & tokens)
        if overlap:
            score = overlap / max(len(query_tokens), 1)
            results.append(
                {
                    "content": text[:2000],
                    "score": score,
                    "metadata": {"source": md.name, "type": "news", "via": "mcp_server"},
                    "source": "mcp",
                }
            )
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def handle(request: dict) -> dict:
    params = request.get("params", {})
    name = params.get("name", "")
    arguments = params.get("arguments", {})
    if name == "search_news_context":
        chunks = search_news(arguments.get("query", ""), int(arguments.get("top_k", 3)))
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {"content": [{"type": "text", "text": json.dumps({"chunks": chunks})}]},
        }
    return {
        "jsonrpc": "2.0",
        "id": request.get("id"),
        "error": {"code": -32601, "message": f"Unknown tool: {name}"},
    }


if __name__ == "__main__":
  line = sys.stdin.readline()
  if line.strip():
      req = json.loads(line)
      sys.stdout.write(json.dumps(handle(req)) + "\n")
      sys.stdout.flush()
