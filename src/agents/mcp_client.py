"""
MCP client — gọi external capability qua MCP tool protocol.

Hỗ trợ:
  1. MCP stdio server (MCP_SERVER_COMMAND trong .env)
  2. Local fallback: search trong news markdown đã crawl (Task 2)

Tool schema tuân theo MCP tools/list + tools/call.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

NEWS_DIR = Path(__file__).parent.parent.parent / "data" / "standardized" / "news"

MCP_TOOLS = [
    {
        "name": "search_news_context",
        "description": "Tìm tin tức liên quan từ nguồn external (MCP) hoặc local news index",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Câu truy vấn tìm tin"},
                "top_k": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    }
]


def list_tools() -> list[dict]:
    return MCP_TOOLS


def _local_news_search(query: str, top_k: int = 3) -> list[dict]:
    """Fallback: keyword search trên news markdown đã standardized."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return []

    candidates: list[tuple[float, dict]] = []
    search_dirs = [NEWS_DIR, NEWS_DIR.parent / "news"]
    seen: set[str] = set()

    for base in search_dirs:
        if not base.exists():
            continue
        for md_file in base.rglob("*.md"):
            key = str(md_file)
            if key in seen:
                continue
            seen.add(key)
            try:
                content = md_file.read_text(encoding="utf-8")
            except OSError:
                continue
            content_tokens = set(re.findall(r"\w+", content.lower()))
            overlap = len(query_tokens & content_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(query_tokens), 1)
            candidates.append(
                (
                    score,
                    {
                        "content": content[:2500],
                        "score": score,
                        "metadata": {
                            "source": md_file.name,
                            "type": "news",
                            "via": "mcp_local_fallback",
                        },
                        "source": "mcp",
                    },
                )
            )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in candidates[:top_k]]


def _call_mcp_stdio(tool_name: str, arguments: dict) -> dict | None:
    """Gọi MCP server qua stdio JSON-RPC (nếu MCP_SERVER_COMMAND được cấu hình)."""
    cmd = os.getenv("MCP_SERVER_COMMAND", "").strip()
    if not cmd:
        return None

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return None
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        if not lines:
            return None
        response = json.loads(lines[-1])
        return response.get("result", {})
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None


def call_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Gọi MCP tool. Trả về { success, chunks, via }.

    via: "mcp_server" | "mcp_local_fallback"
    """
    if tool_name != "search_news_context":
        return {"success": False, "chunks": [], "via": "none", "error": f"Unknown tool: {tool_name}"}

    query = arguments.get("query", "")
    top_k = int(arguments.get("top_k", 3))

    mcp_result = _call_mcp_stdio(tool_name, arguments)
    if mcp_result and "content" in mcp_result:
        try:
            text = mcp_result["content"][0]["text"]
            parsed = json.loads(text)
            chunks = parsed.get("chunks", [])
            return {"success": True, "chunks": chunks, "via": "mcp_server"}
        except (KeyError, IndexError, json.JSONDecodeError):
            pass

    chunks = _local_news_search(query, top_k)
    return {
        "success": bool(chunks),
        "chunks": chunks,
        "via": "mcp_local_fallback",
    }
