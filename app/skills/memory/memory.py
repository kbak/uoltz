"""Long-term memory tools that call memory-mcp's REST API.

Tier 2 of the memory architecture. Tier 1 (USER.md + MEMORY.md) is injected
directly into the system prompt and doesn't use these tools.
"""

import logging

import httpx
from strands import tool

import config

logger = logging.getLogger(__name__)


def _base() -> str:
    return config.memory.base_url.rstrip("/")


@tool
def remember(content: str) -> str:
    """Save a durable fact to long-term memory.

    Call this when the user states a lasting preference, a stable fact about
    themselves or their setup, or explicitly asks you to remember something.
    Do NOT call this for transient conversation, questions you just asked,
    your own reasoning, or anything the user did not actually state.

    The write is asynchronous — this returns immediately while Mem0 extracts
    facts in the background. Safe to call mid-conversation.

    Args:
        content: The fact to remember, phrased as a short declarative statement.
    """
    try:
        resp = httpx.post(
            f"{_base()}/v1/memory",
            json={"content": content, "user_id": config.memory.default_user_id},
            timeout=config.memory.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return f"Queued ({data.get('pending', '?')} pending)."
    except Exception as e:
        logger.warning("remember failed: %s", e)
        return f"Memory save failed: {e}"


@tool
def search_memory(query: str, limit: int = 5) -> str:
    """Search long-term memory for facts relevant to a query.

    Call this when you need context from past conversations or user
    preferences that aren't in the current window. Returns the top matches
    ranked by semantic similarity.

    Args:
        query: Natural-language search query.
        limit: Maximum number of results to return.
    """
    try:
        resp = httpx.get(
            f"{_base()}/v1/memory/search",
            params={
                "query": query,
                "user_id": config.memory.default_user_id,
                "limit": limit,
            },
            timeout=config.memory.timeout,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as e:
        logger.warning("search_memory failed: %s", e)
        return f"Memory search failed: {e}"

    if not results:
        return "No matching memories."

    lines = []
    for r in results:
        score = r.get("score", 0.0)
        text = r.get("memory") or r.get("text") or ""
        lines.append(f"[{score:.2f}] {text}")
    return "\n".join(lines)
