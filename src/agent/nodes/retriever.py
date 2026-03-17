"""Retriever Node.

Retrieves relevant chunks from the RAG MCP Server.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.agent.state import AgentState
from src.core.types import Chunk
from src.core.utils import Timer
from src.mcp_client.tools import RAGTools


def _traced(func):
    """Decorator to trace function with langfuse if available."""
    try:
        from langfuse import observe

        return observe(name=func.__name__)(func)
    except ImportError:
        return func


async def retrieve_from_rag(query: str, rag_tools: RAGTools, top_k: int = 10) -> List[Chunk]:
    """Retrieve chunks from RAG server.

    Args:
        query: Query string.
        rag_tools: RAG tools client.
        top_k: Number of results.

    Returns:
        List of retrieved chunks.
    """
    with Timer(f"retrieve: {query[:50]}..."):
        result = await rag_tools.search(query, top_k=top_k)

    if result.success and result.data:
        return result.data.chunks
    return []


async def retrieve_node(
    state: AgentState, rag_tools: Optional[RAGTools] = None, top_k: int = 10
) -> Dict[str, Any]:
    """Retrieve chunks for the query.

    This node:
    1. Gets queries for retrieval (priority: rewritten_sub_queries > sub_queries > rewritten_query > original_query)
    2. Retrieves chunks from RAG server for each query
    3. Merges and deduplicates results

    Args:
        state: Current agent state.
        rag_tools: RAG tools client (injected).
        top_k: Number of results per query.

    Returns:
        State updates with retrieved chunks.
    """
    if rag_tools is None:
        from src.core.config import load_settings

        settings = load_settings()
        rag_tools = RAGTools(settings.rag_server, use_mock=False)

    all_chunks: List[Chunk] = []

    queries = state.get_queries_for_retrieval()

    for query in queries:
        chunks = await retrieve_from_rag(query, rag_tools, top_k)
        all_chunks.extend(chunks)

    seen_ids = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk.id not in seen_ids:
            unique_chunks.append(chunk)
            seen_ids.add(chunk.id)

    unique_chunks.sort(key=lambda c: c.score, reverse=True)

    if unique_chunks:
        avg_score = sum(c.score for c in unique_chunks[:5]) / min(5, len(unique_chunks))
    else:
        avg_score = 0.0

    decision = f"retrieve: {len(unique_chunks)} chunks, avg_score={avg_score:.2f}"

    return {
        "chunks": unique_chunks,
        "retrieval_score": avg_score,
        "decision_path": [decision],
    }


@_traced
def retrieve_node_sync(
    state: AgentState, rag_tools: Optional[RAGTools] = None, top_k: int = 10
) -> Dict[str, Any]:
    """Synchronous version of retrieve node.

    Args:
        state: Current agent state.
        rag_tools: RAG tools client (injected).
        top_k: Number of results.

    Returns:
        State updates with retrieved chunks.
    """
    return asyncio.run(_retrieve_node_async_wrapper(state, rag_tools, top_k))


async def _retrieve_node_async_wrapper(
    state: AgentState, rag_tools: Optional[RAGTools], top_k: int
) -> Dict[str, Any]:
    """Async wrapper for retrieve_node."""
    return await retrieve_node(state, rag_tools, top_k)
