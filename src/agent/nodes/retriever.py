"""Retriever Node.

Retrieves relevant chunks from the RAG MCP Server.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agent.state import AgentState
from src.core.types import Chunk
from src.core.utils import Timer
from src.mcp_client.tools import RAGTools, format_chunks_for_context


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
    1. Gets the current query (original or rewritten)
    2. Retrieves chunks from RAG server
    3. If sub-queries exist, retrieves for each

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
        rag_tools = RAGTools(settings.rag_server, use_mock=True)

    all_chunks: List[Chunk] = []

    queries = state.sub_queries if state.sub_queries else [state.get_current_query()]

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


def retrieve_node_sync(state: AgentState) -> Dict[str, Any]:
    """Synchronous version of retrieve node for simple graph.

    Args:
        state: Current agent state.

    Returns:
        State updates with mock chunks.
    """
    query = state.get_current_query()

    mock_chunks = [
        Chunk(
            id=f"mock_{i}",
            text=f"这是关于 '{query}' 的模拟检索结果 {i + 1}。",
            score=0.9 - i * 0.15,
            metadata={"source_path": f"doc_{i}.pdf", "page_num": i + 1},
        )
        for i in range(5)
    ]

    avg_score = sum(c.score for c in mock_chunks) / len(mock_chunks)

    decision = f"retrieve: {len(mock_chunks)} chunks, avg_score={avg_score:.2f}"

    return {
        "chunks": mock_chunks,
        "retrieval_score": avg_score,
        "decision_path": [decision],
    }
