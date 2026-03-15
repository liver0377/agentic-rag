"""Agent State definition for LangGraph.

This module defines the state that flows through the Agent's state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import add_messages

from src.core.types import Chunk


def reduce_chunks(left: List[Chunk], right: List[Chunk]) -> List[Chunk]:
    """Reducer for chunks - deduplicates by id."""
    seen = {c.id for c in left}
    result = list(left)
    for chunk in right:
        if chunk.id not in seen:
            result.append(chunk)
            seen.add(chunk.id)
    return result


def reduce_strings(left: List[str], right: List[str]) -> List[str]:
    """Reducer for string lists - appends without duplicates."""
    seen = set(left)
    result = list(left)
    for s in right:
        if s not in seen:
            result.append(s)
            seen.add(s)
    return result


@dataclass(kw_only=True)
class AgentState:
    """State that flows through the Agent graph.

    Attributes:
        original_query: The user's original question.
        rewritten_query: The rewritten query (if any).
        sub_queries: List of decomposed sub-queries.
        chunks: Retrieved chunks from RAG server.
        retrieval_score: Score indicating retrieval quality.
        is_sufficient: Whether retrieved chunks are sufficient.
        evaluation_reason: Reason for the evaluation decision.
        rewrite_count: Number of query rewrites attempted.
        final_response: The generated response.
        citations: List of citations.
        trace_id: Trace ID for observability.
        decision_path: List of decisions made by the agent.
        messages: Conversation messages (for multi-turn support).
        error: Error message if any.
    """

    original_query: str = ""
    rewritten_query: Optional[str] = None
    sub_queries: Annotated[List[str], reduce_strings] = field(default_factory=list)
    chunks: Annotated[List[Chunk], reduce_chunks] = field(default_factory=list)
    retrieval_score: Optional[float] = None
    is_sufficient: Optional[bool] = None
    evaluation_reason: Optional[str] = None
    rewrite_count: int = 0
    final_response: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    trace_id: Optional[str] = None
    decision_path: Annotated[List[str], reduce_strings] = field(default_factory=list)
    messages: Annotated[List[Dict[str, str]], add_messages] = field(default_factory=list)
    error: Optional[str] = None

    def add_decision(self, decision: str) -> None:
        """Add a decision to the path."""
        self.decision_path = self.decision_path + [decision]

    def get_current_query(self) -> str:
        """Get the current query to use for retrieval."""
        return self.rewritten_query or self.original_query

    def to_output_dict(self) -> Dict[str, Any]:
        """Convert to output dictionary."""
        return {
            "query": self.original_query,
            "response": self.final_response,
            "citations": self.citations,
            "sub_queries": self.sub_queries,
            "rewritten_query": self.rewritten_query,
            "decision_path": self.decision_path,
            "total_chunks": len(self.chunks),
            "trace_id": self.trace_id,
            "error": self.error,
        }


def create_initial_state(query: str, trace_id: Optional[str] = None) -> AgentState:
    """Create initial state for a new query.

    Args:
        query: The user's question.
        trace_id: Optional trace ID.

    Returns:
        Initial AgentState.
    """
    from src.core.utils import generate_trace_id

    return AgentState(
        original_query=query,
        trace_id=trace_id or generate_trace_id(),
        decision_path=["start"],
    )
