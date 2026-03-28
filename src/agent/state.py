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
        rewritten_query: The rewritten query (if any) - for simple queries.
        sub_queries: List of decomposed sub-queries.
        rewritten_sub_queries: List of rewritten sub-queries (after rewrite on decomposed queries).
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
        user_id: User ID for memory management.
        session_id: Session ID for memory management.
        need_memory: Whether memory recall is needed.
        memory_type: Type of memory needed ("short_term" | "long_term").
        recalled_memories: Recalled memories from memory system.
        saved_memories: Memories saved during this turn.
    """

    original_query: str = ""
    rewritten_query: Optional[str] = None
    sub_queries: Annotated[List[str], reduce_strings] = field(default_factory=list)
    rewritten_sub_queries: Optional[List[str]] = None
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
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    need_memory: bool = False
    memory_type: Optional[str] = None
    recalled_memories: List[Dict[str, Any]] = field(default_factory=list)
    saved_memories: List[Dict[str, Any]] = field(default_factory=list)
    is_new_session: bool = False
    memory_context: str = ""

    def add_decision(self, decision: str) -> None:
        self.decision_path = self.decision_path + [decision]

    def get_current_query(self) -> str:
        return self.rewritten_query or self.original_query

    def get_queries_for_retrieval(self) -> List[str]:
        if self.rewritten_sub_queries:
            return self.rewritten_sub_queries
        if self.sub_queries:
            return self.sub_queries
        if self.rewritten_query:
            return [self.rewritten_query]
        return [self.original_query]

    def to_output_dict(self) -> Dict[str, Any]:
        return {
            "query": self.original_query,
            "response": self.final_response,
            "citations": self.citations,
            "sub_queries": self.sub_queries,
            "rewritten_query": self.rewritten_query,
            "rewritten_sub_queries": self.rewritten_sub_queries,
            "decision_path": self.decision_path,
            "total_chunks": len(self.chunks),
            "trace_id": self.trace_id,
            "error": self.error,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "need_memory": self.need_memory,
            "memory_type": self.memory_type,
            "recalled_memories": self.recalled_memories,
            "saved_memories": self.saved_memories,
            "is_new_session": self.is_new_session,
            "memory_context": self.memory_context,
        }

    def add_decision(self, decision: str) -> None:
        """Add a decision to the path."""
        self.decision_path = self.decision_path + [decision]

    def get_current_query(self) -> str:
        """Get the current query to use for retrieval (for simple queries)."""
        return self.rewritten_query or self.original_query

    def get_queries_for_retrieval(self) -> List[str]:
        """Get the queries to use for retrieval.

        Priority: rewritten_sub_queries > sub_queries > rewritten_query > original_query

        Returns:
            List of queries to retrieve.
        """
        if self.rewritten_sub_queries:
            return self.rewritten_sub_queries
        if self.sub_queries:
            return self.sub_queries
        if self.rewritten_query:
            return [self.rewritten_query]
        return [self.original_query]

    def to_output_dict(self) -> Dict[str, Any]:
        """Convert to output dictionary."""
        return {
            "query": self.original_query,
            "response": self.final_response,
            "citations": self.citations,
            "sub_queries": self.sub_queries,
            "rewritten_query": self.rewritten_query,
            "rewritten_sub_queries": self.rewritten_sub_queries,
            "decision_path": self.decision_path,
            "total_chunks": len(self.chunks),
            "trace_id": self.trace_id,
            "error": self.error,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "need_memory": self.need_memory,
            "memory_type": self.memory_type,
            "recalled_memories": self.recalled_memories,
            "saved_memories": self.saved_memories,
        }


def create_initial_state(
    query: str,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> AgentState:
    """Create initial state for a new query.

    Args:
        query: The user's question.
        trace_id: Optional trace ID.
        session_id: Optional session ID for memory.
        user_id: Optional user ID for memory.

    Returns:
        Initial AgentState.
    """
    from src.core.utils import generate_trace_id

    return AgentState(
        original_query=query,
        trace_id=trace_id or generate_trace_id(),
        session_id=session_id,
        user_id=user_id,
        decision_path=["start"],
    )
