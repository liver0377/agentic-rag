"""Type definitions for Agentic RAG Assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """Retrieved chunk from RAG server."""

    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            score=data.get("score", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Citation:
    """Citation for a source document."""

    chunk_id: str
    source_path: Optional[str] = None
    page_num: Optional[int] = None
    text_snippet: Optional[str] = None
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "page_num": self.page_num,
            "text_snippet": self.text_snippet,
            "score": self.score,
        }

    @classmethod
    def from_chunk(cls, chunk: Chunk, snippet_length: int = 100) -> "Citation":
        text_snippet = (
            chunk.text[:snippet_length] + "..." if len(chunk.text) > snippet_length else chunk.text
        )
        return cls(
            chunk_id=chunk.id,
            source_path=chunk.metadata.get("source_path"),
            page_num=chunk.metadata.get("page_num"),
            text_snippet=text_snippet,
            score=chunk.score,
        )


@dataclass
class AgentOutput:
    """Final output from the agent."""

    query: str
    response: str
    citations: List[Citation] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    rewritten_queries: List[str] = field(default_factory=list)
    decision_path: List[str] = field(default_factory=list)
    total_chunks: int = 0
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "citations": [c.to_dict() for c in self.citations],
            "sub_queries": self.sub_queries,
            "rewritten_queries": self.rewritten_queries,
            "decision_path": self.decision_path,
            "total_chunks": self.total_chunks,
            "trace_id": self.trace_id,
        }


@dataclass
class RetrievalResult:
    """Result from RAG server retrieval."""

    chunks: List[Chunk]
    collection: str
    query: str
    total_count: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        chunks = [Chunk.from_dict(c) for c in data.get("chunks", [])]
        return cls(
            chunks=chunks,
            collection=data.get("collection", "default"),
            query=data.get("query", ""),
            total_count=len(chunks),
        )
