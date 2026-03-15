"""Custom metrics for Agent evaluation.

This module defines metrics specific to Agentic RAG:
- Agent performance metrics (latency, decisions)
- Response quality metrics (citations)

Note on Evaluation Architecture (see DEV_SPEC.md Section 9):
- RAG Server handles offline evaluation (Ragas: faithfulness, answer_relevancy)
- Agent handles real-time decision (evaluate_node in evaluator.py)
- This module provides post-processing metrics for observability
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentMetrics:
    """Metrics for agent performance."""

    query_length: int = 0
    query_complexity: str = "simple"
    sub_query_count: int = 0

    retrieval_count: int = 0
    avg_chunk_score: float = 0.0
    unique_chunks: int = 0

    rewrite_count: int = 0
    decision_path: List[str] = field(default_factory=list)

    response_length: int = 0
    citation_count: int = 0

    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": {
                "length": self.query_length,
                "complexity": self.query_complexity,
                "sub_query_count": self.sub_query_count,
            },
            "retrieval": {
                "count": self.retrieval_count,
                "avg_chunk_score": self.avg_chunk_score,
                "unique_chunks": self.unique_chunks,
            },
            "decisions": {
                "rewrite_count": self.rewrite_count,
                "path": self.decision_path,
            },
            "response": {
                "length": self.response_length,
                "citation_count": self.citation_count,
            },
            "timing": {
                "total_latency_ms": self.total_latency_ms,
            },
        }


def calculate_metrics(agent_output: Dict[str, Any]) -> AgentMetrics:
    """Calculate metrics from agent output.

    Args:
        agent_output: Output from the agent.

    Returns:
        Calculated metrics.
    """
    metrics = AgentMetrics()

    query = agent_output.get("query", "")
    metrics.query_length = len(query)

    if "复杂" in query or "和" in query and query.count("和") > 1:
        metrics.query_complexity = "complex"
    elif "比较" in query or "区别" in query:
        metrics.query_complexity = "comparative"
    else:
        metrics.query_complexity = "simple"

    sub_queries = agent_output.get("sub_queries", [])
    metrics.sub_query_count = len(sub_queries)

    chunks_data = agent_output.get("chunks", [])
    metrics.retrieval_count = len(chunks_data)

    if chunks_data:
        scores = [c.get("score", 0) for c in chunks_data if isinstance(c, dict)]
        metrics.avg_chunk_score = sum(scores) / len(scores) if scores else 0
        metrics.unique_chunks = len(
            set(c.get("id", "") for c in chunks_data if isinstance(c, dict))
        )

    metrics.rewrite_count = agent_output.get("rewrite_count", 0)
    metrics.decision_path = agent_output.get("decision_path", [])

    response = agent_output.get("response", "")
    metrics.response_length = len(response)

    citations = agent_output.get("citations", [])
    metrics.citation_count = len(citations)

    return metrics


def calculate_citation_coverage(response: str, citations: List[Dict[str, Any]]) -> float:
    """Calculate how well citations cover the response.

    Args:
        response: Generated response.
        citations: List of citations.

    Returns:
        Coverage score from 0 to 1.
    """
    if not response or not citations:
        return 0.0

    citation_refs = 0
    for i in range(1, len(citations) + 1):
        if f"[文档{i}]" in response or f"[{i}]" in response:
            citation_refs += 1

    return citation_refs / len(citations) if citations else 0.0
