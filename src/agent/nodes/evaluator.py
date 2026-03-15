"""Retrieval Evaluator Node.

Evaluates whether retrieved chunks are sufficient to answer the query.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agent.state import AgentState
from src.core.types import Chunk


def evaluate_retrieval(query: str, chunks: List[Chunk], threshold: float = 0.5) -> Dict[str, Any]:
    """Evaluate if retrieval results are sufficient.

    Args:
        query: The original query.
        chunks: Retrieved chunks.
        threshold: Minimum acceptable score.

    Returns:
        Evaluation result with is_sufficient and reason.
    """
    if not chunks:
        return {
            "is_sufficient": False,
            "reason": "未检索到任何相关文档",
            "score": 0.0,
        }

    top_chunks = chunks[:5]
    avg_score = sum(c.score for c in top_chunks) / len(top_chunks)

    query_lower = query.lower()
    query_keywords = set(query_lower.split())

    matched_chunks = 0
    for chunk in top_chunks:
        chunk_lower = chunk.text.lower()
        matches = sum(1 for kw in query_keywords if kw in chunk_lower)
        if matches >= 2:
            matched_chunks += 1

    keyword_match_ratio = matched_chunks / len(top_chunks)

    combined_score = avg_score * 0.6 + keyword_match_ratio * 0.4

    if avg_score < 0.3:
        return {
            "is_sufficient": False,
            "reason": f"检索结果相关性较低 (score={avg_score:.2f})，建议改写查询",
            "score": combined_score,
        }

    if len(chunks) < 3:
        return {
            "is_sufficient": False,
            "reason": f"检索结果数量不足 ({len(chunks)}个)，建议改写查询获取更多结果",
            "score": combined_score,
        }

    if keyword_match_ratio < 0.3:
        return {
            "is_sufficient": False,
            "reason": "检索结果与查询关键词匹配度较低，建议改写查询",
            "score": combined_score,
        }

    return {
        "is_sufficient": combined_score >= threshold,
        "reason": f"检索结果较为充分 (score={combined_score:.2f})"
        if combined_score >= threshold
        else f"检索结果质量有待提高 (score={combined_score:.2f})",
        "score": combined_score,
    }


def evaluate_node(state: AgentState, threshold: float = 0.5) -> Dict[str, Any]:
    """Evaluate retrieval results.

    This node:
    1. Checks if retrieved chunks are relevant
    2. Determines if query should be rewritten
    3. Updates evaluation result in state

    Args:
        state: Current agent state.
        threshold: Minimum score threshold.

    Returns:
        State updates with evaluation results.
    """
    query = state.original_query
    chunks = state.chunks

    evaluation = evaluate_retrieval(query, chunks, threshold)

    decision = f"evaluate: {'sufficient' if evaluation['is_sufficient'] else 'insufficient'} (score={evaluation['score']:.2f})"

    return {
        "is_sufficient": evaluation["is_sufficient"],
        "evaluation_reason": evaluation["reason"],
        "retrieval_score": evaluation["score"],
        "decision_path": [decision],
    }


def should_rewrite(state: AgentState, max_attempts: int = 2) -> str:
    """Determine if query should be rewritten.

    Args:
        state: Current agent state.
        max_attempts: Maximum rewrite attempts.

    Returns:
        "rewrite" or "generate" based on evaluation.
    """
    if state.is_sufficient:
        return "generate"

    if state.rewrite_count >= max_attempts:
        return "generate"

    return "rewrite"
