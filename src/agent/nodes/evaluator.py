"""Retrieval Evaluator Node.

Evaluates whether retrieved chunks are sufficient to answer the query.
This is a decision node for flow control, NOT a quality evaluation.

Design Decision:
- RAG Server handles offline evaluation (Ragas: faithfulness, answer_relevancy)
- Agent handles real-time decision (whether to rewrite query)
- These are complementary, not redundant

See DEV_SPEC.md Section 9 for architecture details.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agent.state import AgentState
from src.core.types import Chunk


def evaluate_retrieval(query: str, chunks: List[Chunk], threshold: float = 0.5) -> Dict[str, Any]:
    """Evaluate if retrieval results are sufficient.

    This is a simplified decision logic based on RAG-returned scores.
    No additional LLM calls or keyword matching - just threshold-based decision.

    Args:
        query: The original query (for logging purposes).
        chunks: Retrieved chunks with scores from RAG.
        threshold: Minimum acceptable average score.

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

    min_chunks_required = 3
    is_sufficient = avg_score >= threshold and len(chunks) >= min_chunks_required

    if not is_sufficient:
        if avg_score < threshold:
            reason = f"检索结果相关性较低 (avg_score={avg_score:.2f} < threshold={threshold})"
        else:
            reason = f"检索结果数量不足 ({len(chunks)} < {min_chunks_required})"
    else:
        reason = f"检索结果充分 (avg_score={avg_score:.2f}, chunks={len(chunks)})"

    return {
        "is_sufficient": is_sufficient,
        "reason": reason,
        "score": avg_score,
    }


def evaluate_node(state: AgentState, threshold: float = 0.5) -> Dict[str, Any]:
    """Evaluate retrieval results.

    This node:
    1. Checks if retrieved chunks are relevant (based on RAG scores)
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
