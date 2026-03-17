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


def _traced(func):
    """Decorator to trace function with langfuse if available."""
    try:
        from langfuse import observe

        return observe(name=func.__name__)(func)
    except ImportError:
        return func


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


def evaluate_retrieval_with_llm(
    query: str, chunks: List[Chunk], llm_client: Any = None
) -> Dict[str, Any]:
    """Evaluate retrieval results using LLM.

    Args:
        query: The original query.
        chunks: Retrieved chunks with scores from RAG.
        llm_client: LLM client instance.

    Returns:
        Evaluation result with is_sufficient, reason and suggestions.
    """
    if not chunks:
        return {
            "is_sufficient": False,
            "reason": "未检索到任何相关文档",
            "score": 0.0,
        }

    if llm_client is None:
        avg_score = sum(c.score for c in chunks[:5]) / min(5, len(chunks))
        return {
            "is_sufficient": avg_score >= 0.5 and len(chunks) >= 3,
            "reason": f"检索结果相关性: {avg_score:.2f}",
            "score": avg_score,
        }

    context = "\n\n".join(
        [
            f"[文档{i + 1}]\n{c.text[:500]}..." if len(c.text) > 500 else f"[文档{i + 1}]\n{c.text}"
            for i, c in enumerate(chunks[:5])
        ]
    )

    prompt = f"""请评估以下检索到的文档是否能充分回答用户的问题。

用户问题：{query}

检索到的文档：
{context}

请从以下方面评估：
1. 文档内容是否与问题相关
2. 信息是否足够完整
3. 是否需要补充其他信息

请以JSON格式返回评估结果：
{{
    "is_sufficient": true/false,
    "relevance_score": 0.0-1.0,
    "reason": "评估原因",
    "missing_aspects": ["缺失的方面1", "缺失的方面2"]
}}

只返回JSON，不要其他内容："""

    try:
        response = llm_client.chat(prompt)
        import json

        result = json.loads(response.strip().strip("```json").strip("```"))
        return {
            "is_sufficient": result.get("is_sufficient", False),
            "reason": result.get("reason", ""),
            "score": result.get("relevance_score", 0.0),
            "missing_aspects": result.get("missing_aspects", []),
        }
    except Exception:
        avg_score = sum(c.score for c in chunks[:5]) / min(5, len(chunks))
        return {
            "is_sufficient": avg_score >= 0.5 and len(chunks) >= 3,
            "reason": f"LLM评估失败，使用规则评估: {avg_score:.2f}",
            "score": avg_score,
        }


@_traced
def evaluate_node(
    state: AgentState, threshold: float = 0.5, llm_client: Any = None
) -> Dict[str, Any]:
    """Evaluate retrieval results.

    This node:
    1. Checks if retrieved chunks are relevant (based on RAG scores or LLM)
    2. Determines if query should be rewritten
    3. Updates evaluation result in state

    Args:
        state: Current agent state.
        threshold: Minimum score threshold.
        llm_client: LLM client (injected).

    Returns:
        State updates with evaluation results.
    """
    query = state.original_query
    chunks = state.chunks

    if llm_client is not None:
        evaluation = evaluate_retrieval_with_llm(query, chunks, llm_client)
    else:
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
