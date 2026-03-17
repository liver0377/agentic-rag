"""Query Rewriter Node.

Rewrites the query to improve retrieval results.

When the query has been decomposed into sub-queries, this node rewrites
each sub-query individually to preserve the semantic granularity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agent.state import AgentState


def _traced(func):
    """Decorator to trace function with langfuse if available."""
    try:
        from langfuse import observe

        return observe(name=func.__name__)(func)
    except ImportError:
        return func


def rewrite_query(
    original_query: str, evaluation_reason: str, previous_rewrites: Optional[List[str]] = None
) -> str:
    """Rewrite the query for better retrieval using rules.

    Args:
        original_query: The original query.
        evaluation_reason: Reason for insufficient results.
        previous_rewrites: Previously attempted rewrites.

    Returns:
        Rewritten query.
    """
    rewrite_strategies = []

    if "相关性较低" in evaluation_reason:
        rewrite_strategies.append(lambda q: q + " 的详细信息")
        rewrite_strategies.append(lambda q: f"关于{q}的说明文档")

    if "数量不足" in evaluation_reason:
        rewrite_strategies.append(lambda q: q.replace("如何", "步骤和方法"))
        rewrite_strategies.append(lambda q: q + " 教程 指南")

    if "匹配度较低" in evaluation_reason:
        rewrite_strategies.append(lambda q: q + " 定义 概念")
        rewrite_strategies.append(lambda q: f"什么是{q}")

    rewrite_strategies.extend(
        [
            lambda q: q + " 相关内容",
            lambda q: f"查找{q}的相关信息",
            lambda q: q.replace("的", "") + "的详细说明",
        ]
    )

    previous_rewrites = previous_rewrites or []
    for strategy in rewrite_strategies:
        rewritten = strategy(original_query)
        if rewritten != original_query and rewritten not in previous_rewrites:
            return rewritten

    return original_query + " 相关信息"


def rewrite_query_with_llm(
    original_query: str,
    evaluation_reason: str,
    previous_rewrites: List[str],
    llm_client: Any = None,
) -> str:
    """Rewrite the query for better retrieval using LLM.

    Args:
        original_query: The original query.
        evaluation_reason: Reason for insufficient results.
        previous_rewrites: Previously attempted rewrites.
        llm_client: LLM client instance.

    Returns:
        Rewritten query.
    """
    if llm_client is None:
        return rewrite_query(original_query, evaluation_reason, previous_rewrites)

    previous_str = "、".join(previous_rewrites) if previous_rewrites else "无"

    prompt = f"""用户的原始查询未能检索到足够的相关文档，需要优化查询。

原始查询：{original_query}
检索失败原因：{evaluation_reason}
已尝试的改写：{previous_str}

请生成一个新的优化查询，要求：
1. 保持原问题的核心意图
2. 使用更精确或更通用的关键词
3. 可以添加相关术语、同义词或上下文
4. 避免与已尝试的改写重复
5. 只返回改写后的查询，不要解释

优化后的查询："""

    try:
        rewritten = llm_client.chat(prompt).strip()
        if rewritten and rewritten != original_query and rewritten not in previous_rewrites:
            return rewritten
        return rewrite_query(original_query, evaluation_reason, previous_rewrites)
    except Exception:
        return rewrite_query(original_query, evaluation_reason, previous_rewrites)


def rewrite_sub_queries(
    sub_queries: List[str],
    evaluation_reason: str,
    previous_rewrites: Optional[List[List[str]]] = None,
    llm_client: Any = None,
) -> List[str]:
    """Rewrite each sub-query for better retrieval.

    Args:
        sub_queries: List of sub-queries to rewrite.
        evaluation_reason: Reason for insufficient results.
        previous_rewrites: Previously attempted rewrites (list of lists).
        llm_client: LLM client instance.

    Returns:
        List of rewritten sub-queries.
    """
    previous_rewrites = previous_rewrites or []
    previous_flat = [q for sublist in previous_rewrites for q in sublist]

    rewritten_sub_queries = []
    for i, sq in enumerate(sub_queries):
        sub_previous = (
            [
                previous_rewrites[j][i]
                for j in range(len(previous_rewrites))
                if i < len(previous_rewrites[j])
            ]
            if previous_rewrites
            else []
        )
        rewritten = rewrite_query_with_llm(
            sq, evaluation_reason, sub_previous + previous_flat, llm_client
        )
        rewritten_sub_queries.append(rewritten)

    return rewritten_sub_queries


@_traced
def rewrite_node(state: AgentState, llm_client: Any = None) -> Dict[str, Any]:
    """Rewrite the query for better retrieval.

    This node:
    1. Checks if sub-queries exist (from decomposition)
    2. If sub-queries exist, rewrites each sub-query individually
    3. If no sub-queries, rewrites the single query
    4. Returns rewritten queries for another retrieval attempt

    Args:
        state: Current agent state.
        llm_client: LLM client (injected).

    Returns:
        State updates with rewritten query or rewritten sub-queries.
    """
    evaluation_reason = state.evaluation_reason or ""
    new_count = state.rewrite_count + 1

    if state.sub_queries:
        previous_rewrites = []
        if state.rewritten_sub_queries:
            previous_rewrites.append(state.rewritten_sub_queries)

        rewritten_subs = rewrite_sub_queries(
            state.sub_queries, evaluation_reason, previous_rewrites, llm_client
        )

        decision = f"rewrite (attempt {new_count}): {len(rewritten_subs)} sub-queries rewritten"

        return {
            "rewritten_sub_queries": rewritten_subs,
            "rewrite_count": new_count,
            "decision_path": [decision],
        }
    else:
        original_query = state.original_query

        previous_rewrites = []
        if state.rewritten_query:
            previous_rewrites.append(state.rewritten_query)

        rewritten = rewrite_query_with_llm(
            original_query, evaluation_reason, previous_rewrites, llm_client
        )

        decision = (
            f"rewrite (attempt {new_count}): '{original_query[:30]}...' -> '{rewritten[:30]}...'"
        )

        return {
            "rewritten_query": rewritten,
            "rewrite_count": new_count,
            "decision_path": [decision],
        }
