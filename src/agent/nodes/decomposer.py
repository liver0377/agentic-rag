"""Query Decomposer Node.

Decomposes complex queries into sub-queries for parallel retrieval.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from src.agent.state import AgentState


def decompose_query_by_rule(query: str) -> List[str]:
    """Decompose a complex query using rule-based approach.

    Args:
        query: The complex query to decompose.

    Returns:
        List of sub-queries.
    """
    sub_queries = []

    connectors = ["和", "以及", "另外", "同时", "并且"]

    parts = [query]
    for connector in connectors:
        new_parts = []
        for part in parts:
            if connector in part:
                new_parts.extend(part.split(connector))
            else:
                new_parts.append(part)
        parts = new_parts

    sub_queries = [p.strip() for p in parts if p.strip() and len(p.strip()) > 5]

    if len(sub_queries) <= 1:
        if "比较" in query or "区别" in query:
            match = re.search(r"(.+?)和(.+?)(?:的|之间)?(?:比较|区别|不同)", query)
            if match:
                sub_queries = [
                    f"什么是{match.group(1).strip()}",
                    f"什么是{match.group(2).strip()}",
                    f"{match.group(1).strip()}和{match.group(2).strip()}的区别",
                ]
        elif "如何" in query or "怎么" in query:
            match = re.search(r"(如何|怎么)(.+)", query)
            if match:
                sub_queries = [query, f"{match.group(2).strip()}的步骤"]

    return sub_queries if len(sub_queries) > 1 else [query]


def decompose_query_with_llm(query: str, llm_client: Any = None) -> List[str]:
    """Decompose a complex query using LLM.

    Args:
        query: The complex query to decompose.
        llm_client: LLM client instance.

    Returns:
        List of sub-queries.
    """
    if llm_client is None:
        return decompose_query_by_rule(query)

    prompt = f"""请将以下复杂问题分解为多个独立的子问题。

要求：
1. 每个子问题应该是独立可回答的
2. 子问题之间不要有重叠
3. 保留原问题的核心意图
4. 按逻辑顺序排列子问题
5. 每行一个子问题，不要编号

复杂问题：{query}

子问题列表："""

    try:
        response = llm_client.chat(prompt)
        sub_queries = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip()
            and not line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "-", "•"))
        ]

        for i, sq in enumerate(sub_queries):
            sq = sq.lstrip("0123456789.-•) ")
            sub_queries[i] = sq

        sub_queries = [sq for sq in sub_queries if len(sq) > 3]

        return sub_queries if len(sub_queries) > 1 else [query]
    except Exception:
        return decompose_query_by_rule(query)


def decompose_node(state: AgentState, llm_client: Any = None) -> Dict[str, Any]:
    """Decompose complex query into sub-queries.

    This node:
    1. Identifies sub-questions in the query
    2. Creates a list of sub-queries (with LLM if available)
    3. Each sub-query will be retrieved separately

    Args:
        state: Current agent state.
        llm_client: LLM client (injected).

    Returns:
        State updates with sub-queries.
    """
    query = state.original_query

    sub_queries = decompose_query_with_llm(query, llm_client)

    decision = f"decompose: {len(sub_queries)} sub-queries"

    return {
        "sub_queries": sub_queries,
        "decision_path": [decision],
    }
