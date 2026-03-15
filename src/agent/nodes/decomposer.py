"""Query Decomposer Node.

Decomposes complex queries into sub-queries for parallel retrieval.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from src.agent.state import AgentState
from src.agent.prompts import get_prompt


def decompose_query_with_llm(query: str, llm_client: Any = None) -> List[str]:
    """Decompose a complex query using LLM.

    Args:
        query: The complex query to decompose.
        llm_client: Optional LLM client.

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


def decompose_node(state: AgentState) -> Dict[str, Any]:
    """Decompose complex query into sub-queries.

    This node:
    1. Identifies sub-questions in the query
    2. Creates a list of sub-queries
    3. Each sub-query will be retrieved separately

    Args:
        state: Current agent state.

    Returns:
        State updates with sub-queries.
    """
    query = state.original_query

    sub_queries = decompose_query_with_llm(query)

    decision = f"decompose: {len(sub_queries)} sub-queries"

    return {
        "sub_queries": sub_queries,
        "decision_path": [decision],
    }
