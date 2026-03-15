"""Query Analyzer Node.

Analyzes the user's query to determine:
1. Query complexity (simple vs complex)
2. Query type (factual, procedural, analytical, etc.)
3. Whether decomposition is needed
"""

from __future__ import annotations

import json
import re
from typing import Dict, Any

from src.agent.state import AgentState
from src.agent.prompts import get_prompt


def analyze_query(query: str) -> Dict[str, Any]:
    """Analyze query characteristics.

    Args:
        query: The user's query.

    Returns:
        Analysis result with complexity, type, etc.
    """
    query_lower = query.lower()
    word_count = len(query.split())

    complexity_indicators = [
        "和",
        "以及",
        "同时",
        "另外",
        "还有",
        "并且",
        "首先",
        "其次",
        "最后",
        "如何",
        "为什么",
        "比较",
        "区别",
        "关系",
        "影响",
    ]

    complexity_score = sum(1 for ind in complexity_indicators if ind in query_lower)

    if complexity_score >= 2 or word_count > 30 or "?" in query and query.count("?") > 1:
        is_complex = True
    elif complexity_score >= 1 or word_count > 20:
        is_complex = "比较" in query or "区别" in query or "和" in query
    else:
        is_complex = False

    if re.search(r"(如何|怎么|步骤|流程|方法)", query_lower):
        query_type = "procedural"
    elif re.search(r"(为什么|原因|导致|影响)", query_lower):
        query_type = "analytical"
    elif re.search(r"(是什么|什么是|定义|概念)", query_lower):
        query_type = "factual"
    elif re.search(r"(比较|区别|不同|相同)", query_lower):
        query_type = "comparative"
    else:
        query_type = "general"

    needs_decomposition = is_complex and (
        query_type in ["comparative", "analytical"] or query.count("和") + query.count("以及") >= 2
    )

    return {
        "is_complex": is_complex,
        "query_type": query_type,
        "needs_decomposition": needs_decomposition,
        "word_count": word_count,
        "complexity_score": complexity_score,
    }


def analyze_node(state: AgentState) -> Dict[str, Any]:
    """Analyze the query and determine next steps.

    This node:
    1. Analyzes query complexity
    2. Determines if decomposition is needed
    3. Adds decision to path

    Args:
        state: Current agent state.

    Returns:
        State updates with analysis results.
    """
    query = state.original_query

    analysis = analyze_query(query)

    decision = f"analyze: {'complex' if analysis['is_complex'] else 'simple'} query, type={analysis['query_type']}"

    return {
        "decision_path": [decision],
        "sub_queries": [],
    }


def should_decompose(state: AgentState) -> str:
    """Determine if query should be decomposed.

    Args:
        state: Current agent state.

    Returns:
        "decompose" or "retrieve" based on analysis.
    """
    query = state.original_query
    analysis = analyze_query(query)

    if analysis["needs_decomposition"]:
        return "decompose"
    return "retrieve"
