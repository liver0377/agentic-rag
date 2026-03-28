"""Memory Router Node.

Determines if a query needs long-term memory recall.

设计说明：
- 短期记忆：统一使用 AgentState.messages 字段（LangGraph 原生机制），无需手动召回
- 长期记忆：通过 memory_recall_node 从向量库召回

读取时机（按需触发）：
1. 新会话初始化：加载用户偏好类记忆，实现个性化开场
2. 问题语义相关时：通过规则判断，动态召回长期记忆
3. 生成回答前：将检索到的记忆格式化后注入 prompt

轻量级过滤：避免每次都查询向量库
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from src.agent.state import AgentState

logger = logging.getLogger(__name__)

LONG_TERM_INDICATORS = {
    "上次",
    "之前",
    "以前",
    "历史",
    "之前问过",
    "上个会话",
    "昨天",
    "上周",
    "上次讨论",
    "之前说的",
    "上次提到",
    "以前设置",
    "之前配置",
    "我记得",
    "我之前",
    "老样子",
    "和上次一样",
}

SHORT_TERM_INDICATORS = [
    "它",
    "这个",
    "上面",
    "刚才",
    "前面提到的",
    "那个",
    "这两个",
    "上述",
    "刚刚",
    "前面说的",
    "之前提到的",
]

PREFERENCE_QUERY_INDICATORS = [
    "我喜欢",
    "我偏好",
    "记住",
    "我习惯",
    "以后都这样",
    "帮我记住",
    "不要忘记",
    "我更喜欢",
    "我一直用",
    "我的习惯是",
    "请记住",
    "记得",
    "别忘了",
    "默认",
]

SESSION_HISTORY_INDICATORS = [
    "上次",
    "之前",
    "以前",
    "历史",
    "之前问过",
    "上个会话",
    "昨天",
    "上周",
    "上次讨论",
    "之前说的",
    "上次提到",
]

KNOWN_SESSIONS: Set[str] = set()


@dataclass
class MemoryRoutingResult:
    """Result of memory routing analysis."""

    need_memory: bool
    memory_type: Optional[str]
    reason: str
    confidence: float = 1.0
    is_new_session: bool = False
    is_preference_query: bool = False


def is_new_session(session_id: Optional[str]) -> bool:
    """Check if this is a new session.

    Args:
        session_id: Session ID to check.

    Returns:
        True if this is a new session.
    """
    if not session_id:
        return False
    if session_id in KNOWN_SESSIONS:
        return False
    KNOWN_SESSIONS.add(session_id)
    return True


def detect_preference_query(query: str) -> bool:
    """Detect if query contains preference indicators.

    轻量级过滤: 快判断是否需要召回长期记忆

    Args:
        query: User's query.

    Returns:
        True if preference indicators detected.
    """
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in PREFERENCE_QUERY_INDICATORS)


def detect_memory_need(
    query: str,
    has_history: bool = False,
    is_new: bool = False,
) -> MemoryRoutingResult:
    """Detect if query needs long-term memory recall.

    设计说明：
    - 短期记忆使用 AgentState.messages，无需手动召回
    - 本函数只判断是否需要召回长期记忆

    轻量级过滤策略：
    1. 新会话 → 召回长期记忆（偏好类）
    2. 检测到长期记忆关键词 → 召回长期记忆
    3. 检测到短期记忆关键词 → 无需召回（messages 已包含）
    4. 偏好类查询 → 召回长期记忆

    Args:
        query: The user's query.
        has_history: Whether there are previous messages in the conversation.
        is_new: Whether this is a new session.

    Returns:
        MemoryRoutingResult with routing decision.
    """
    query_lower = query.lower()

    if is_new:
        return MemoryRoutingResult(
            need_memory=True,
            memory_type="long_term",
            reason="新会话初始化，加载用户偏好",
            confidence=0.9,
            is_new_session=True,
        )

    for indicator in LONG_TERM_INDICATORS:
        if indicator in query_lower:
            return MemoryRoutingResult(
                need_memory=True,
                memory_type="long_term",
                reason=f"检测到长期记忆关键词: '{indicator}'",
                confidence=0.9,
            )

    for indicator in SESSION_HISTORY_INDICATORS:
        if indicator in query_lower:
            return MemoryRoutingResult(
                need_memory=True,
                memory_type="long_term",
                reason=f"检测到历史引用关键词: '{indicator}'",
                confidence=0.9,
            )

    if detect_preference_query(query):
        return MemoryRoutingResult(
            need_memory=True,
            memory_type="long_term",
            reason="检测到偏好类查询",
            confidence=0.8,
        )

    for indicator in SHORT_TERM_INDICATORS:
        if indicator in query_lower:
            return MemoryRoutingResult(
                need_memory=False,
                memory_type=None,
                reason=f"短期记忆关键词 '{indicator}'，使用 state.messages",
                confidence=0.85,
            )

    if has_history:
        return MemoryRoutingResult(
            need_memory=False,
            memory_type=None,
            reason="存在历史对话，使用 state.messages",
            confidence=0.6,
        )

    return MemoryRoutingResult(
        need_memory=False,
        memory_type=None,
        reason="无需长期记忆召回",
        confidence=1.0,
    )


def memory_router_node(state: AgentState) -> Dict[str, Any]:
    """Route to appropriate memory type based on query analysis.

    Args:
        state: Current agent state.

    Returns:
        State updates with memory routing decision.
    """
    is_new = is_new_session(state.session_id) if state.session_id else False
    has_history = len(state.messages) > 0 if state.messages else False

    routing_result = detect_memory_need(
        query=state.original_query,
        has_history=has_history,
        is_new=is_new,
    )

    decision = f"memory_router: need={routing_result.need_memory}"
    if routing_result.need_memory:
        decision += f", type={routing_result.memory_type}"
    if routing_result.is_new_session:
        decision += " (new session)"

    logger.info(
        f"Memory routing: need_memory={routing_result.need_memory}, "
        f"type={routing_result.memory_type}, reason={routing_result.reason}"
    )

    return {
        "need_memory": routing_result.need_memory,
        "memory_type": routing_result.memory_type,
        "is_new_session": routing_result.is_new_session,
        "decision_path": [decision],
    }


def should_recall_memory(state: AgentState) -> str:
    """Determine if memory recall node should be executed.

    注意：短期记忆使用 AgentState.messages，本函数只判断长期记忆。

    Args:
        state: Current agent state.

    Returns:
        "recall" if long-term memory is needed, "skip" otherwise.
    """
    if state.need_memory and state.memory_type == "long_term":
        return "recall"
    return "skip"
