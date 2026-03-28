"""Memory Recall Node.

长期记忆召回节点。

设计说明：
- 短期记忆：统一使用 AgentState.messages 字段（LangGraph 原生机制）
- 长期记忆：通过本节点从向量库召回

读取时机（按需触发）：
1. 新会话初始化：加载用户偏好类记忆，实现个性化开场
2. 问题语义相关时：通过规则 + 向量检索，动态召回相关历史记忆
3. 生成回答前：将检索到的记忆格式化后注入 prompt

缓存策略：同一 session 内的重复查询走缓存
降级策略：检索失败时优雅降级，保证主流程不受影响
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.agent.state import AgentState

logger = logging.getLogger(__name__)

_MEMORY_CACHE: OrderedDict[str, Tuple[List[Dict[str, Any]], str]] = OrderedDict()
_CACHE_MAX_SIZE = 100


@dataclass
class RecallConfig:
    """记忆召回配置。"""

    enable_cache: bool = True
    cache_max_size: int = 100
    recall_timeout_ms: int = 200
    similarity_threshold: float = 0.3


@dataclass
class RecallResult:
    """记忆召回结果。"""

    memories: List[Dict[str, Any]] = field(default_factory=list)
    context: str = ""
    from_cache: bool = False
    degraded: bool = False
    error: Optional[str] = None


def _get_cache_key(session_id: str, query: str) -> str:
    """生成缓存 key。"""
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    return f"{session_id}:{query_hash}"


def _get_from_cache(cache_key: str) -> Optional[List[Dict[str, Any]]]:
    """从缓存获取记忆。"""
    if cache_key in _MEMORY_CACHE:
        _MEMORY_CACHE.move_to_end(cache_key)
        return _MEMORY_CACHE[cache_key][0]
    return None


def _set_to_cache(cache_key: str, memories: List[Dict[str, Any]], session_id: str) -> None:
    """设置缓存。"""
    if len(_MEMORY_CACHE) >= _CACHE_MAX_SIZE:
        _MEMORY_CACHE.popitem(last=False)
    _MEMORY_CACHE[cache_key] = (memories, session_id)


def clear_session_cache(session_id: str) -> int:
    """清除指定会话的缓存。"""
    keys_to_remove = [k for k, (_, sid) in _MEMORY_CACHE.items() if sid == session_id]
    for key in keys_to_remove:
        del _MEMORY_CACHE[key]
    return len(keys_to_remove)


async def _do_recall_with_timeout(
    store: Any,
    query: str,
    user_id: str,
    memory_type: Optional[str],
    top_k: int,
    timeout_ms: int,
) -> List[Tuple[Any, float]]:
    """带超时的记忆召回。"""
    try:
        return await asyncio.wait_for(
            store.hybrid_search(
                query=query,
                user_id=user_id,
                memory_type=memory_type,
                top_k=top_k,
            ),
            timeout=timeout_ms / 1000.0,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Memory recall timed out after {timeout_ms}ms")
        return []


def _recall_preference_memories(
    store: Any,
    user_id: str,
    timeout_ms: int,
) -> List[Dict[str, Any]]:
    """召回用户偏好类记忆（新会话初始化时使用）。"""
    try:
        results = asyncio.run(
            _do_recall_with_timeout(
                store=store,
                query="用户偏好 设置 习惯",
                user_id=user_id,
                memory_type="preference",
                top_k=5,
                timeout_ms=timeout_ms,
            )
        )
        return [
            {
                "id": r.id,
                "type": r.type.value,
                "content": r.content,
                "importance": r.importance,
                "access_count": r.access_count,
            }
            for r, _ in results
            if not r.is_expired()
        ]
    except Exception as e:
        logger.warning(f"Failed to recall preference memories: {e}")
        return []


async def recall_for_new_session(
    user_id: str,
    config: Optional[RecallConfig] = None,
) -> RecallResult:
    """新会话初始化时召回用户偏好类记忆。

    Args:
        user_id: 用户 ID
        config: 召回配置

    Returns:
        RecallResult 包含偏好记忆
    """
    config = config or RecallConfig()
    memories: List[Dict[str, Any]] = []
    degraded = False
    error = None

    try:
        from src.mcp_server.memory import get_memory_store

        store = get_memory_store()
        results = await _do_recall_with_timeout(
            store=store,
            query="用户偏好 设置 习惯",
            user_id=user_id,
            memory_type="preference",
            top_k=5,
            timeout_ms=config.recall_timeout_ms,
        )

        valid_results = [(r, s) for r, s in results if not r.is_expired()]
        valid_results.sort(
            key=lambda x: (x[0].importance * 0.6 + min(x[0].access_count / 10, 0.4)),
            reverse=True,
        )

        for record, _ in valid_results[:5]:
            record.access_count += 1
            await store.upsert([record], collection="long_term_memory")

        memories = [
            {
                "id": r.id,
                "type": r.type.value,
                "content": r.content,
                "importance": r.importance,
                "access_count": r.access_count,
                "score": score,
            }
            for r, score in valid_results
        ]

        logger.info(
            f"Recalled {len(memories)} preference memories for new session (user={user_id})"
        )

    except ImportError as e:
        logger.warning(f"Memory modules not available: {e}")
        degraded = True
        error = str(e)
    except asyncio.TimeoutError:
        logger.warning(f"Memory recall timed out for new session (user={user_id})")
        degraded = True
        error = "timeout"
    except Exception as e:
        logger.error(f"Memory recall failed for new session: {e}")
        degraded = True
        error = str(e)

    return RecallResult(
        memories=memories,
        context=format_memories_for_context(memories),
        degraded=degraded,
        error=error,
    )


async def memory_recall_node(state: AgentState) -> Dict[str, Any]:
    """长期记忆召回节点。

    注意：短期记忆统一使用 AgentState.messages 字段，本节点只处理长期记忆召回。

    Args:
        state: 当前 Agent 状态

    Returns:
        状态更新
    """
    user_id = getattr(state, "user_id", None) or state.session_id or "anonymous"
    session_id = state.session_id or "default"
    config = RecallConfig()

    recalled_memories: List[Dict[str, Any]] = []
    memory_context: str = ""
    from_cache = False
    degraded = False

    if not state.need_memory:
        return {
            "recalled_memories": recalled_memories,
            "memory_context": "",
            "decision_path": ["memory_recall: skipped (no memory needed)"],
        }

    if state.memory_type == "short_term":
        return {
            "recalled_memories": recalled_memories,
            "memory_context": "",
            "decision_path": ["memory_recall: short_term uses state.messages"],
        }

    cache_key = _get_cache_key(session_id, state.original_query)
    if config.enable_cache:
        cached = _get_from_cache(cache_key)
        if cached is not None:
            recalled_memories = cached
            memory_context = format_memories_for_context(recalled_memories)
            from_cache = True
            logger.info(f"Memory recall from cache: {len(recalled_memories)} memories")
            return {
                "recalled_memories": recalled_memories,
                "memory_context": memory_context,
                "decision_path": [
                    "memory_recall: recalled={len(recalled_memories)} memories (cached)"
                ],
            }

    try:
        from src.mcp_server.memory import (
            MemoryRecord,
            MemoryType,
            get_memory_store,
        )

        store = get_memory_store()

        results = await _do_recall_with_timeout(
            store=store,
            query=state.original_query,
            user_id=user_id,
            memory_type=None,
            top_k=5,
            timeout_ms=config.recall_timeout_ms,
        )

        if not results:
            return {
                "recalled_memories": [],
                "memory_context": "",
                "decision_path": ["memory_recall: no results"],
            }

        valid_results = [(r, s) for r, s in results if not r.is_expired()]

        if config.similarity_threshold > 0:
            valid_results = [(r, s) for r, s in valid_results if s >= config.similarity_threshold]

        valid_results.sort(
            key=lambda x: (x[0].importance * 0.6 + min(x[0].access_count / 10, 0.4)),
            reverse=True,
        )

        for record, _ in valid_results[:5]:
            record.access_count += 1
            await store.upsert([record], collection="long_term_memory")

        recalled_memories = [
            {
                "id": r.id,
                "type": r.type.value,
                "content": r.content,
                "importance": r.importance,
                "access_count": r.access_count,
                "score": score,
            }
            for r, score in valid_results
        ]

        memory_context = format_memories_for_context(recalled_memories)

        if config.enable_cache and recalled_memories:
            _set_to_cache(cache_key, recalled_memories, session_id)

        logger.info(f"Recalled {len(recalled_memories)} long-term memories for user {user_id}")

    except ImportError as e:
        logger.warning(f"Memory modules not available: {e}")
        degraded = True
    except asyncio.TimeoutError:
        logger.warning(f"Memory recall timed out, degrading gracefully")
        degraded = True
    except Exception as e:
        logger.error(f"Memory recall failed: {e}")
        degraded = True

    decision = f"memory_recall: recalled={len(recalled_memories)} memories"
    if degraded:
        decision += " (degraded)"

    return {
        "recalled_memories": recalled_memories,
        "memory_context": memory_context,
        "decision_path": [decision],
    }


def format_memories_for_context(memories: List[Dict[str, Any]]) -> str:
    """格式化记忆为 Prompt 上下文。

    用于生成回答前注入 prompt。

    Args:
        memories: 记忆列表

    Returns:
        格式化后的上下文字符串
    """
    if not memories:
        return ""

    lines = ["## 用户画像与历史记忆\n"]
    for i, m in enumerate(memories, 1):
        lines.append(f"{i}. {m['content']}")
        importance = m.get("importance", 0.7)
        if importance >= 0.8:
            lines.append(f"   (重要偏好)\n")
        else:
            lines.append(f"   (相关性: {importance:.1f})\n")

    return "\n".join(lines)


async def session_init_node(state: AgentState) -> Dict[str, Any]:
    """会话初始化节点。

    新会话开始时加载用户偏好类记忆，实现个性化开场。

    Args:
        state: 当前 Agent 状态

    Returns:
        状态更新
    """
    user_id = getattr(state, "user_id", None) or state.session_id or "anonymous"
    is_new_session = getattr(state, "is_new_session", True)

    if not is_new_session:
        return {
            "decision_path": ["session_init: existing session, skip"],
        }

    result = await recall_for_new_session(user_id)

    decision = f"session_init: loaded {len(result.memories)} preference memories"
    if result.degraded:
        decision += " (degraded)"

    return {
        "recalled_memories": result.memories,
        "memory_context": result.context,
        "is_new_session": False,
        "decision_path": [decision],
    }
