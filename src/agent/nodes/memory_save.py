"""Memory Save Node.

在对话结束后检测触发条件，使用 LLM 提取关键信息，保存到长期记忆。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.agent.state import AgentState

logger = logging.getLogger(__name__)


MEMORY_EXTRACTION_PROMPT = """请从以下用户对话中提取需要长期记住的关键信息。

## 用户对话
{user_query}

## Agent 回复
{assistant_response}

## 触发关键词
{trigger_keyword}

## 提取要求

根据触发关键词类型，提取对应类型的记忆：

1. **如果是偏好类关键词**（我喜欢、我偏好、记住等）：
   - 提取用户的偏好信息
   - 格式：简洁的一句话描述

2. **如果是任务完成类关键词**（完成了、成功了等）：
   - 提取完成的任务类型和结果
   - 格式：任务描述 + 结果

3. **如果是会话结束类关键词**（再见、结束等）：
   - 提取本次会话的关键事实
   - 格式：关键事实描述

## 输出格式

请严格按照以下 JSON 格式输出：

```json
{{
    "extracted_content": "提取的关键信息（一句话）",
    "memory_type": "preference" 或 "fact" 或 "task",
    "importance": 0.5 到 1.0 之间的数值，
    "confidence": 0.0 到 1.0 之间的数值
}}
```

注意：
- 只输出 JSON，不要包含其他内容
- `extracted_content` 应该简洁明确，便于后续检索
- `importance` 根据信息的重要性评估，用户主动表达的偏好重要性较高（0.8-1.0）
- `confidence` 表示对提取结果的置信度
"""


class MemoryExtractor:
    """使用 LLM 提取记忆内容的提取器。"""

    def __init__(self, llm_client: Optional[Any] = None):
        self._llm_client = llm_client

    def _get_llm_client(self) -> Any:
        if self._llm_client is None:
            from src.core.llm_client import create_llm_client

            self._llm_client = create_llm_client()
        return self._llm_client

    async def extract(
        self,
        user_query: str,
        assistant_response: Optional[str],
        trigger_keyword: str,
        trigger_type: str,
    ) -> Dict[str, Any]:
        """使用 LLM 提取记忆内容。

        Args:
            user_query: 用户查询
            assistant_response: Agent 回复
            trigger_keyword: 触发关键词
            trigger_type: 触发类型

        Returns:
            提取结果字典
        """
        import json
        import re

        prompt = MEMORY_EXTRACTION_PROMPT.format(
            user_query=user_query,
            assistant_response=assistant_response or "无",
            trigger_keyword=trigger_keyword,
        )

        try:
            llm = self._get_llm_client()
            response = await llm.chat_async(prompt, temperature=0.1, max_tokens=500)

            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)

            return {
                "success": True,
                "extracted_content": result.get("extracted_content", ""),
                "memory_type": result.get("memory_type", "fact"),
                "importance": float(result.get("importance", 0.7)),
                "confidence": float(result.get("confidence", 0.8)),
            }

        except Exception as e:
            logger.warning(f"LLM extraction failed, using fallback: {e}")
            return self._fallback_extract(
                user_query, assistant_response, trigger_keyword, trigger_type
            )

    def _fallback_extract(
        self,
        user_query: str,
        assistant_response: Optional[str],
        trigger_keyword: str,
        trigger_type: str,
    ) -> Dict[str, Any]:
        """降级：基于规则的简单提取。"""
        type_mapping = {
            "preference": ("preference", 0.7),
            "task_completion": ("fact", 0.8),
            "session_end": ("fact", 0.5),
        }

        memory_type, importance = type_mapping.get(trigger_type, ("fact", 0.5))

        idx = user_query.find(trigger_keyword)
        if idx >= 0:
            start = max(0, idx - 10)
            end = min(len(user_query), idx + len(trigger_keyword) + 30)
            content = user_query[start:end].strip()
        else:
            content = user_query

        return {
            "success": True,
            "extracted_content": f"用户偏好: {content}",
            "memory_type": memory_type,
            "importance": importance,
            "confidence": 0.6,
        }


_extractor_instance: Optional[MemoryExtractor] = None


def get_memory_extractor(llm_client: Optional[Any] = None) -> MemoryExtractor:
    """获取提取器单例。"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = MemoryExtractor(llm_client=llm_client)
    return _extractor_instance


async def memory_save_node(state: AgentState) -> Dict[str, Any]:
    """记忆保存节点。

    检测触发条件，使用 LLM 提取关键信息，保存到长期记忆。

    Args:
        state: 当前 Agent 状态

    Returns:
        状态更新
    """
    saved_memories: List[Dict[str, Any]] = []
    trigger_results: List[Dict[str, Any]] = []

    user_id = getattr(state, "user_id", None) or state.session_id or "anonymous"
    session_id = state.session_id or "default"

    try:
        from src.mcp_server.memory import (
            MemoryRecord,
            MemoryType,
            MemoryTrigger,
            TriggerType,
            get_memory_trigger,
            get_memory_store,
            get_session_manager,
        )

        trigger = get_memory_trigger()

        result = trigger.detect(
            user_query=state.original_query,
            assistant_response=state.final_response,
        )

        if result.triggered:
            trigger_results.append(
                {
                    "type": result.trigger_type.value if result.trigger_type else None,
                    "keyword": result.matched_keyword,
                    "raw_content": result.content_to_save,
                }
            )

            if result.trigger_type != TriggerType.SESSION_END:
                extractor = get_memory_extractor()

                extraction = await extractor.extract(
                    user_query=state.original_query,
                    assistant_response=state.final_response,
                    trigger_keyword=result.matched_keyword or "",
                    trigger_type=result.trigger_type.value if result.trigger_type else "",
                )

                if extraction["success"] and extraction["extracted_content"]:
                    type_mapping = {
                        "preference": MemoryType.PREFERENCE,
                        "fact": MemoryType.FACT,
                        "task": MemoryType.FACT,
                    }
                    memory_type = type_mapping.get(extraction["memory_type"], MemoryType.FACT)

                    memory_record = MemoryRecord(
                        type=memory_type,
                        content=extraction["extracted_content"],
                        metadata={
                            "user_id": user_id,
                            "source_session_id": session_id,
                            "trigger_keyword": result.matched_keyword or "",
                            "extraction_confidence": extraction["confidence"],
                        },
                        importance=extraction["importance"],
                    )

                    try:
                        store = get_memory_store()
                        await store.upsert([memory_record], collection="long_term_memory")
                        saved_memories.append(
                            {
                                "type": memory_record.type.value,
                                "content": memory_record.content,
                                "importance": memory_record.importance,
                                "confidence": extraction["confidence"],
                            }
                        )
                        logger.info(
                            f"Saved memory (LLM extracted): {memory_record.type.value} - "
                            f"{memory_record.content[:50]}..."
                        )
                    except Exception as e:
                        logger.error(f"Failed to save memory: {e}")

        session_manager = get_session_manager()
        session_manager.update_activity(session_id, user_id)

        if result.triggered and result.trigger_type == TriggerType.SESSION_END:
            session_manager.end_session(session_id)

    except ImportError as e:
        logger.warning(f"Memory modules not available: {e}")
    except Exception as e:
        logger.error(f"Memory save node error: {e}")

    decision = "memory_save: triggered=" + str(len(trigger_results) > 0)
    if trigger_results:
        decision += f", saved={len(saved_memories)}"

    return {
        "saved_memories": saved_memories,
        "trigger_results": trigger_results,
        "decision_path": [decision],
    }
