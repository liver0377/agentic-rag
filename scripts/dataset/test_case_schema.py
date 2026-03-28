"""Memory 测试用例数据模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryTestCase:
    """Memory 系统测试用例。

    用于评估长期记忆的召回和使用效果。

    Attributes:
        id: 测试用例 ID，如 "TC_ENHANCE_001"
        scenario: 场景类型 (enhance/cross_session/conflict/baseline)
        priority: 优先级 (P0/P1/P2)
        user_id: 用户 ID
        cross_session: 是否跨会话
        user_memories: 预置的用户记忆列表
        knowledge_context: 知识库上下文（可选）
        query: 用户问题
        expected_behavior: 期望行为描述
        should_use_memory: 是否应该使用记忆
        should_use_knowledge: 是否应该使用知识库
        memory_keywords: 期望输出中包含的记忆关键词
        knowledge_keywords: 期望输出中包含的知识关键词
        evaluation_criteria: 评估标准列表
    """

    id: str
    scenario: str
    priority: str
    user_id: str
    cross_session: bool
    user_memories: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_context: Optional[str] = None
    query: str = ""
    expected_behavior: str = ""
    should_use_memory: bool = True
    should_use_knowledge: bool = False
    memory_keywords: List[str] = field(default_factory=list)
    knowledge_keywords: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return asdict(self)
