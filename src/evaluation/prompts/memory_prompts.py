"""Prompt templates for Memory evaluation.

记忆系统评估专用的 Prompt 模板。
"""

from __future__ import annotations

MEMORY_HIT_EVALUATION_PROMPT = """你是一个记忆系统评估专家。

## 任务
判断召回的记忆内容是否能够支持回答当前问题。

## 当前问题
{query}

## 召回的记忆
{recalled_memories}

## 期望召回的内容特征
{memory_relevance_prompt}

## 评估标准

### 相关性等级
1. **relevant（相关）**: 召回的记忆完整包含回答问题所需的关键信息
   - 示例：问题问"它的目标读者"，召回内容包含"CKafka 的目标读者是架构师"
   
2. **partially_relevant（部分相关）**: 召回的记忆包含部分有用信息，但不够完整
   - 示例：问题问"它的目标读者"，召回内容包含"CKafka 是消息队列"但未提及目标读者
   
3. **not_relevant（不相关）**: 召回的记忆与当前问题无关
   - 示例：问题问"它的目标读者"，召回内容是关于"如何安装 MySQL"

### 置信度
- 0.9-1.0: 非常确定
- 0.7-0.9: 比较确定
- 0.5-0.7: 有一定把握
- < 0.5: 不太确定

## 输出格式（JSON）
```json
{{
    "is_hit": true或false,
    "relevance": "relevant"或"partially_relevant"或"not_relevant",
    "confidence": 0.0到1.0之间的数值,
    "matched_keywords": ["匹配到的关键词列表"],
    "reasoning": "简要说明判断依据（1-2句话）"
}}
```

请严格按照 JSON 格式输出，不要包含其他内容。"""


CONTEXT_ACCURACY_PROMPT = """你是一个多轮对话评估专家。

## 任务
判断回答是否正确理解了问题中的指代词（如"它"、"这个"、"上面提到的"等）。

## 问题
{query}

## Agent 回答
{answer}

## 期望引用的上下文
{refer_to_previous}

## 期望引用的关键术语
{expected_terms}

## 评估标准

1. **correct（正确）**: 回答正确理解了指代词，引用了正确的上下文信息
2. **incorrect（错误）**: 回答误解了指代词，引用了错误或无关的上下文
3. **unclear（不明确）**: 回答没有明确引用任何上下文，无法判断

## 输出格式（JSON）
```json
{{
    "is_correct": true或false,
    "accuracy": "correct"或"incorrect"或"unclear",
    "confidence": 0.0到1.0之间的数值,
    "found_terms": ["回答中包含的期望术语"],
    "reasoning": "简要说明判断依据"
}}
```

请严格按照 JSON 格式输出，不要包含其他内容。"""


MEMORY_QUALITY_PROMPT = """你是一个记忆系统质量评估专家。

## 任务
综合评估记忆系统的表现。

## 评估维度

1. **召回完整性**: 召回的记忆是否覆盖了回答问题所需的所有信息
2. **召回性相关性**: 召回的记忆是否与当前问题相关
3. **召回效率**: 召回的记忆数量是否合理（不太多也不太少）
4. **响应质量**: 基于召回记忆生成的回答质量如何

## 测试结果
{test_results}

## 输出格式（JSON）
```json
{{
    "overall_score": 0.0到1.0之间的数值,
    "dimensions": {{
        "completeness": 0.0到1.0,
        "relevance": 0.0到1.0,
        "efficiency": 0.0到1.0,
        "response_quality": 0.0到1.0
    }},
    "strengths": ["系统的优点"],
    "weaknesses": ["需要改进的地方"],
    "recommendations": ["改进建议"]
}}
```

请严格按照 JSON 格式输出，不要包含其他内容。"""


def get_memory_prompt(name: str) -> str:
    """Get memory evaluation prompt template by name.

    Args:
        name: Prompt name (memory_hit, context_accuracy, memory_quality).

    Returns:
        Prompt template string.
    """
    prompts = {
        "memory_hit": MEMORY_HIT_EVALUATION_PROMPT,
        "context_accuracy": CONTEXT_ACCURACY_PROMPT,
        "memory_quality": MEMORY_QUALITY_PROMPT,
    }
    return prompts.get(name, "")
