"""Prompt templates for Agent nodes."""

ANALYZE_PROMPT = """你是一个查询分析专家。分析用户的问题，判断：
1. 问题的复杂程度（简单/复杂）
2. 问题类型（事实性/程序性/分析性/比较性）
3. 是否需要拆分为子问题

用户问题：{query}

请给出你的分析结果。"""


DECOMPOSE_PROMPT = """你是一个问题拆分专家。将用户的复杂问题拆分为多个简单的子问题，每个子问题应该：
1. 独立可回答
2. 保持原问题的核心意图
3. 合起来能完整回答原问题

用户问题：{query}

请拆分为子问题（每行一个）："""


EVALUATE_PROMPT = """你是一个检索结果评估专家。评估检索到的文档是否足够回答用户的问题。

用户问题：{query}

检索到的文档：
{context}

请评估：
1. 文档是否与问题相关？
2. 文档内容是否足够回答问题？
3. 如果不够，缺少什么信息？

给出评估结果（充分/不充分）和原因："""


REWRITE_PROMPT = """你是一个查询优化专家。根据评估结果，改写用户的查询以获得更好的检索结果。

原问题：{query}
评估原因：{reason}

请改写查询（保持原意，使用不同的表达方式）："""


GENERATE_PROMPT = """你是一个知识助手。基于检索到的文档回答用户的问题。

要求：
1. 回答要准确、完整
2. 引用相关文档（使用[文档X]格式）
3. 如果文档中没有相关信息，请明确说明

用户问题：{query}

检索到的文档：
{context}

请给出回答："""


def get_prompt(name: str) -> str:
    """Get prompt template by name.

    Args:
        name: Prompt name (analyze, decompose, evaluate, rewrite, generate).

    Returns:
        Prompt template string.
    """
    prompts = {
        "analyze": ANALYZE_PROMPT,
        "decompose": DECOMPOSE_PROMPT,
        "evaluate": EVALUATE_PROMPT,
        "rewrite": REWRITE_PROMPT,
        "generate": GENERATE_PROMPT,
    }
    return prompts.get(name, "")
