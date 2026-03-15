"""Response Generator Node.

Generates the final response with citations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agent.state import AgentState
from src.core.types import Chunk, Citation


def format_chunks_for_generation(chunks: List[Chunk], max_length: int = 4000) -> str:
    """Format chunks for LLM generation.

    Args:
        chunks: Retrieved chunks.
        max_length: Maximum context length.

    Returns:
        Formatted context string.
    """
    if not chunks:
        return "未找到相关文档。"

    context_parts = []
    current_length = 0

    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source_path", "未知来源")
        page = chunk.metadata.get("page_num", "")
        page_info = f", 第{page}页" if page else ""

        chunk_text = f"[文档{i}] (来源: {source}{page_info})\n{chunk.text}\n"

        if current_length + len(chunk_text) > max_length:
            break

        context_parts.append(chunk_text)
        current_length += len(chunk_text)

    return "\n".join(context_parts)


def generate_response_with_llm(query: str, context: str, llm_client: Any = None) -> str:
    """Generate response using LLM.

    Args:
        query: User query.
        context: Retrieved context.
        llm_client: LLM client (injected).

    Returns:
        Generated response.
    """
    if llm_client is None:
        return generate_mock_response(query, context)

    prompt = f"""基于以下检索到的文档内容，回答用户的问题。
请确保回答准确、完整，并在回答中引用相关文档（使用[文档X]格式）。

检索到的文档：
{context}

用户问题：{query}

请给出详细回答："""

    try:
        response = llm_client.chat(prompt)
        return response
    except Exception as e:
        return f"生成回答时出错：{e}"


def generate_mock_response(query: str, context: str) -> str:
    """Generate a mock response for testing.

    Args:
        query: User query.
        context: Retrieved context.

    Returns:
        Mock response.
    """
    return f"""根据检索到的文档，关于"{query}"，我找到了以下相关信息：

{context}

以上信息来自检索到的文档。如果您需要更详细的信息，可以查看原始文档。

**注意**：这是一个模拟响应，用于测试目的。"""


def extract_citations(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    """Extract citations from chunks.

    Args:
        chunks: Retrieved chunks.

    Returns:
        List of citation dictionaries.
    """
    citations = []
    for chunk in chunks[:5]:
        citation = Citation.from_chunk(chunk, snippet_length=150)
        citations.append(citation.to_dict())
    return citations


def generate_node(state: AgentState, llm_client: Any = None) -> Dict[str, Any]:
    """Generate the final response.

    This node:
    1. Formats retrieved chunks as context
    2. Generates response using LLM
    3. Extracts citations

    Args:
        state: Current agent state.
        llm_client: LLM client (injected).

    Returns:
        State updates with final response and citations.
    """
    query = state.original_query
    chunks = state.chunks

    context = format_chunks_for_generation(chunks)

    response = generate_response_with_llm(query, context, llm_client)

    citations = extract_citations(chunks)

    decision = f"generate: response length={len(response)}, citations={len(citations)}"

    return {
        "final_response": response,
        "citations": citations,
        "decision_path": [decision],
    }
