"""RAG Tools wrapper for MCP Client.

Provides high-level tool abstractions for Agent use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from src.core.config import RAGServerConfig
from src.core.types import Chunk
from src.mcp_client.client import HTTPMCPClient, MockRAGMCPClient

logger = logging.getLogger(__name__)


@dataclass
class ToolCallResult:
    """Result from a tool call."""

    success: bool
    data: Any
    error: Optional[str] = None


@dataclass
class SearchResult:
    """Result from a search tool call."""

    success: bool
    chunks: List[Chunk]
    error: Optional[str] = None


@dataclass
class MemoryResult:
    """Result from memory tool call."""

    success: bool
    memories: List[Dict[str, Any]]
    error: Optional[str] = None


class RAGTools:
    """High-level RAG tools for Agent use.

    This class wraps the MCP client and provides a simpler interface
    for the Agent to call RAG tools.

    Example:
        >>> tools = RAGTools(config)
        >>> result = await tools.search("What is machine learning?")
        >>> print(result.data.chunks)
    """

    def __init__(self, config: RAGServerConfig, use_mock: bool = False):
        """Initialize RAG tools.

        Args:
            config: RAG server configuration.
            use_mock: If True, use mock client for testing.
        """
        self.config = config
        self.use_mock = use_mock
        self._client: Optional[Union[HTTPMCPClient, MockRAGMCPClient]] = None

    async def _get_client(self) -> Union[HTTPMCPClient, MockRAGMCPClient]:
        """Get or create the MCP client."""
        if self._client is None:
            if self.use_mock:
                self._client = MockRAGMCPClient(self.config)
            else:
                self._client = HTTPMCPClient(self.config)
            await self._client.connect()
        return self._client

    async def search(
        self, query: str, collection: Optional[str] = None, top_k: int = 10
    ) -> ToolCallResult:
        """Search the knowledge hub.

        Args:
            query: Search query.
            collection: Collection to search.
            top_k: Number of results.

        Returns:
            ToolCallResult with RetrievalResult.
        """
        try:
            client = await self._get_client()
            result = await client.query_knowledge_hub(
                query=query, collection=collection, top_k=top_k
            )
            return ToolCallResult(success=True, data=result)
        except Exception as e:
            return ToolCallResult(success=False, data=None, error=str(e))

    async def list_collections(self) -> ToolCallResult:
        """List available collections.

        Returns:
            ToolCallResult with list of collection names.
        """
        try:
            client = await self._get_client()
            collections = await client.list_collections()
            return ToolCallResult(success=True, data=collections)
        except Exception as e:
            return ToolCallResult(success=False, data=[], error=str(e))

    async def get_document(self, doc_id: str, collection: Optional[str] = None) -> ToolCallResult:
        """Get document by ID.

        Args:
            doc_id: Document ID.
            collection: Collection name.

        Returns:
            ToolCallResult with document summary.
        """
        try:
            client = await self._get_client()
            summary = await client.get_document_summary(doc_id=doc_id, collection=collection)
            return ToolCallResult(
                success=summary is not None,
                data=summary,
                error=None if summary else "Document not found",
            )
        except Exception as e:
            return ToolCallResult(success=False, data=None, error=str(e))

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> "RAGTools":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()


class MemoryTools:
    """High-level Memory tools for Agent use.

    This class wraps the MCP client and provides a simpler interface
    for the Agent to call Memory tools.
    """

    def __init__(self, config: RAGServerConfig, use_mock: bool = False):
        """Initialize Memory tools.

        Args:
            config: RAG server configuration.
            use_mock: If True, use mock client for testing.
        """
        self.config = config
        self.use_mock = use_mock
        self._client: Optional[Union[HTTPMCPClient, MockRAGMCPClient]] = None

    async def _get_client(self) -> Union[HTTPMCPClient, MockRAGMCPClient]:
        """Get or create the MCP client."""
        if self._client is None:
            if self.use_mock:
                self._client = MockRAGMCPClient(self.config)
            else:
                self._client = HTTPMCPClient(self.config)
            await self._client.connect()
        return self._client

    async def recall_memory(
        self,
        session_id: str,
        query: str,
        memory_type: str = "short_term",
        limit: int = 5,
    ) -> MemoryResult:
        """Recall memories from the memory system.

        Args:
            session_id: Session ID for short-term memory.
            query: Query for semantic search (long-term memory).
            memory_type: Type of memory ("short_term" or "long_term").
            limit: Maximum number of memories to recall.

        Returns:
            MemoryResult with memories or error.
        """
        try:
            client = await self._get_client()

            if memory_type == "short_term":
                result = await client.call_tool(
                    "recall_memory",
                    {
                        "session_id": session_id,
                        "type": "conversation",
                        "limit": limit,
                    },
                )
            else:
                result = await client.call_tool(
                    "recall_memory",
                    {
                        "query": query,
                        "memory_scope": "long_term",
                        "type": "conversation",
                        "limit": limit,
                    },
                )

            if result.get("isError", False):
                error_msg = "Unknown error"
                content = result.get("content", [])
                if content and isinstance(content[0], dict):
                    error_msg = content[0].get("text", error_msg)
                return MemoryResult(success=False, memories=[], error=error_msg)

            memories = self._parse_memories(result)
            return MemoryResult(success=True, memories=memories)

        except Exception as e:
            logger.error(f"Error recalling memory: {e}")
            return MemoryResult(success=False, memories=[], error=str(e))

    async def save_memory(
        self,
        session_id: str,
        role: str,
        content: str,
        memory_type: str = "short_term",
    ) -> bool:
        """Save a memory to the memory system.

        Args:
            session_id: Session ID.
            role: Role (user/assistant).
            content: Memory content.
            memory_type: Type of memory ("short_term" or "long_term").

        Returns:
            True if successful, False otherwise.
        """
        try:
            client = await self._get_client()

            result = await client.call_tool(
                "save_memory",
                {
                    "session_id": session_id,
                    "memories": [{"role": role, "content": content, "type": "conversation"}],
                    "collection": ("memory" if memory_type == "short_term" else "long_term_memory"),
                },
            )

            return not result.get("isError", False)

        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            return False

    def _parse_memories(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse memories from MCP result.

        Args:
            result: MCP call_tool result.

        Returns:
            List of memory dictionaries.
        """
        memories = []
        content = result.get("content", [])

        if not content or not isinstance(content[0], dict):
            return memories

        text = content[0].get("text", "")
        if not text or "找到" not in text:
            return memories

        memory = {}
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("###"):
                if memory.get("content"):
                    memories.append(memory)
                memory = {"id": "", "role": "", "content": ""}
                parts = line.split()
                if len(parts) >= 3:
                    memory["id"] = parts[2]
            elif line.startswith("**角色:**"):
                memory["role"] = line.replace("**角色:**", "").strip()
            elif line.startswith("**内容:**"):
                memory["content"] = line.replace("**内容:**", "").strip()

        if memory.get("content"):
            memories.append(memory)

        return memories

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> "MemoryTools":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()


def format_chunks_for_context(chunks: List[Chunk], max_length: int = 4000) -> str:
    """Format chunks into context string for LLM.

    Args:
        chunks: List of chunks.
        max_length: Maximum total length.

    Returns:
        Formatted context string.
    """
    if not chunks:
        return "未找到相关信息。"

    context_parts = []
    total_length = 0

    for i, chunk in enumerate(chunks, 1):
        chunk_text = f"[文档{i}]\n{chunk.text}\n"

        if total_length + len(chunk_text) > max_length:
            break

        context_parts.append(chunk_text)
        total_length += len(chunk_text)

    return "\n".join(context_parts)


def format_chunks_with_citations(chunks: List[Chunk]) -> str:
    """Format chunks with citation markers.

    Args:
        chunks: List of chunks.

    Returns:
        Formatted string with citation numbers.
    """
    if not chunks:
        return ""

    lines = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source_path", "未知来源")
        page = chunk.metadata.get("page_num", "")
        page_info = f", 第{page}页" if page else ""
        lines.append(f"[{i}] {source}{page_info}")

    return "\n".join(lines)


def format_memories_for_context(memories: List[Dict[str, Any]]) -> str:
    """Format memories into context string for LLM.

    Args:
        memories: List of memory dictionaries.

    Returns:
        Formatted context string.
    """
    if not memories:
        return ""

    parts = ["## 历史对话上下文\n"]
    for memory in memories:
        role = memory.get("role", "unknown")
        content = memory.get("content", "")
        role_label = "用户" if role == "user" else "助手" if role == "assistant" else role
        parts.append(f"**{role_label}:** {content}\n")

    return "\n".join(parts)
