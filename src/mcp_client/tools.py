"""RAG Tools wrapper for MCP Client.

Provides high-level tool abstractions for Agent use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union

from src.core.config import RAGServerConfig
from src.core.types import Chunk
from src.mcp_client.client import HTTPMCPClient, MockRAGMCPClient


@dataclass
class ToolCallResult:
    """Result from a tool call."""

    success: bool
    data: Any
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
