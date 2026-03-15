"""MCP Client for connecting to RAG MCP Server.

This module provides a client for the Model Context Protocol (MCP)
to communicate with the RAG MCP Server.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import RAGServerConfig
from src.core.types import Chunk, RetrievalResult
from src.core.utils import Timer, generate_trace_id


@dataclass
class MCPToolResult:
    """Result from MCP tool call."""

    content: Any
    is_error: bool = False
    error_message: Optional[str] = None


class RAGMCPClient:
    """MCP Client for RAG Server.

    This client communicates with the RAG MCP Server using stdio transport.
    It provides methods to call the RAG server's tools for knowledge retrieval.

    Example:
        >>> config = RAGServerConfig(command=["python", "-m", "src.mcp_server.server"])
        >>> client = RAGMCPClient(config)
        >>> async with client:
        ...     result = await client.query_knowledge_hub("What is RAG?")
    """

    def __init__(self, config: RAGServerConfig):
        """Initialize the MCP client.

        Args:
            config: RAG server configuration.
        """
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._initialized = False

    async def connect(self) -> None:
        """Connect to the RAG MCP Server."""
        if self._process is not None:
            return

        working_dir = self.config.working_dir
        if working_dir:
            cwd = Path(working_dir).resolve()
        else:
            cwd = Path.cwd()

        self._process = subprocess.Popen(
            self.config.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
            text=True,
            bufsize=1,
        )

        await self._initialize()

    async def _initialize(self) -> None:
        """Initialize MCP connection."""
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agentic-rag-assistant", "version": "0.1.0"},
            },
        }

        await self._send_request(init_request)
        self._initialized = True

    def _next_request_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the MCP server.

        Args:
            request: The JSON-RPC request.

        Returns:
            The response from the server.
        """
        if self._process is None:
            raise RuntimeError("Not connected to MCP server")

        request_json = json.dumps(request) + "\n"
        self._process.stdin.write(request_json)
        self._process.stdin.flush()

        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from MCP server")

        return json.loads(response_line)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments for the tool.

        Returns:
            Result from the tool call.
        """
        if not self._initialized:
            await self.connect()

        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        try:
            response = await self._send_request(request)

            if "error" in response:
                return MCPToolResult(
                    content=None,
                    is_error=True,
                    error_message=response["error"].get("message", "Unknown error"),
                )

            result = response.get("result", {})
            content = result.get("content", [])

            if isinstance(content, list) and len(content) > 0:
                text_content = content[0].get("text", "")
                try:
                    parsed_content = json.loads(text_content)
                    return MCPToolResult(content=parsed_content)
                except json.JSONDecodeError:
                    return MCPToolResult(content=text_content)

            return MCPToolResult(content=content)

        except Exception as e:
            return MCPToolResult(content=None, is_error=True, error_message=str(e))

    async def query_knowledge_hub(
        self, query: str, collection: Optional[str] = None, top_k: int = 10
    ) -> RetrievalResult:
        """Query the knowledge hub for relevant chunks.

        Args:
            query: The query string.
            collection: Collection to search (default from config).
            top_k: Number of results to return.

        Returns:
            RetrievalResult with chunks.
        """
        with Timer(f"query_knowledge_hub: {query[:50]}...") as timer:
            result = await self.call_tool(
                "query_knowledge_hub",
                {
                    "query": query,
                    "collection": collection or self.config.collection,
                    "top_k": top_k,
                },
            )

        if result.is_error:
            return RetrievalResult(
                chunks=[],
                collection=collection or self.config.collection,
                query=query,
                total_count=0,
            )

        content = result.content
        if isinstance(content, dict):
            chunks_data = content.get("chunks", [])
            chunks = [Chunk.from_dict(c) for c in chunks_data]
            return RetrievalResult(
                chunks=chunks,
                collection=content.get("collection", self.config.collection),
                query=query,
                total_count=len(chunks),
            )

        return RetrievalResult(
            chunks=[], collection=collection or self.config.collection, query=query, total_count=0
        )

    async def list_collections(self) -> List[str]:
        """List available collections.

        Returns:
            List of collection names.
        """
        result = await self.call_tool("list_collections", {})

        if result.is_error:
            return []

        content = result.content
        if isinstance(content, dict):
            return content.get("collections", [])
        elif isinstance(content, list):
            return content

        return []

    async def get_document_summary(
        self, doc_id: str, collection: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get document summary by ID.

        Args:
            doc_id: Document ID.
            collection: Collection name.

        Returns:
            Document summary or None if not found.
        """
        result = await self.call_tool(
            "get_document_summary",
            {"doc_id": doc_id, "collection": collection or self.config.collection},
        )

        if result.is_error:
            return None

        return result.content

    async def close(self) -> None:
        """Close the connection to the MCP server."""
        if self._process:
            self._process.terminate()
            self._process.wait()
            self._process = None
            self._initialized = False

    async def __aenter__(self) -> "RAGMCPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.close()


class MockRAGMCPClient:
    """Mock MCP Client for testing without RAG server.

    This client returns mock data for testing purposes.
    """

    def __init__(self, config: RAGServerConfig):
        self.config = config

    async def connect(self) -> None:
        pass

    async def query_knowledge_hub(
        self, query: str, collection: Optional[str] = None, top_k: int = 10
    ) -> RetrievalResult:
        """Return mock retrieval result."""
        mock_chunks = [
            Chunk(
                id=f"mock_chunk_{i}",
                text=f"这是关于 '{query}' 的模拟检索结果 {i + 1}。",
                score=0.9 - i * 0.1,
                metadata={
                    "source_path": f"doc_{i}.pdf",
                    "page_num": i + 1,
                },
            )
            for i in range(min(top_k, 3))
        ]

        return RetrievalResult(
            chunks=mock_chunks,
            collection=collection or self.config.collection,
            query=query,
            total_count=len(mock_chunks),
        )

    async def list_collections(self) -> List[str]:
        return ["knowledge_hub", "test_collection"]

    async def get_document_summary(
        self, doc_id: str, collection: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        return {"doc_id": doc_id, "title": f"Document {doc_id}", "chunk_count": 10}

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "MockRAGMCPClient":
        return self

    async def __aexit__(self, *args) -> None:
        pass
