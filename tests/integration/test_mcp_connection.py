"""Test script to verify MCP connection to RAG Server."""

import asyncio

from src.core.config import load_settings
from src.mcp_client.client import HTTPMCPClient


async def test_connection():
    """Test connection to RAG MCP Server."""
    settings = load_settings()
    print(f"Connecting to RAG Server at: {settings.rag_server.url}")

    client = HTTPMCPClient(settings.rag_server)

    try:
        await client.connect()
        print("Connected successfully!")

        result = await client.query_knowledge_hub("测试查询", top_k=3)
        print(f"Query result: {result.total_count} chunks")

        for i, chunk in enumerate(result.chunks[:3], 1):
            print(f"\n[Chunk {i}] score={chunk.score:.2f}")
            print(f"Source: {chunk.metadata.get('source_path', 'unknown')}")
            print(f"Text: {chunk.text[:100]}...")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_connection())
