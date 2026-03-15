"""Quick test script for the agent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agent.graph import run_agent

if __name__ == "__main__":
    print("Testing Agentic RAG Agent...")
    print("-" * 60)

    result = run_agent("什么是机器学习?")

    print(f"\n问题: {result.get('query', '')}")
    print(f"\n回答:\n{result.get('response', '')[:500]}")
    print(f"\n决策路径: {result.get('decision_path', [])}")
    print(f"\n引用数量: {len(result.get('citations', []))}")
    print("-" * 60)
    print("Test completed successfully!")
