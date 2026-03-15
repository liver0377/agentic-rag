"""CLI script to start the Agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import KnowledgeAssistant
from src.core.config import load_settings


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Agentic RAG Assistant CLI")
    parser.add_argument("query", nargs="?", help="Query to ask the assistant")
    parser.add_argument("--config", "-c", default="config/settings.yaml", help="Config file path")
    parser.add_argument("--no-decompose", action="store_true", help="Disable query decomposition")
    parser.add_argument("--no-rewrite", action="store_true", help="Disable query rewriting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    settings = load_settings(args.config)

    assistant = KnowledgeAssistant(
        enable_decomposition=not args.no_decompose,
        enable_rewrite=not args.no_rewrite,
        max_rewrite_attempts=settings.agent.max_rewrite_attempts,
    )

    if args.query:
        print(f"\n问题: {args.query}")
        print("-" * 60)

        result = assistant.ask(args.query)

        print(f"\n回答:\n{result.get('response', '无回答')}")

        if args.verbose:
            print("\n" + "=" * 60)
            print("详细信息:")

            sub_queries = result.get("sub_queries", [])
            if sub_queries:
                print(f"  子问题: {sub_queries}")

            citations = result.get("citations", [])
            if citations:
                print(f"  引用数量: {len(citations)}")

            decision_path = result.get("decision_path", [])
            if decision_path:
                print(f"  决策路径: {decision_path}")
    else:
        print("Agentic RAG Assistant - 交互模式")
        print("输入 'quit' 或 'exit' 退出")
        print("-" * 60)

        while True:
            try:
                query = input("\n请输入问题: ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    print("再见!")
                    break

                if not query:
                    continue

                result = assistant.ask(query)

                print(f"\n回答:\n{result.get('response', '无回答')}")

                citations = result.get("citations", [])
                if citations:
                    print(f"\n引用: {len(citations)} 个来源")

            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                print(f"\n错误: {e}")


if __name__ == "__main__":
    main()
