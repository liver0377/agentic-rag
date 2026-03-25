#!/usr/bin/env python
"""Agent Ablation Experiment Runner.

Executes agent queries with different configurations to measure
the contribution of each module (decomposition, rewrite, memory).

Usage:
    python scripts/run_agent_ablation.py --config baseline
    python scripts/run_agent_ablation.py --config decompose
    python scripts/run_agent_ablation.py --config rewrite
    python scripts/run_agent_ablation.py --config full
    python scripts/run_agent_ablation.py --config full --output eval_data/eval_results_agent_full.json

Configurations:
    - baseline:   No decomposition, no rewrite, no memory
    - decompose:  Enable decomposition only
    - rewrite:    Enable decomposition + rewrite
    - full:       Enable all features (decomposition + rewrite + memory)

Output format (eval_results_agent_{config}.json):
    {
        "config": "full",
        "timestamp": "2026-03-25T12:00:00",
        "total_queries": 128,
        "successful": 128,
        "failed": 0,
        "avg_latency_ms": 1800,
        "results": [
            {
                "query": "...",
                "ground_truth": "...",
                "answer": "...",
                "contexts": [...],
                "sub_queries": [...],
                "used_memory": false,
                "latency_ms": 2450,
                "tokens": 1200,
                "success": true,
                "group": "complex_decompose",
                "difficulty": "complex",
                ...
            }
        ]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_MAP = {
    "baseline": {
        "enable_decomposition": False,
        "enable_rewrite": False,
        "enable_memory": False,
    },
    "decompose": {
        "enable_decomposition": True,
        "enable_rewrite": False,
        "enable_memory": False,
    },
    "rewrite": {
        "enable_decomposition": True,
        "enable_rewrite": True,
        "enable_memory": False,
    },
    "full": {
        "enable_decomposition": True,
        "enable_rewrite": True,
        "enable_memory": True,
    },
}


def load_test_set(input_path: Path) -> List[Dict[str, Any]]:
    """Load test set from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {input_path}")

    return data


def process_single_query(
    test_case: Dict[str, Any],
    config_name: str,
    config: Dict[str, Any],
    query_id: int,
    total: int,
    memory_cache: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a single query with the agent."""
    from dotenv import load_dotenv

    load_dotenv()

    from src.agent.graph import KnowledgeAssistant
    from src.core.config import load_settings

    query = test_case.get("query", "")
    group = test_case.get("group", "unknown")
    scenario_id = test_case.get("scenario_id")
    turn = test_case.get("turn", 1)

    print(
        f"[{query_id}/{total}] [{group}] Processing: {query[:50]}{'...' if len(query) > 50 else ''}"
    )

    settings = load_settings()
    if collection:
        settings.rag_server.collection = collection

    assistant = KnowledgeAssistant(
        enable_decomposition=config.get("enable_decomposition", False),
        enable_rewrite=config.get("enable_rewrite", False),
        use_llm=True,
        settings=settings,
    )

    start_time = time.monotonic()
    result = {"query": query, "success": False, "error": None}
    used_memory = False

    try:
        agent_result = assistant.ask(query)

        answer = agent_result.get("response", "")
        sub_queries = agent_result.get("sub_queries", [])
        contexts_raw = agent_result.get("contexts", [])
        total_chunks = agent_result.get("total_chunks", 0)
        error = agent_result.get("error")

        contexts = []
        if contexts_raw:
            for c in contexts_raw:
                if isinstance(c, dict):
                    contexts.append(c.get("text", str(c)))
                else:
                    contexts.append(str(c))

        if config.get("enable_memory") and memory_cache is not None and scenario_id:
            cache_key = f"scenario_{scenario_id}"
            if cache_key in memory_cache:
                used_memory = True

        result = {
            "query": query,
            "ground_truth": test_case.get("ground_truth", ""),
            "answer": answer,
            "contexts": contexts,
            "sub_queries": sub_queries,
            "total_chunks": total_chunks,
            "used_memory": used_memory,
            "success": error is None,
            "error": error,
            "group": group,
            "difficulty": test_case.get("difficulty", "medium"),
            "source_file": test_case.get("source_file", ""),
            "expected_keywords": test_case.get("expected_keywords", []),
            "expected_sub_questions": test_case.get("expected_sub_questions", []),
            "memory_type": test_case.get("memory_type"),
            "needs_context": test_case.get("needs_context", False),
            "refer_to_previous": test_case.get("refer_to_previous"),
            "scenario_id": scenario_id,
            "turn": turn,
        }

        if config.get("enable_memory") and memory_cache is not None and scenario_id:
            cache_key = f"scenario_{scenario_id}"
            memory_cache[cache_key] = {
                "query": query,
                "answer": answer,
                "contexts": contexts,
            }

    except Exception as e:
        logger.error(f"Error processing query '{query[:50]}': {e}")
        result = {
            "query": query,
            "ground_truth": test_case.get("ground_truth", ""),
            "answer": "",
            "contexts": [],
            "sub_queries": [],
            "total_chunks": 0,
            "used_memory": False,
            "success": False,
            "error": str(e),
            "group": group,
            "difficulty": test_case.get("difficulty", "medium"),
            "source_file": test_case.get("source_file", ""),
            "expected_keywords": test_case.get("expected_keywords", []),
            "expected_sub_questions": test_case.get("expected_sub_questions", []),
            "memory_type": test_case.get("memory_type"),
            "needs_context": test_case.get("needs_context", False),
            "refer_to_previous": test_case.get("refer_to_previous"),
            "scenario_id": scenario_id,
            "turn": turn,
        }

    latency_ms = (time.monotonic() - start_time) * 1000
    result["latency_ms"] = round(latency_ms, 2)
    result["tokens"] = estimate_tokens(
        result.get("answer", "") + " ".join(result.get("contexts", []))
    )

    print(f"[{query_id}/{total}] Done - latency: {latency_ms:.0f}ms, success: {result['success']}")
    return result


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation)."""
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + other_chars / 4)


def run_ablation(
    test_set: List[Dict[str, Any]],
    config_name: str,
    config: Dict[str, Any],
    concurrency: int = 3,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ablation experiment with the given configuration."""
    memory_cache: Dict[str, Any] = {}

    if config.get("enable_memory"):
        test_set_sorted = sorted(
            test_set, key=lambda x: (x.get("scenario_id", 0), x.get("turn", 1))
        )
    else:
        test_set_sorted = test_set

    results: List[Optional[Dict[str, Any]]] = [None] * len(test_set_sorted)
    start_time = time.monotonic()
    total = len(test_set_sorted)

    if concurrency > 1 and not config.get("enable_memory"):
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {}
            for i, test_case in enumerate(test_set_sorted):
                future = executor.submit(
                    process_single_query,
                    test_case,
                    config_name,
                    config,
                    i + 1,
                    total,
                    None,
                    collection,
                )
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Unexpected error for query {idx}: {e}")
                    results[idx] = {
                        "query": test_set_sorted[idx].get("query", ""),
                        "success": False,
                        "error": str(e),
                    }
    else:
        for i, test_case in enumerate(test_set_sorted):
            results[i] = process_single_query(
                test_case,
                config_name,
                config,
                i + 1,
                total,
                memory_cache if config.get("enable_memory") else None,
                collection,
            )

    elapsed = time.monotonic() - start_time

    valid_results = [r for r in results if r is not None]
    success_count = sum(1 for r in valid_results if r.get("success"))
    fail_count = sum(1 for r in valid_results if not r.get("success"))

    latencies = [r.get("latency_ms", 0) for r in valid_results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    output = {
        "config": config_name,
        "configuration": config,
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(test_set_sorted),
        "successful": success_count,
        "failed": fail_count,
        "avg_latency_ms": round(avg_latency, 2),
        "elapsed_seconds": round(elapsed, 2),
        "results": valid_results,
    }

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Agent Ablation Experiments")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        choices=list(CONFIG_MAP.keys()),
        help="Configuration name (baseline, decompose, rewrite, full)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="eval_data/raw/test_set_raw.json",
        help="Input test set JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output results JSON file",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent requests (default: 3, ignored for memory-enabled configs)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="eval_full",
        help="RAG collection to query (default: eval_full)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = CONFIG_MAP[args.config]

    print(f"Configuration: {args.config}")
    print(f"  - enable_decomposition: {config.get('enable_decomposition', False)}")
    print(f"  - enable_rewrite: {config.get('enable_rewrite', False)}")
    print(f"  - enable_memory: {config.get('enable_memory', False)}")
    print(f"  - collection: {args.collection}")
    print()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "eval_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eval_results_agent_{args.config}.json"

    print(f"Loading test set from: {input_path}")
    test_set = load_test_set(input_path)
    print(f"Found {len(test_set)} test cases")
    print()

    result = run_ablation(
        test_set, args.config, config, concurrency=args.concurrency, collection=args.collection
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print(f"Completed {result['total_queries']} queries in {result['elapsed_seconds']:.1f}s")
    print(f"  Success: {result['successful']}, Failed: {result['failed']}")
    print(f"  Avg latency: {result['avg_latency_ms']:.0f}ms")
    print(f"  Results saved to: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
