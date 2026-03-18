"""Batch query runner for Agentic RAG Assistant.

Reads queries from a JSON file and runs them through the agent with thread pool concurrency.

Usage:
    python scripts/batch_query.py --input queries.json --output results.json
    python scripts/batch_query.py --input queries.json --concurrency 5
    python scripts/batch_query.py --input queries.json --concurrency 10 --verbose

Input format (queries.json):
    ["问题1", "问题2", "问题3"]

Output format (results.json):
    {
        "results": [
            {
                "query": "问题1",
                "response": "...",
                "contexts": [...],
                "total_chunks": 5,
                "trace_id": "...",
                "error": null
            }
        ],
        "summary": {
            "total": 10,
            "success": 9,
            "failed": 1,
            "elapsed_seconds": 45.2
        }
    }
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_queries(input_path: Path) -> List[str]:
    """Load queries from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if all(isinstance(q, str) for q in data):
            return data
        elif all(isinstance(q, dict) and "query" in q for q in data):
            return [q["query"] for q in data]

    raise ValueError(
        f"Invalid format in {input_path}. Expected array of strings or objects with 'query' field."
    )


def process_query(
    query: str,
    query_id: int,
    total: int,
    settings: Any,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Process a single query (runs in thread pool)."""
    from dotenv import load_dotenv

    load_dotenv()

    from src.agent.graph import KnowledgeAssistant

    assistant = KnowledgeAssistant(settings)

    print(f"[{query_id}/{total}] Processing: {query[:50]}{'...' if len(query) > 50 else ''}")

    try:
        result = assistant.ask(query)

        output = {
            "query": result.get("query", query),
            "response": result.get("response"),
            "contexts": [],
            "total_chunks": result.get("total_chunks", 0),
            "trace_id": result.get("trace_id"),
            "trace_url": assistant._tracer.get_trace_url() if assistant._tracer else None,
            "error": result.get("error"),
        }

        if verbose:
            resp_preview = (result.get("response") or "")[:80]
            print(f"[{query_id}/{total}] Done - Response: {resp_preview}...")
            if assistant._tracer:
                print(f"[{query_id}/{total}] Trace: {assistant._tracer.get_trace_url()}")

        return output

    except Exception as e:
        print(f"[{query_id}/{total}] Error: {e}")
        return {
            "query": query,
            "response": None,
            "contexts": [],
            "total_chunks": 0,
            "trace_id": None,
            "trace_url": None,
            "error": str(e),
        }


def run_batch(
    queries: List[str],
    output_path: Path,
    concurrency: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run batch queries with thread pool concurrency."""
    from dotenv import load_dotenv

    load_dotenv()

    from src.core.config import load_settings

    settings = load_settings()

    results: List[Optional[Dict[str, Any]]] = [None] * len(queries)

    start_time = time.monotonic()
    total = len(queries)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {}
        for i, query in enumerate(queries):
            future = executor.submit(process_query, query, i + 1, total, settings, verbose)
            future_to_idx[future] = i

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[{idx + 1}/{total}] Unexpected error: {e}")
                results[idx] = {
                    "query": queries[idx],
                    "response": None,
                    "contexts": [],
                    "total_chunks": 0,
                    "trace_id": None,
                    "trace_url": None,
                    "error": str(e),
                }

    elapsed = time.monotonic() - start_time

    success_count = sum(1 for r in results if r and not r.get("error"))
    fail_count = sum(1 for r in results if r and r.get("error"))

    output = {
        "results": results,
        "summary": {
            "total": len(queries),
            "success": success_count,
            "failed": fail_count,
            "elapsed_seconds": round(elapsed, 2),
            "concurrency": concurrency,
            "timestamp": datetime.now().isoformat(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print(f"Completed {len(queries)} queries in {elapsed:.1f}s (concurrency={concurrency})")
    print(f"  Success: {success_count}, Failed: {fail_count}")
    print(f"  Results saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Batch query runner for Agentic RAG Assistant")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input JSON file with queries",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file (default: batch_results_{timestamp}.json)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=5,
        help="Max concurrent requests (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("eval_data/batch_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"batch_results_{timestamp}.json"

    print(f"Loading queries from: {input_path}")
    queries = load_queries(input_path)
    print(f"Found {len(queries)} queries")
    print(f"Concurrency: {args.concurrency}")
    print()

    run_batch(queries, output_path, concurrency=args.concurrency, verbose=args.verbose)


if __name__ == "__main__":
    main()
