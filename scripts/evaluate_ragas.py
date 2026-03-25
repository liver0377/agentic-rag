#!/usr/bin/env python
"""Ragas evaluation CLI for Agentic RAG Assistant.

Runs Ragas evaluation against a test set and outputs metrics report.

Usage:
    # Evaluate with a test set
    python scripts/evaluate_ragas.py --test-set eval_data/annotated/test_set.json

    # Evaluate from ablation results (supports group statistics)
    python scripts/evaluate_ragas.py --ablation-input eval_data/eval_results_agent_full.json

    # Specify output directory
    python scripts/evaluate_ragas.py --test-set test_set.json --output-dir eval_reports/

    # Collect from LangFuse (last 7 days)
    python scripts/evaluate_ragas.py --collect-from-langfuse --days 7

    # JSON output
    python scripts/evaluate_ragas.py --test-set test_set.json --json

    # Evaluate specific metrics only
    python scripts/evaluate_ragas.py --test-set test_set.json --metrics faithfulness answer_relevancy

Exit codes:
    0 - Success
    1 - Evaluation failure
    2 - Configuration error
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ragas evaluation for Agentic RAG Assistant.")
    parser.add_argument(
        "--test-set",
        type=str,
        help="Path to test set JSON file.",
    )
    parser.add_argument(
        "--ablation-input",
        type=str,
        help="Path to ablation results JSON file (from run_agent_ablation.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_reports",
        help="Output directory for reports (default: eval_reports).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path (for ablation mode).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["faithfulness", "answer_relevancy", "context_precision"],
        help="Metrics to evaluate (default: all).",
    )
    parser.add_argument(
        "--collect-from-langfuse",
        action="store_true",
        help="Collect test cases from LangFuse traces.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days to look back when collecting from LangFuse (default: 7).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max traces to collect from LangFuse (default: 100).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output.",
    )
    return parser.parse_args()


def load_settings() -> Any:
    try:
        from src.core.config import load_settings

        return load_settings()
    except Exception as exc:
        logger.error("Failed to load settings: %s", exc)
        sys.exit(2)


def load_ablation_results(input_path: Path) -> tuple:
    """Load results from run_agent_ablation.py output."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config_name = data.get("config", "unknown")
    results = data.get("results", [])

    test_cases = []
    for r in results:
        answer = r.get("answer", "")
        contexts = r.get("contexts", [])

        if not answer or not contexts:
            continue

        test_cases.append(
            {
                "query": r.get("query", ""),
                "answer": answer,
                "contexts": contexts,
                "ground_truth": r.get("ground_truth", ""),
                "group": r.get("group", "unknown"),
                "difficulty": r.get("difficulty", "medium"),
            }
        )

    return test_cases, config_name


def calculate_metrics_by_group(
    results: List[Any],
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Calculate metrics grouped by test group."""
    groups: Dict[str, List[int]] = {}

    for i, tc in enumerate(test_cases):
        group = tc.get("group", "unknown")
        if group not in groups:
            groups[group] = []
        groups[group].append(i)

    by_group: Dict[str, Dict[str, Any]] = {}
    for group_name, indices in groups.items():
        group_results = [results[i] for i in indices if i < len(results)]

        if not group_results:
            continue

        metrics_sum: Dict[str, float] = {}
        valid_count = 0

        for r in group_results:
            if hasattr(r, "metrics") and r.metrics:
                valid_count += 1
                for k, v in r.metrics.items():
                    if k not in metrics_sum:
                        metrics_sum[k] = 0.0
                    metrics_sum[k] += float(v)

        if valid_count > 0:
            by_group[group_name] = {k: round(v / valid_count, 4) for k, v in metrics_sum.items()}
            by_group[group_name]["count"] = valid_count

    return by_group


def run_ablation_evaluation(args: argparse.Namespace, settings: Any) -> int:
    """Run evaluation from ablation results with group statistics."""
    input_path = Path(args.ablation_input)
    if not input_path.exists():
        print(f"Error: Ablation input not found: {input_path}", file=sys.stderr)
        return 2

    try:
        from src.evaluation.ragas_evaluator import RagasEvaluator, create_ragas_evaluator
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Install required packages: pip install ragas datasets", file=sys.stderr)
        return 2

    evaluator = create_ragas_evaluator(
        llm_config=settings.llm,
        metrics=args.metrics,
    )
    if evaluator is None:
        print("Error: Failed to create RagasEvaluator. Check LLM configuration.", file=sys.stderr)
        return 2

    print(f"Loading ablation results from: {input_path}")
    test_cases, config_name = load_ablation_results(input_path)

    if not test_cases:
        print("Error: No valid test cases found in ablation results.", file=sys.stderr)
        return 1

    print(f"\nRunning Ragas evaluation...")
    print(f"  Config: {config_name}")
    print(f"  Test cases: {len(test_cases)}")
    print(f"  Metrics: {args.metrics or ['faithfulness', 'answer_relevancy', 'context_precision']}")
    print()

    report = evaluator.evaluate_batch(test_cases)

    by_group = calculate_metrics_by_group(report.per_case_results, test_cases)

    output_data = {
        "config": config_name,
        "timestamp": datetime.now().isoformat(),
        "total_cases": report.total_cases,
        "aggregate_metrics": {k: round(v, 4) for k, v in report.aggregate_metrics.items()},
        "by_group": by_group,
        "per_case_results": [
            {
                "query": r.query,
                "group": test_cases[i].get("group", "unknown")
                if i < len(test_cases)
                else "unknown",
                "metrics": {k: round(v, 4) for k, v in r.metrics.items()},
                "error": r.error,
            }
            for i, r in enumerate(report.per_case_results)
        ],
        "total_elapsed_ms": round(report.total_elapsed_ms, 1),
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "eval_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"ragas_metrics_{config_name}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("  RAGAS EVALUATION REPORT (ABLATION MODE)")
    print("=" * 60)
    print(f"  Config: {config_name}")
    print(f"  Test Cases: {report.total_cases}")
    print(f"  Total Time: {report.total_elapsed_ms:.0f} ms")
    print()

    print("-" * 60)
    print("  AGGREGATE METRICS")
    print("-" * 60)
    for metric, value in sorted(report.aggregate_metrics.items()):
        bar = _render_bar(value)
        print(f"  {metric:<25s} {bar} {value:.4f}")
    print()

    if by_group:
        print("-" * 60)
        print("  METRICS BY GROUP")
        print("-" * 60)
        for group_name, metrics in sorted(by_group.items()):
            count = metrics.pop("count", 0)
            print(f"\n  [{group_name}] (n={count})")
            for metric, value in sorted(metrics.items()):
                print(f"    {metric:<23s} {value:.4f}")
        print()

    print("=" * 60)
    print(f"  Report saved to: {output_path}")
    print("=" * 60)

    return 0


def main() -> int:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    settings = load_settings()

    if args.ablation_input:
        return run_ablation_evaluation(args, settings)

    if not args.test_set and not args.collect_from_langfuse:
        print(
            "Error: Either --test-set, --ablation-input, or --collect-from-langfuse is required.",
            file=sys.stderr,
        )
        return 2

    try:
        from src.evaluation.ragas_evaluator import RagasEvaluator, create_ragas_evaluator
        from src.evaluation.data_collector import DataCollector, TestSet, create_data_collector
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Install required packages: pip install ragas datasets", file=sys.stderr)
        return 2

    evaluator = create_ragas_evaluator(
        llm_config=settings.llm,
        metrics=args.metrics,
    )
    if evaluator is None:
        print("Error: Failed to create RagasEvaluator. Check LLM configuration.", file=sys.stderr)
        return 2

    test_set: TestSet
    if args.collect_from_langfuse:
        if not settings.langfuse.enabled:
            print("Error: LangFuse is not enabled in configuration.", file=sys.stderr)
            return 2

        collector = create_data_collector(settings.langfuse)
        test_set = collector.collect_from_langfuse(days=args.days, limit=args.limit)

        if not test_set.test_cases:
            print("Warning: No test cases collected from LangFuse.", file=sys.stderr)
            return 0

        raw_path = (
            Path("eval_data/raw") / f"langfuse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        test_set.save(raw_path)
        print(f"Saved collected data to {raw_path}")

    else:
        test_path = Path(args.test_set)
        if not test_path.exists():
            print(f"Error: Test set not found: {test_path}", file=sys.stderr)
            return 2

        collector = DataCollector()
        test_set = collector.import_from_json(test_path)

    print(f"\nRunning Ragas evaluation...")
    print(f"  Test cases: {len(test_set.test_cases)}")
    print(f"  Metrics: {args.metrics or ['faithfulness', 'answer_relevancy', 'context_precision']}")
    print()

    test_cases = [tc.to_dict() for tc in test_set.test_cases]
    report = evaluator.evaluate_batch(test_cases)

    report.test_set_path = str(args.test_set) if args.test_set else "langfuse_collection"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")
    report_path = output_dir / timestamp / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    latest_path = output_dir / "latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_report(report, report_path)

    return 0


def _print_report(report: Any, report_path: Path) -> None:
    print("=" * 60)
    print("  RAGAS EVALUATION REPORT")
    print("=" * 60)
    print(f"  Evaluator: {report.evaluator}")
    print(f"  Test Set:  {report.test_set_path}")
    print(f"  Test Cases: {report.total_cases}")
    print(f"  Total Time: {report.total_elapsed_ms:.0f} ms")
    print()

    if report.pass_count + report.fail_count > 0:
        print("-" * 60)
        print("  PASS RATE SUMMARY")
        print("-" * 60)
        total_labeled = report.pass_count + report.fail_count
        pass_pct = (report.pass_count / total_labeled * 100) if total_labeled > 0 else 0
        print(f"  Pass: {report.pass_count} ({pass_pct:.1f}%)")
        print(f"  Fail: {report.fail_count}")
        print(f"  Unlabeled: {report.total_cases - total_labeled}")
        print()

    print("-" * 60)
    print("  AGGREGATE METRICS (RAGAS)")
    print("-" * 60)
    if report.aggregate_metrics:
        for metric, value in sorted(report.aggregate_metrics.items()):
            bar = _render_bar(value)
            print(f"  {metric:<25s} {bar} {value:.4f}")
    else:
        print("  (no metrics computed)")
    print()

    if report.pass_group_metrics or report.fail_group_metrics:
        print("-" * 60)
        print("  METRICS BY PASS/FAIL")
        print("-" * 60)
        if report.pass_group_metrics:
            print("  Pass Group:")
            for metric, value in sorted(report.pass_group_metrics.items()):
                print(f"    {metric:<23s} {value:.4f}")
        if report.fail_group_metrics:
            print("  Fail Group:")
            for metric, value in sorted(report.fail_group_metrics.items()):
                print(f"    {metric:<23s} {value:.4f}")
        print()

    print("-" * 60)
    print("  PER-CASE RESULTS")
    print("-" * 60)
    for i, result in enumerate(report.per_case_results[:10], 1):
        status = "✓" if not result.error else "✗"
        pass_label = f" [{result.pass_rate}]" if result.pass_rate else ""
        print(
            f"\n  [{i}] {status} {result.query[:50]}{'...' if len(result.query) > 50 else ''}{pass_label}"
        )
        if result.metrics:
            for metric, value in sorted(result.metrics.items()):
                print(f"      {metric}: {value:.4f}")
        if result.error:
            print(f"      Error: {result.error}")

    if len(report.per_case_results) > 10:
        print(f"\n  ... and {len(report.per_case_results) - 10} more cases")

    print()
    print("=" * 60)
    print(f"  Report saved to: {report_path}")
    print("=" * 60)


def _render_bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


if __name__ == "__main__":
    sys.exit(main())
