#!/usr/bin/env python
"""Agent Results Comparison Report Generator.

Aggregates evaluation results from all ablation configurations and
generates a comprehensive comparison report.

Usage:
    python scripts/compare_agent_results.py
    python scripts/compare_agent_results.py --output eval_data/eval_comparison_agent.json
    python scripts/compare_agent_results.py --baseline eval_data/eval_results_agent_baseline.json

Output format (eval_comparison_agent.json):
    {
        "timestamp": "2026-03-25T12:00:00",
        "configurations": {
            "baseline": {"decompose": false, "rewrite": false, "memory": false},
            "decompose": {"decompose": true, "rewrite": false, "memory": false},
            "rewrite": {"decompose": true, "rewrite": true, "memory": false},
            "full": {"decompose": true, "rewrite": true, "memory": true}
        },
        "overall_metrics": {
            "baseline": {"faithfulness": 0.68, "completeness": 0.58, ...},
            "full": {"faithfulness": 0.86, "completeness": 0.92, ...}
        },
        "by_group": {...},
        "improvements": {
            "baseline_to_full": {"faithfulness": "+26.5%", "completeness": "+58.6%"}
        }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
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

CONFIGURATIONS = {
    "baseline": {"decompose": False, "rewrite": False, "memory": False},
    "decompose": {"decompose": True, "rewrite": False, "memory": False},
    "rewrite": {"decompose": True, "rewrite": True, "memory": False},
    "full": {"decompose": True, "rewrite": True, "memory": True},
}

CONFIG_ORDER = ["baseline", "decompose", "rewrite", "full"]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file safely."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ablation_results(config_name: str, eval_data_dir: Path) -> Optional[Dict[str, Any]]:
    """Load ablation results for a configuration."""
    path = eval_data_dir / f"eval_results_agent_{config_name}.json"
    return load_json(path)


def load_ragas_metrics(config_name: str, eval_data_dir: Path) -> Optional[Dict[str, Any]]:
    """Load RAGAS metrics for a configuration."""
    path = eval_data_dir / f"ragas_metrics_{config_name}.json"
    return load_json(path)


def load_completeness_metrics(config_name: str, eval_data_dir: Path) -> Optional[Dict[str, Any]]:
    """Load completeness metrics for a configuration."""
    path = eval_data_dir / f"completeness_metrics_{config_name}.json"
    return load_json(path)


def load_memory_metrics(config_name: str, eval_data_dir: Path) -> Optional[Dict[str, Any]]:
    """Load memory metrics for a configuration."""
    path = eval_data_dir / f"memory_metrics_{config_name}.json"
    return load_json(path)


def calculate_latency_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate latency statistics from results."""
    if not results:
        return {"avg_latency_ms": 0, "p95_latency_ms": 0, "min_latency_ms": 0, "max_latency_ms": 0}

    latencies = sorted([r.get("latency_ms", 0) for r in results])

    avg_latency = sum(latencies) / len(latencies)
    p95_idx = int(len(latencies) * 0.95)
    p95_latency = latencies[min(p95_idx, len(latencies) - 1)]
    min_latency = latencies[0]
    max_latency = latencies[-1]

    return {
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "min_latency_ms": round(min_latency, 2),
        "max_latency_ms": round(max_latency, 2),
    }


def calculate_success_rate(results: List[Dict[str, Any]]) -> float:
    """Calculate success rate from results."""
    if not results:
        return 0.0
    success_count = sum(1 for r in results if r.get("success"))
    return round(success_count / len(results), 4)


def calculate_avg_tokens(results: List[Dict[str, Any]]) -> int:
    """Calculate average token count from results."""
    if not results:
        return 0
    tokens = [r.get("tokens", 0) for r in results]
    return int(sum(tokens) / len(tokens))


def aggregate_metrics_by_group(
    results: List[Dict[str, Any]],
    ragas_metrics: Optional[Dict[str, Any]],
    completeness_metrics: Optional[Dict[str, Any]],
    memory_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate metrics by test group."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in results:
        group = r.get("group", "unknown")
        groups[group].append(r)

    by_group: Dict[str, Dict[str, Any]] = {}

    for group_name, group_results in groups.items():
        group_metrics: Dict[str, Any] = {
            "count": len(group_results),
        }

        latency_stats = calculate_latency_stats(group_results)
        group_metrics.update(latency_stats)

        group_metrics["success_rate"] = calculate_success_rate(group_results)
        group_metrics["avg_tokens"] = calculate_avg_tokens(group_results)

        if ragas_metrics and "by_group" in ragas_metrics:
            ragas_group = ragas_metrics["by_group"].get(group_name, {})
            for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
                if metric in ragas_group:
                    group_metrics[metric] = ragas_group[metric]

        if completeness_metrics and group_name == "complex_decompose":
            if "sub_query_accuracy" in completeness_metrics:
                group_metrics["sub_query_accuracy"] = completeness_metrics["sub_query_accuracy"]
            if "completeness" in completeness_metrics:
                group_metrics["completeness"] = completeness_metrics["completeness"]

        if memory_metrics and group_name == "memory_system":
            if "context_accuracy" in memory_metrics:
                group_metrics["context_accuracy"] = memory_metrics["context_accuracy"]
            if "memory_hit_rate" in memory_metrics:
                group_metrics["memory_hit_rate"] = memory_metrics["memory_hit_rate"]
            if "speedup" in memory_metrics:
                group_metrics["speedup"] = memory_metrics["speedup"]

        by_group[group_name] = group_metrics

    return by_group


def calculate_improvement(baseline_value: float, new_value: float) -> str:
    """Calculate percentage improvement between two values."""
    if baseline_value == 0:
        return "N/A"

    change = ((new_value - baseline_value) / baseline_value) * 100

    if change >= 0:
        return f"+{change:.1f}%"
    else:
        return f"{change:.1f}%"


def generate_comparison_report(eval_data_dir: Path) -> Dict[str, Any]:
    """Generate the full comparison report."""
    all_results: Dict[str, Dict[str, Any]] = {}
    all_ragas: Dict[str, Dict[str, Any]] = {}
    all_completeness: Dict[str, Dict[str, Any]] = {}
    all_memory: Dict[str, Dict[str, Any]] = {}

    for config_name in CONFIG_ORDER:
        ablation = load_ablation_results(config_name, eval_data_dir)
        if ablation:
            all_results[config_name] = ablation

        ragas = load_ragas_metrics(config_name, eval_data_dir)
        if ragas:
            all_ragas[config_name] = ragas

        completeness = load_completeness_metrics(config_name, eval_data_dir)
        if completeness:
            all_completeness[config_name] = completeness

        memory = load_memory_metrics(config_name, eval_data_dir)
        if memory:
            all_memory[config_name] = memory

    overall_metrics: Dict[str, Dict[str, Any]] = {}
    by_group_all: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for config_name in CONFIG_ORDER:
        if config_name not in all_results:
            continue

        results = all_results[config_name].get("results", [])
        ragas = all_ragas.get(config_name)
        completeness = all_completeness.get(config_name)
        memory = all_memory.get(config_name)

        metrics: Dict[str, Any] = {
            "total_queries": len(results),
        }

        latency_stats = calculate_latency_stats(results)
        metrics.update(latency_stats)
        metrics["success_rate"] = calculate_success_rate(results)
        metrics["avg_tokens"] = calculate_avg_tokens(results)

        if ragas and "aggregate_metrics" in ragas:
            for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
                if metric in ragas["aggregate_metrics"]:
                    metrics[metric] = ragas["aggregate_metrics"][metric]

        if completeness:
            if "sub_query_accuracy" in completeness:
                metrics["sub_query_accuracy"] = completeness["sub_query_accuracy"]
            if "completeness" in completeness:
                metrics["completeness"] = completeness["completeness"]

        if memory:
            if "context_accuracy" in memory:
                metrics["context_accuracy"] = memory["context_accuracy"]
            if "memory_hit_rate" in memory:
                metrics["memory_hit_rate"] = memory["memory_hit_rate"]
            if "speedup" in memory:
                metrics["speedup"] = memory["speedup"]

        overall_metrics[config_name] = metrics

        group_metrics = aggregate_metrics_by_group(results, ragas, completeness, memory)
        for group_name, gm in group_metrics.items():
            by_group_all[group_name][config_name] = gm

    improvements: Dict[str, Dict[str, str]] = {}

    comparisons = [
        ("baseline", "decompose", "baseline_to_decompose"),
        ("decompose", "rewrite", "decompose_to_rewrite"),
        ("rewrite", "full", "rewrite_to_full"),
        ("baseline", "full", "baseline_to_full"),
    ]

    for baseline_name, new_name, improvement_key in comparisons:
        if baseline_name not in overall_metrics or new_name not in overall_metrics:
            continue

        baseline_metrics = overall_metrics[baseline_name]
        new_metrics = overall_metrics[new_name]

        improvement: Dict[str, str] = {}

        metric_keys = [
            "faithfulness",
            "answer_relevancy",
            "completeness",
            "sub_query_accuracy",
            "context_accuracy",
            "memory_hit_rate",
            "speedup",
        ]

        for key in metric_keys:
            if key in baseline_metrics and key in new_metrics:
                b_val = baseline_metrics[key]
                n_val = new_metrics[key]
                if b_val is not None and n_val is not None and isinstance(b_val, (int, float)):
                    improvement[key] = calculate_improvement(b_val, n_val)

        latency_keys = ["p95_latency_ms", "avg_latency_ms"]
        for key in latency_keys:
            if key in baseline_metrics and key in new_metrics:
                b_val = baseline_metrics[key]
                n_val = new_metrics[key]
                if b_val > 0:
                    change = ((n_val - b_val) / b_val) * 100
                    improvement[key] = f"{change:.1f}%"

        improvements[improvement_key] = improvement

    report = {
        "timestamp": datetime.now().isoformat(),
        "configurations": CONFIGURATIONS,
        "overall_metrics": overall_metrics,
        "by_group": dict(by_group_all),
        "improvements": improvements,
    }

    return report


def print_report_summary(report: Dict[str, Any]) -> None:
    """Print a summary of the comparison report."""
    print()
    print("=" * 70)
    print("  AGENT ABLATION COMPARISON REPORT")
    print("=" * 70)
    print(f"  Generated: {report['timestamp']}")
    print()

    print("-" * 70)
    print("  CONFIGURATIONS")
    print("-" * 70)
    for config_name in CONFIG_ORDER:
        if config_name in report["configurations"]:
            config = report["configurations"][config_name]
            flags = []
            if config.get("decompose"):
                flags.append("decompose")
            if config.get("rewrite"):
                flags.append("rewrite")
            if config.get("memory"):
                flags.append("memory")
            flags_str = ", ".join(flags) if flags else "none"
            print(f"  {config_name:<12} : {flags_str}")
    print()

    print("-" * 70)
    print("  OVERALL METRICS")
    print("-" * 70)

    metrics_to_show = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("completeness", "Completeness"),
        ("p95_latency_ms", "P95 Latency (ms)"),
        ("avg_tokens", "Avg Tokens"),
    ]

    header = f"  {'Metric':<25}"
    for config_name in CONFIG_ORDER:
        if config_name in report["overall_metrics"]:
            header += f" {config_name:>12}"
    print(header)
    print("  " + "-" * 60)

    for metric_key, metric_name in metrics_to_show:
        row = f"  {metric_name:<25}"
        for config_name in CONFIG_ORDER:
            if config_name in report["overall_metrics"]:
                value = report["overall_metrics"][config_name].get(metric_key)
                if value is not None:
                    if "latency" in metric_key:
                        row += f" {value:>12.0f}"
                    elif isinstance(value, float):
                        row += f" {value:>12.4f}"
                    else:
                        row += f" {value:>12}"
                else:
                    row += " " + "N/A".rjust(12)
        print(row)
    print()

    if report.get("improvements"):
        print("-" * 70)
        print("  IMPROVEMENTS (baseline -> full)")
        print("-" * 70)

        if "baseline_to_full" in report["improvements"]:
            improvements = report["improvements"]["baseline_to_full"]
            for metric, change in sorted(improvements.items()):
                print(f"  {metric:<25} : {change}")

    print()
    print("-" * 70)
    print("  BY GROUP SUMMARY")
    print("-" * 70)

    for group_name, group_data in sorted(report.get("by_group", {}).items()):
        print(f"\n  [{group_name}]")
        for config_name in CONFIG_ORDER:
            if config_name in group_data:
                config_metrics = group_data[config_name]
                count = config_metrics.get("count", 0)
                faithfulness = config_metrics.get("faithfulness", "N/A")
                completeness = config_metrics.get("completeness", "N/A")
                context_acc = config_metrics.get("context_accuracy", "N/A")

                parts = [f"n={count}"]
                if faithfulness != "N/A":
                    parts.append(f"faith={faithfulness:.2f}")
                if completeness != "N/A":
                    parts.append(f"comp={completeness:.2f}")
                if context_acc != "N/A":
                    parts.append(f"ctx_acc={context_acc:.2f}")

                print(f"    {config_name:<12} : {', '.join(parts)}")

    print()
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Agent Ablation Comparison Report")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--eval-data-dir",
        type=str,
        default="eval_data",
        help="Directory containing evaluation data (default: eval_data)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    eval_data_dir = PROJECT_ROOT / args.eval_data_dir

    if not eval_data_dir.exists():
        print(f"Error: Evaluation data directory not found: {eval_data_dir}", file=sys.stderr)
        return 1

    print(f"Generating comparison report from: {eval_data_dir}")

    report = generate_comparison_report(eval_data_dir)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = eval_data_dir / "eval_comparison_agent.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_report_summary(report)

    print(f"\nReport saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
