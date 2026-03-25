#!/usr/bin/env python
"""Memory System Evaluator for Agent Ablation Experiments.

Calculates:
1. Context accuracy (multi-turn dialogue understanding)
2. Memory hit rate (memory recall rate)
3. Response speedup (with vs without memory)

Usage:
    python scripts/evaluate_memory.py --input eval_data/eval_results_agent_full.json
    python scripts/evaluate_memory.py --input eval_data/eval_results_agent_full.json --baseline eval_data/eval_results_agent_baseline.json

Output format:
    {
        "config": "full",
        "context_accuracy": 0.91,
        "memory_hit_rate": 0.78,
        "speedup": 0.40,
        "by_scenario": {
            "1": {"turn1_latency": 2500, "avg_other_latency": 1500, "speedup": 0.40},
            ...
        }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
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


@dataclass
class ContextCheckResult:
    """Result of context accuracy check."""

    query: str
    answer: str
    refer_to_previous: str
    is_correct: bool
    reason: str


class MemoryEvaluator:
    """Evaluator for memory system metrics."""

    CONTEXT_CHECK_PROMPT = """请判断以下回答是否正确引用了上下文信息。

问题: {query}
回答: {answer}
应该引用的上下文内容: {refer_to_previous}

判断标准:
- 如果回答正确理解并引用了"{refer_to_previous}"相关的内容，输出 "correct"
- 如果回答误解了上下文或引用了错误的内容，输出 "incorrect"

请只输出 correct 或 incorrect，不要输出其他内容。"""

    def __init__(self, llm_client: Optional[Any] = None):
        """Initialize evaluator.

        Args:
            llm_client: Optional LLM client. If None, will create from settings.
        """
        self._llm_client = llm_client

    def _get_llm_client(self) -> Any:
        """Get or create LLM client."""
        if self._llm_client is None:
            from src.core.llm_client import create_llm_client

            self._llm_client = create_llm_client()
        return self._llm_client

    def check_context_accuracy_rule(
        self,
        query: str,
        answer: str,
        refer_to_previous: str,
    ) -> ContextCheckResult:
        """Check context accuracy using simple rules.

        Args:
            query: The query that needs context.
            answer: The generated answer.
            refer_to_previous: What the answer should reference.

        Returns:
            ContextCheckResult with correctness and reason.
        """
        if not answer or not refer_to_previous:
            return ContextCheckResult(
                query=query,
                answer=answer,
                refer_to_previous=refer_to_previous,
                is_correct=False,
                reason="Missing answer or reference",
            )

        pronouns = ["它", "这个", "那个", "其", "该", "此"]
        has_pronoun_only = True
        query_lower = query.lower()

        for pronoun in pronouns:
            if pronoun in query_lower:
                words_before_pronoun = query_lower.split(pronoun)[0].strip()
                if words_before_pronoun and len(words_before_pronoun) > 1:
                    has_pronoun_only = False
                    break

        key_terms = self._extract_key_terms(refer_to_previous)
        answer_lower = answer.lower()

        found_terms = []
        for term in key_terms:
            if term.lower() in answer_lower:
                found_terms.append(term)

        if found_terms:
            return ContextCheckResult(
                query=query,
                answer=answer,
                refer_to_previous=refer_to_previous,
                is_correct=True,
                reason=f"Found key terms: {found_terms}",
            )

        if has_pronoun_only and any(p in answer_lower for p in pronouns):
            return ContextCheckResult(
                query=query,
                answer=answer,
                refer_to_previous=refer_to_previous,
                is_correct=False,
                reason="Answer uses pronouns without explicit reference",
            )

        return ContextCheckResult(
            query=query,
            answer=answer,
            refer_to_previous=refer_to_previous,
            is_correct=False,
            reason="No key terms found in answer",
        )

    def check_context_accuracy_llm(
        self,
        query: str,
        answer: str,
        refer_to_previous: str,
    ) -> ContextCheckResult:
        """Check context accuracy using LLM.

        Args:
            query: The query that needs context.
            answer: The generated answer.
            refer_to_previous: What the answer should reference.

        Returns:
            ContextCheckResult with correctness and reason.
        """
        if not answer or not refer_to_previous:
            return ContextCheckResult(
                query=query,
                answer=answer,
                refer_to_previous=refer_to_previous,
                is_correct=False,
                reason="Missing answer or reference",
            )

        try:
            llm = self._get_llm_client()
            prompt = self.CONTEXT_CHECK_PROMPT.format(
                query=query,
                answer=answer,
                refer_to_previous=refer_to_previous,
            )

            response = llm.chat(prompt).strip().lower()

            is_correct = "correct" in response and "incorrect" not in response

            return ContextCheckResult(
                query=query,
                answer=answer,
                refer_to_previous=refer_to_previous,
                is_correct=is_correct,
                reason=f"LLM judgment: {response}",
            )

        except Exception as e:
            logger.error(f"Error checking context accuracy with LLM: {e}")
            return self.check_context_accuracy_rule(query, answer, refer_to_previous)

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        terms = []

        patterns = [
            r"[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*",
            r"[\u4e00-\u9fff]{2,}",
            r"\b[A-Z]{2,}\b",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            terms.extend(matches)

        stopwords = {"的", "是", "在", "和", "了", "有", "不", "这", "那", "为"}
        terms = [t for t in terms if t.lower() not in stopwords and len(t) > 1]

        return list(set(terms))

    def calculate_context_accuracy(
        self,
        results: List[Dict[str, Any]],
        use_llm: bool = False,
    ) -> float:
        """Calculate context accuracy for memory_system group.

        Args:
            results: List of result dicts.
            use_llm: Whether to use LLM for checking.

        Returns:
            Context accuracy score (0-1).
        """
        memory_results = [
            r
            for r in results
            if r.get("group") == "memory_system"
            and r.get("needs_context") is True
            and r.get("refer_to_previous")
        ]

        if not memory_results:
            return 0.0

        correct_count = 0

        for result in memory_results:
            query = result.get("query", "")
            answer = result.get("answer", "")
            refer_to_previous = result.get("refer_to_previous", "")

            if use_llm:
                check_result = self.check_context_accuracy_llm(query, answer, refer_to_previous)
            else:
                check_result = self.check_context_accuracy_rule(query, answer, refer_to_previous)

            if check_result.is_correct:
                correct_count += 1

        return correct_count / len(memory_results)

    def calculate_memory_hit_rate(self, results: List[Dict[str, Any]]) -> float:
        """Calculate memory hit rate.

        Args:
            results: List of result dicts.

        Returns:
            Memory hit rate (0-1).
        """
        memory_results = [
            r
            for r in results
            if r.get("group") == "memory_system"
            and r.get("memory_type") == "short_term"
            and r.get("turn", 1) > 1
        ]

        if not memory_results:
            return 0.0

        hit_count = sum(1 for r in memory_results if r.get("used_memory") is True)

        return hit_count / len(memory_results)

    def calculate_speedup(
        self,
        full_results: List[Dict[str, Any]],
        baseline_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Calculate response speedup from memory.

        Args:
            full_results: Results with memory enabled.
            baseline_results: Results without memory (for comparison).

        Returns:
            Dict with speedup metrics.
        """
        full_memory = [r for r in full_results if r.get("group") == "memory_system"]

        if not full_memory:
            return {"speedup": 0.0, "by_scenario": {}}

        scenarios: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: {"turn1": [], "other": []}
        )

        for r in full_memory:
            scenario_id = r.get("scenario_id")
            turn = r.get("turn", 1)
            latency = r.get("latency_ms", 0)

            if scenario_id is None:
                continue

            if turn == 1:
                scenarios[scenario_id]["turn1"].append(latency)
            else:
                scenarios[scenario_id]["other"].append(latency)

        speedups = []
        by_scenario = {}

        for scenario_id, latencies in scenarios.items():
            if not latencies["turn1"] or not latencies["other"]:
                continue

            turn1_avg = sum(latencies["turn1"]) / len(latencies["turn1"])
            other_avg = sum(latencies["other"]) / len(latencies["other"])

            if turn1_avg > 0:
                speedup = (turn1_avg - other_avg) / turn1_avg
                speedups.append(speedup)

                by_scenario[str(scenario_id)] = {
                    "turn1_latency": round(turn1_avg, 2),
                    "avg_other_latency": round(other_avg, 2),
                    "speedup": round(speedup, 4),
                }

        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0

        return {
            "speedup": round(avg_speedup, 4),
            "by_scenario": by_scenario,
        }

    def evaluate_batch(
        self,
        results: List[Dict[str, Any]],
        baseline_results: Optional[List[Dict[str, Any]]] = None,
        use_llm_for_context: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate memory metrics for a batch of results.

        Args:
            results: Results from run_agent_ablation.py.
            baseline_results: Baseline results for speedup comparison.
            use_llm_for_context: Use LLM for context accuracy check.

        Returns:
            Evaluation report dict.
        """
        config_name = results[0].get("config", "unknown") if results else "unknown"

        memory_results = [r for r in results if r.get("group") == "memory_system"]

        if not memory_results:
            return {
                "config": config_name,
                "context_accuracy": None,
                "memory_hit_rate": None,
                "speedup": None,
                "by_scenario": {},
                "count": 0,
                "message": "No memory_system results found",
            }

        context_accuracy = self.calculate_context_accuracy(results, use_llm=use_llm_for_context)
        memory_hit_rate = self.calculate_memory_hit_rate(results)
        speedup_data = self.calculate_speedup(results, baseline_results)

        return {
            "config": config_name,
            "context_accuracy": round(context_accuracy, 4),
            "memory_hit_rate": round(memory_hit_rate, 4),
            "speedup": speedup_data["speedup"],
            "by_scenario": speedup_data["by_scenario"],
            "count": len(memory_results),
            "timestamp": datetime.now().isoformat(),
        }


def load_results(input_path: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate memory system metrics")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input results JSON file from run_agent_ablation.py",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        type=str,
        default=None,
        help="Baseline results for speedup comparison",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output metrics JSON file",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for context accuracy check",
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

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    print(f"Loading results from: {input_path}")
    data = load_results(input_path)
    config_name = data.get("config", "unknown")
    results = data.get("results", [])

    baseline_results = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            print(f"Loading baseline from: {baseline_path}")
            baseline_data = load_results(baseline_path)
            baseline_results = baseline_data.get("results", [])

    print(f"Config: {config_name}")
    print(f"Total results: {len(results)}")
    print()

    evaluator = MemoryEvaluator()
    report = evaluator.evaluate_batch(
        results,
        baseline_results=baseline_results,
        use_llm_for_context=args.use_llm,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "eval_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"memory_metrics_{config_name}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("MEMORY SYSTEM EVALUATION REPORT")
    print("=" * 60)
    print(f"  Config: {config_name}")
    print(f"  Memory system queries: {report['count']}")
    print()
    print(
        f"  Context Accuracy: {report['context_accuracy']:.2%}"
        if report["context_accuracy"]
        else "  Context Accuracy: N/A"
    )
    print(
        f"  Memory Hit Rate: {report['memory_hit_rate']:.2%}"
        if report["memory_hit_rate"]
        else "  Memory Hit Rate: N/A"
    )
    print(
        f"  Response Speedup: {report['speedup']:.2%}"
        if report["speedup"]
        else "  Response Speedup: N/A"
    )
    print()

    if report.get("by_scenario"):
        print("  By Scenario:")
        for scenario_id, data in report["by_scenario"].items():
            print(
                f"    Scenario {scenario_id}: turn1={data['turn1_latency']:.0f}ms, "
                f"avg_other={data['avg_other_latency']:.0f}ms, speedup={data['speedup']:.2%}"
            )

    print()
    print(f"  Report saved to: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
