#!/usr/bin/env python
"""Completeness Evaluator for Agent Ablation Experiments.

Calculates:
1. Sub-query decomposition accuracy (LLM-as-Judge)
2. Answer completeness (coverage of expected sub-questions)

Usage:
    python scripts/evaluate_completeness.py --input eval_data/eval_results_agent_full.json
    python scripts/evaluate_completeness.py --input eval_data/eval_results_agent_decompose.json --output eval_data/completeness_metrics.json

Output format:
    {
        "config": "full",
        "sub_query_accuracy": 0.85,
        "completeness": 0.92,
        "by_group": {
            "complex_decompose": {
                "sub_query_accuracy": 0.85,
                "completeness": 0.92,
                "count": 29
            }
        },
        "details": [...]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
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
class EvaluationResult:
    """Result of evaluating a single query."""

    query: str
    group: str
    sub_query_accuracy: Optional[float] = None
    completeness: Optional[float] = None
    expected_sub_questions: List[str] = field(default_factory=list)
    actual_sub_queries: List[str] = field(default_factory=list)
    answer: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "group": self.group,
            "sub_query_accuracy": self.sub_query_accuracy,
            "completeness": self.completeness,
            "expected_sub_questions": self.expected_sub_questions,
            "actual_sub_queries": self.actual_sub_queries,
            "answer_preview": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "error": self.error,
        }


class CompletenessEvaluator:
    """Evaluator for sub-query decomposition and answer completeness."""

    SUB_QUERY_ACCURACY_PROMPT = """请评估以下复杂问题的分解质量。

原问题: {query}

期望的子问题（参考）:
{expected_sub_questions}

实际分解的子问题:
{actual_sub_queries}

评分标准（1-5分）:
- 5分: 完全覆盖原问题的所有方面，子问题之间无冗余，每个子问题可独立回答
- 4分: 基本覆盖，有小瑕疵但不影响理解
- 3分: 部分覆盖或有明显冗余
- 2分: 覆盖不足，遗漏重要方面
- 1分: 分解错误，与原问题无关

请只输出一个数字（1-5），不要输出其他内容。"""

    COMPLETENESS_PROMPT = """请判断以下回答是否覆盖了给定的子问题。

子问题: {sub_question}

回答: {answer}

请判断该子问题是否在回答中被充分解答:
- 如果回答中明确包含了该子问题的答案，输出 "yes"
- 如果回答中部分涉及但不够充分，输出 "partial"
- 如果回答中没有涉及该子问题，输出 "no"

请只输出 yes、partial 或 no，不要输出其他内容。"""

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

    def evaluate_sub_query_accuracy(
        self,
        query: str,
        expected_sub_questions: List[str],
        actual_sub_queries: List[str],
    ) -> float:
        """Evaluate the quality of sub-query decomposition.

        Args:
            query: Original complex query.
            expected_sub_questions: Expected sub-questions (reference).
            actual_sub_queries: Actual decomposed sub-queries.

        Returns:
            Score between 0 and 1.
        """
        if not expected_sub_questions or not actual_sub_queries:
            return 0.0

        if not any(sq for sq in actual_sub_queries if sq.strip() and sq != query):
            return 0.0

        try:
            llm = self._get_llm_client()
            prompt = self.SUB_QUERY_ACCURACY_PROMPT.format(
                query=query,
                expected_sub_questions="\n".join(f"- {sq}" for sq in expected_sub_questions),
                actual_sub_queries="\n".join(f"- {sq}" for sq in actual_sub_queries),
            )

            response = llm.chat(prompt).strip()

            for char in response:
                if char.isdigit():
                    score = int(char)
                    if 1 <= score <= 5:
                        return score / 5.0

            logger.warning(f"Invalid score response: {response}")
            return 0.0

        except Exception as e:
            logger.error(f"Error evaluating sub-query accuracy: {e}")
            return 0.0

    def evaluate_completeness(
        self,
        answer: str,
        expected_sub_questions: List[str],
    ) -> float:
        """Evaluate how completely the answer covers expected sub-questions.

        Args:
            answer: The generated answer.
            expected_sub_questions: Expected sub-questions to cover.

        Returns:
            Completeness score between 0 and 1.
        """
        if not expected_sub_questions:
            return 1.0

        if not answer or not answer.strip():
            return 0.0

        try:
            llm = self._get_llm_client()
            covered = 0
            partial = 0

            for sub_q in expected_sub_questions:
                prompt = self.COMPLETENESS_PROMPT.format(
                    sub_question=sub_q,
                    answer=answer,
                )

                response = llm.chat(prompt).strip().lower()

                if "yes" in response:
                    covered += 1
                elif "partial" in response:
                    partial += 1

            score = (covered + 0.5 * partial) / len(expected_sub_questions)
            return round(score, 4)

        except Exception as e:
            logger.error(f"Error evaluating completeness: {e}")
            return 0.0

    def evaluate_single(self, result: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single result.

        Args:
            result: Result dict from run_agent_ablation.py

        Returns:
            EvaluationResult with scores.
        """
        query = result.get("query", "")
        group = result.get("group", "unknown")
        expected_sub_questions = result.get("expected_sub_questions", [])
        actual_sub_queries = result.get("sub_queries", [])
        answer = result.get("answer", "")

        eval_result = EvaluationResult(
            query=query,
            group=group,
            expected_sub_questions=expected_sub_questions,
            actual_sub_queries=actual_sub_queries,
            answer=answer,
        )

        if not expected_sub_questions:
            eval_result.error = "No expected sub_questions"
            return eval_result

        eval_result.completeness = self.evaluate_completeness(answer, expected_sub_questions)

        if group == "complex_decompose" and actual_sub_queries:
            eval_result.sub_query_accuracy = self.evaluate_sub_query_accuracy(
                query, expected_sub_questions, actual_sub_queries
            )

        return eval_result

    def evaluate_batch(
        self,
        results: List[Dict[str, Any]],
        target_group: str = "complex_decompose",
    ) -> Dict[str, Any]:
        """Evaluate a batch of results.

        Args:
            results: List of result dicts from run_agent_ablation.py
            target_group: Group to evaluate (default: complex_decompose)

        Returns:
            Evaluation report dict.
        """
        filtered_results = [r for r in results if r.get("group") == target_group]

        if not filtered_results:
            logger.warning(f"No results found for group: {target_group}")
            return {
                "config": results[0].get("config", "unknown") if results else "unknown",
                "target_group": target_group,
                "count": 0,
                "sub_query_accuracy": None,
                "completeness": None,
                "by_group": {},
                "details": [],
            }

        config_name = (
            filtered_results[0].get("config", "unknown") if filtered_results else "unknown"
        )
        eval_results: List[EvaluationResult] = []

        total = len(filtered_results)
        for i, result in enumerate(filtered_results, 1):
            print(f"  [{i}/{total}] Evaluating: {result.get('query', '')[:50]}...")
            eval_result = self.evaluate_single(result)
            eval_results.append(eval_result)

        accuracy_scores = [
            r.sub_query_accuracy for r in eval_results if r.sub_query_accuracy is not None
        ]
        completeness_scores = [r.completeness for r in eval_results if r.completeness is not None]

        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
        avg_completeness = (
            sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        )

        return {
            "config": config_name,
            "target_group": target_group,
            "count": len(filtered_results),
            "sub_query_accuracy": round(avg_accuracy, 4),
            "completeness": round(avg_completeness, 4),
            "by_group": {
                target_group: {
                    "sub_query_accuracy": round(avg_accuracy, 4),
                    "completeness": round(avg_completeness, 4),
                    "count": len(filtered_results),
                }
            },
            "details": [r.to_dict() for r in eval_results],
            "timestamp": datetime.now().isoformat(),
        }


def load_results(input_path: Path) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate completeness metrics")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input results JSON file from run_agent_ablation.py",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output metrics JSON file",
    )
    parser.add_argument(
        "--group",
        "-g",
        type=str,
        default="complex_decompose",
        help="Target group to evaluate (default: complex_decompose)",
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

    print(f"Config: {config_name}")
    print(f"Total results: {len(results)}")
    print(f"Target group: {args.group}")
    print()

    evaluator = CompletenessEvaluator()
    report = evaluator.evaluate_batch(results, target_group=args.group)

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "eval_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"completeness_metrics_{config_name}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("COMPLETENESS EVALUATION REPORT")
    print("=" * 60)
    print(f"  Config: {config_name}")
    print(f"  Target Group: {args.group}")
    print(f"  Evaluated: {report['count']} queries")
    print()
    print(f"  Sub-query Accuracy: {report['sub_query_accuracy']:.2%}")
    print(f"  Completeness: {report['completeness']:.2%}")
    print()
    print(f"  Report saved to: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
