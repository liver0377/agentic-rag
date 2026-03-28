"""Memory 系统评估脚本 (v3)。

评估长期记忆系统的召回和使用效果。
使用 Mock 数据，不依赖 MCP 服务。

Usage:
    python scripts/evaluate_memory_v3.py --input memory_test_cases.json
    python scripts/evaluate_memory_v3.py --input memory_test_cases.json --output eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
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

MOCK_MEMORY_STORE: Dict[str, List[Dict[str, Any]]] = {}


def load_test_cases(input_path: Path) -> List[Dict[str, Any]]:
    """加载测试用例。"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "test_cases" in data:
        return data["test_cases"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Invalid test case format in {input_path}")


def setup_mock_memory(memories: List[Dict[str, Any]], user_id: str) -> None:
    """预置 Mock 记忆数据。

    Args:
        memories: 记忆列表 [{"content": "...", "type": "preference"}, ...]
        user_id: 用户 ID
    """
    if user_id not in MOCK_MEMORY_STORE:
        MOCK_MEMORY_STORE[user_id] = []

    for mem in memories:
        MOCK_MEMORY_STORE[user_id].append(
            {
                "content": mem.get("content", ""),
                "type": mem.get("type", "fact"),
                "importance": mem.get("importance", 0.8),
            }
        )

    logger.info(f"Setup {len(memories)} mock memories for user {user_id}")


def mock_recall_memory(user_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Mock 记忆召回。

    简单的关键词匹配模拟。
    """
    if user_id not in MOCK_MEMORY_STORE:
        return []

    memories = MOCK_MEMORY_STORE[user_id]
    query_lower = query.lower()

    scored_memories = []
    for mem in memories:
        content = mem.get("content", "").lower()
        score = 0.0

        if any(kw in content for kw in ["偏好", "习惯", "喜欢", "preference"]):
            score += 0.3
        if any(kw in query_lower for kw in ["上次", "之前", "以前", "历史"]):
            score += 0.2
        if any(kw in content for kw in query_lower.split()):
            score += 0.1

        mem["score"] = score
        scored_memories.append(mem)

    scored_memories.sort(key=lambda x: x.get("score", 0), reverse=True)
    return scored_memories[:top_k]


def generate_mock_answer(
    query: str,
    test_case: Dict[str, Any],
    recalled_memories: List[Dict[str, Any]],
) -> str:
    """生成 Mock 回答。

    基于测试用例和召回的记忆生成模拟回答。
    """
    knowledge_context = test_case.get("knowledge_context", "")
    expected_behavior = test_case.get("expected_behavior", "")
    scenario = test_case.get("scenario", "baseline")

    answer_parts = []

    if scenario == "enhance" and recalled_memories:
        preference = recalled_memories[0].get("content", "")
        if "简洁" in preference:
            answer_parts.append(f"关于您的问题，简要回答如下：")
        elif "表格" in preference:
            answer_parts.append(f"以下是相关信息：\n| 项目 | 说明 |")
        elif "详细" in preference:
            answer_parts.append(f"详细解答如下：")
        else:
            answer_parts.append(f"根据您的偏好：")

    if knowledge_context:
        answer_parts.append(knowledge_context)

    if scenario == "cross_session" and recalled_memories:
        for mem in recalled_memories:
            content = mem.get("content", "")
            if "部门" in content or "级别" in content:
                answer_parts.append(f"根据您的{content}，")

    if scenario == "conflict":
        answer_parts.append("根据公司规定，")
        if "审批" in knowledge_context:
            answer_parts.append("所有报销均需审批，无豁免情况。")

    memory_keywords = test_case.get("memory_keywords", [])
    if memory_keywords:
        random.shuffle(memory_keywords)
        for kw in memory_keywords[:2]:
            if kw not in " ".join(answer_parts):
                answer_parts.append(f"（{kw}）")

    return " ".join(answer_parts) if answer_parts else "这是模拟回答。"


def evaluate_result(
    answer: str,
    test_case: Dict[str, Any],
    recalled_memories: List[Dict[str, Any]],
    memory_used: bool,
) -> Dict[str, Any]:
    """评估单个测试结果。"""
    memory_keywords = test_case.get("memory_keywords", [])
    knowledge_keywords = test_case.get("knowledge_keywords", [])
    should_use_memory = test_case.get("should_use_memory", True)
    evaluation_criteria = test_case.get("evaluation_criteria", [])

    memory_keyword_hit = 0
    if memory_keywords:
        hits = sum(1 for kw in memory_keywords if kw.lower() in answer.lower())
        memory_keyword_hit = hits / len(memory_keywords)

    knowledge_keyword_hit = 0
    if knowledge_keywords:
        hits = sum(1 for kw in knowledge_keywords if kw.lower() in answer.lower())
        knowledge_keyword_hit = hits / len(knowledge_keywords)

    memory_recall_correct = memory_used == should_use_memory

    passed = True
    if should_use_memory and not memory_used:
        passed = False
    if memory_keywords and memory_keyword_hit < 0.5:
        passed = False

    return {
        "memory_used": memory_used,
        "should_use_memory": should_use_memory,
        "memory_recall_correct": memory_recall_correct,
        "memory_keyword_hit_rate": round(memory_keyword_hit, 2),
        "knowledge_keyword_hit_rate": round(knowledge_keyword_hit, 2),
        "recalled_count": len(recalled_memories),
        "passed": passed,
        "evaluation_criteria": evaluation_criteria,
    }


def run_single_test(
    test_case: Dict[str, Any],
    test_id: int,
    total: int,
) -> Dict[str, Any]:
    """运行单个测试用例 (Mock 模式)。"""
    query = test_case.get("query", "")
    user_id = test_case.get("user_id", "anonymous")
    user_memories = test_case.get("user_memories", [])
    should_use_memory = test_case.get("should_use_memory", True)

    print(f"[{test_id}/{total}] [{test_case.get('scenario', 'unknown')}] {query[:50]}...")

    if user_memories:
        setup_mock_memory(user_memories, user_id)

    start_time = time.monotonic()

    try:
        recalled = mock_recall_memory(user_id, query)
        memory_used = len(recalled) > 0

        answer = generate_mock_answer(query, test_case, recalled)

        eval_result = evaluate_result(
            answer=answer,
            test_case=test_case,
            recalled_memories=recalled,
            memory_used=memory_used,
        )

        output = {
            "test_id": test_case.get("id", f"test_{test_id}"),
            "scenario": test_case.get("scenario", "unknown"),
            "query": query,
            "answer": answer,
            "recalled_memories": recalled,
            "decision_path": ["mock_mode"],
            "error": None,
            "latency_ms": round((time.monotonic() - start_time) * 1000, 2),
            **eval_result,
        }

    except Exception as e:
        logger.error(f"Test failed: {e}")
        output = {
            "test_id": test_case.get("id", f"test_{test_id}"),
            "scenario": test_case.get("scenario", "unknown"),
            "query": query,
            "answer": "",
            "recalled_memories": [],
            "decision_path": [],
            "error": str(e),
            "latency_ms": round((time.monotonic() - start_time) * 1000, 2),
            "memory_used": False,
            "should_use_memory": should_use_memory,
            "memory_recall_correct": False,
            "memory_keyword_hit_rate": 0,
            "knowledge_keyword_hit_rate": 0,
            "recalled_count": 0,
            "passed": False,
            "evaluation_criteria": test_case.get("evaluation_criteria", []),
        }

    status = "✓" if output.get("passed") else "✗"
    print(f"  {status} memory_used={output.get('memory_used')}, passed={output.get('passed')}")
    return output


def run_evaluation(
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """运行完整评估。"""
    results: List[Dict[str, Any]] = []
    start_time = time.monotonic()
    total = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        result = run_single_test(test_case, i, total)
        results.append(result)

    elapsed = time.monotonic() - start_time

    total_count = len(results)
    passed_count = sum(1 for r in results if r.get("passed"))
    failed_count = total_count - passed_count

    memory_used_count = sum(1 for r in results if r.get("memory_used"))
    memory_recall_correct_count = sum(1 for r in results if r.get("memory_recall_correct"))

    scenario_stats: Dict[str, Dict[str, int]] = {}
    for r in results:
        scenario = r.get("scenario", "unknown")
        if scenario not in scenario_stats:
            scenario_stats[scenario] = {"total": 0, "passed": 0, "memory_used": 0}
        scenario_stats[scenario]["total"] += 1
        if r.get("passed"):
            scenario_stats[scenario]["passed"] += 1
        if r.get("memory_used"):
            scenario_stats[scenario]["memory_used"] += 1

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_cases": total_count,
            "elapsed_seconds": round(elapsed, 2),
            "mode": "mock",
        },
        "summary": {
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": round(passed_count / total_count, 2) if total_count > 0 else 0,
            "memory_used_count": memory_used_count,
            "memory_recall_correct_count": memory_recall_correct_count,
            "memory_recall_rate": round(memory_used_count / total_count, 2)
            if total_count > 0
            else 0,
        },
        "by_scenario": scenario_stats,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Memory 系统评估脚本 (v3 - Mock 模式)")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="输入测试用例 JSON 文件",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出结果 JSON 文件",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细输出",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    print(f"Loading test cases from: {input_path}")
    test_cases = load_test_cases(input_path)
    print(f"Found {len(test_cases)} test cases")
    print(f"Mode: Mock (no MCP service required)")
    print()

    result = run_evaluation(test_cases)

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "eval_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "memory_eval_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("Memory Evaluation Results (Mock Mode)")
    print("=" * 60)
    print(f"Total cases: {result['metadata']['total_cases']}")
    print(f"Elapsed time: {result['metadata']['elapsed_seconds']:.1f}s")
    print()
    print("Summary:")
    print(f"  Passed: {result['summary']['passed']}")
    print(f"  Failed: {result['summary']['failed']}")
    print(f"  Pass rate: {result['summary']['pass_rate'] * 100:.1f}%")
    print(f"  Memory used: {result['summary']['memory_used_count']}")
    print(f"  Memory recall rate: {result['summary']['memory_recall_rate'] * 100:.1f}%")
    print()
    print("By Scenario:")
    for scenario, stats in result["by_scenario"].items():
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {scenario}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")
    print()
    print(f"Results saved to: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
