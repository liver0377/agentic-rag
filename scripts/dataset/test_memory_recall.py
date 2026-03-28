"""长期记忆模块测试脚本。

测试两个核心指标：
1. 记忆命中率：测试系统能否正确召回相关记忆
2. 重复问题响应速度提升：测试记忆系统对响应速度的影响

Usage:
    python scripts/dataset/test_memory_recall.py --input preset_memories.json
    python scripts/dataset/test_memory_recall.py --input preset_memories.json --iterations 10
    python scripts/dataset/test_memory_recall.py --report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """查询结果数据类。"""

    query_id: str
    user_id: str
    query: str
    expected_memory_ids: List[str]
    retrieved_memory_ids: List[str]
    hit_count: int
    expected_count: int
    recall_rate: float
    query_time_ms: float
    scenario: str


@dataclass
class TestMetrics:
    """测试指标数据类。"""

    total_queries: int = 0
    total_hits: int = 0
    total_expected: int = 0
    avg_recall_rate: float = 0.0
    avg_query_time_ms: float = 0.0
    p50_query_time_ms: float = 0.0
    p95_query_time_ms: float = 0.0
    p99_query_time_ms: float = 0.0
    by_scenario: Dict[str, Dict[str, float]] = field(default_factory=dict)


class MemoryRecallTester:
    """记忆召回测试器。

    测试长期记忆系统的两个核心指标：
    1. 记忆命中率 (Recall Rate)
    2. 重复问题响应速度提升
    """

    def __init__(
        self,
        persist_directory: str = "./data/memory_test_db",
        collection_name: str = "memory_test",
        embedding_provider: str = "mock",
    ):
        """初始化测试器。

        Args:
            persist_directory: ChromaDB 持久化目录
            collection_name: 集合名称
            embedding_provider: Embedding 提供商
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self._loader = None
        self._results: List[QueryResult] = []
        self._baseline_times: Dict[str, List[float]] = {}

    @property
    def loader(self):
        """延迟加载 MemoryLoader。"""
        if self._loader is None:
            from scripts.dataset.memory_loader import MemoryLoader

            self._loader = MemoryLoader(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_provider=self.embedding_provider,
            )
        return self._loader

    def run_recall_test(self, preset_data: Dict[str, Any], top_k: int = 5) -> List[QueryResult]:
        """运行记忆命中率测试。

        Args:
            preset_data: 预置记忆数据
            top_k: 召回数量

        Returns:
            查询结果列表
        """
        test_queries = preset_data.get("test_queries", [])
        results = []

        logger.info(f"开始记忆命中率测试，共 {len(test_queries)} 个查询...")

        for i, test_query in enumerate(test_queries):
            query_id = test_query["query_id"]
            user_id = test_query["user_id"]
            query = test_query["query"]
            expected_ids = set(test_query.get("expected_memory_ids", []))
            scenario = test_query.get("scenario", "unknown")

            start_time = time.perf_counter()
            retrieved = self.loader.query_memories(query, user_id=user_id, top_k=top_k)
            query_time_ms = (time.perf_counter() - start_time) * 1000

            retrieved_ids = set(r["id"] for r in retrieved)
            hit_count = len(expected_ids & retrieved_ids)
            expected_count = len(expected_ids)
            recall_rate = hit_count / expected_count if expected_count > 0 else 0.0

            result = QueryResult(
                query_id=query_id,
                user_id=user_id,
                query=query,
                expected_memory_ids=list(expected_ids),
                retrieved_memory_ids=list(retrieved_ids),
                hit_count=hit_count,
                expected_count=expected_count,
                recall_rate=recall_rate,
                query_time_ms=query_time_ms,
                scenario=scenario,
            )
            results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(f"已完成 {i + 1}/{len(test_queries)} 个查询")

        self._results = results
        return results

    def run_speed_test(
        self,
        preset_data: Dict[str, Any],
        iterations: int = 10,
        warmup: int = 3,
    ) -> Dict[str, Any]:
        """运行响应速度测试。

        测试重复问题的响应速度提升。

        Args:
            preset_data: 预置记忆数据
            iterations: 每个查询的迭代次数
            warmup: 预热次数

        Returns:
            速度测试结果
        """
        test_queries = preset_data.get("test_queries", [])

        logger.info(f"开始响应速度测试，每个查询迭代 {iterations} 次...")

        speed_results = {
            "by_query": [],
            "first_query_avg_ms": 0.0,
            "subsequent_avg_ms": 0.0,
            "speedup_ratio": 0.0,
        }

        first_query_times = []
        subsequent_times = []

        for query_data in test_queries:
            query_id = query_data["query_id"]
            user_id = query_data["user_id"]
            query = query_data["query"]

            query_times = []

            for _ in range(warmup):
                self.loader.query_memories(query, user_id=user_id, top_k=5)

            for iteration in range(iterations):
                start_time = time.perf_counter()
                self.loader.query_memories(query, user_id=user_id, top_k=5)
                query_time_ms = (time.perf_counter() - start_time) * 1000
                query_times.append(query_time_ms)

                if iteration == 0:
                    first_query_times.append(query_time_ms)
                else:
                    subsequent_times.append(query_time_ms)

            query_result = {
                "query_id": query_id,
                "query": query,
                "avg_time_ms": statistics.mean(query_times),
                "min_time_ms": min(query_times),
                "max_time_ms": max(query_times),
                "std_dev_ms": statistics.stdev(query_times) if len(query_times) > 1 else 0.0,
            }
            speed_results["by_query"].append(query_result)

        if first_query_times:
            speed_results["first_query_avg_ms"] = statistics.mean(first_query_times)
        if subsequent_times:
            speed_results["subsequent_avg_ms"] = statistics.mean(subsequent_times)

        if speed_results["subsequent_avg_ms"] > 0:
            speed_results["speedup_ratio"] = (
                speed_results["first_query_avg_ms"] / speed_results["subsequent_avg_ms"]
            )

        return speed_results

    def calculate_metrics(self) -> TestMetrics:
        """计算测试指标。

        Returns:
            测试指标汇总
        """
        if not self._results:
            return TestMetrics()

        total_queries = len(self._results)
        total_hits = sum(r.hit_count for r in self._results)
        total_expected = sum(r.expected_count for r in self._results)

        query_times = [r.query_time_ms for r in self._results]
        query_times_sorted = sorted(query_times)

        metrics = TestMetrics(
            total_queries=total_queries,
            total_hits=total_hits,
            total_expected=total_expected,
            avg_recall_rate=total_hits / total_expected if total_expected > 0 else 0.0,
            avg_query_time_ms=statistics.mean(query_times),
            p50_query_time_ms=self._percentile(query_times_sorted, 50),
            p95_query_time_ms=self._percentile(query_times_sorted, 95),
            p99_query_time_ms=self._percentile(query_times_sorted, 99),
        )

        by_scenario: Dict[str, List[QueryResult]] = {}
        for result in self._results:
            scenario = result.scenario
            if scenario not in by_scenario:
                by_scenario[scenario] = []
            by_scenario[scenario].append(result)

        for scenario, results in by_scenario.items():
            scenario_hits = sum(r.hit_count for r in results)
            scenario_expected = sum(r.expected_count for r in results)
            scenario_times = [r.query_time_ms for r in results]

            metrics.by_scenario[scenario] = {
                "count": len(results),
                "recall_rate": scenario_hits / scenario_expected if scenario_expected > 0 else 0.0,
                "avg_time_ms": statistics.mean(scenario_times),
            }

        return metrics

    def _percentile(self, sorted_data: List[float], percentile: int) -> float:
        """计算百分位数。

        Args:
            sorted_data: 已排序的数据
            percentile: 百分位 (0-100)

        Returns:
            百分位数值
        """
        if not sorted_data:
            return 0.0

        n = len(sorted_data)
        index = (percentile / 100) * (n - 1)
        lower = int(index)
        upper = lower + 1

        if upper >= n:
            return sorted_data[-1]

        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

    def generate_report(
        self,
        metrics: TestMetrics,
        speed_results: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """生成测试报告。

        Args:
            metrics: 测试指标
            speed_results: 速度测试结果
            output_path: 报告输出路径

        Returns:
            报告内容
        """
        lines = [
            "# 长期记忆模块测试报告",
            "",
            f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. 记忆命中率测试",
            "",
            "### 总体指标",
            "",
            "| 指标 | 值 |",
            "|------|------|",
            f"| 总查询数 | {metrics.total_queries} |",
            f"| 命中记忆数 | {metrics.total_hits} |",
            f"| 期望记忆数 | {metrics.total_expected} |",
            f"| **平均召回率** | **{metrics.avg_recall_rate:.2%}** |",
            "",
            "### 查询延迟",
            "",
            "| 指标 | 值 |",
            "|------|------|",
            f"| 平均延迟 | {metrics.avg_query_time_ms:.2f} ms |",
            f"| P50 延迟 | {metrics.p50_query_time_ms:.2f} ms |",
            f"| P95 延迟 | {metrics.p95_query_time_ms:.2f} ms |",
            f"| P99 延迟 | {metrics.p99_query_time_ms:.2f} ms |",
            "",
            "### 按场景分析",
            "",
            "| 场景 | 查询数 | 召回率 | 平均延迟 |",
            "|------|--------|--------|----------|",
        ]

        for scenario, data in sorted(metrics.by_scenario.items()):
            lines.append(
                f"| {scenario} | {int(data['count'])} | {data['recall_rate']:.2%} | {data['avg_time_ms']:.2f} ms |"
            )

        if speed_results:
            lines.extend(
                [
                    "",
                    "## 2. 响应速度提升测试",
                    "",
                    "### 速度对比",
                    "",
                    "| 指标 | 值 |",
                    "|------|------|",
                    f"| 首次查询平均延迟 | {speed_results['first_query_avg_ms']:.2f} ms |",
                    f"| 后续查询平均延迟 | {speed_results['subsequent_avg_ms']:.2f} ms |",
                    f"| **速度提升比** | **{speed_results['speedup_ratio']:.2f}x** |",
                    "",
                    "### 各查询详情",
                    "",
                    "| 查询 ID | 平均延迟 | 最小延迟 | 最大延迟 | 标准差 |",
                    "|---------|----------|----------|----------|--------|",
                ]
            )

            for query in speed_results["by_query"]:
                lines.append(
                    f"| {query['query_id']} | {query['avg_time_ms']:.2f} ms | "
                    f"{query['min_time_ms']:.2f} ms | {query['max_time_ms']:.2f} ms | "
                    f"{query['std_dev_ms']:.2f} ms |"
                )

        lines.extend(
            [
                "",
                "## 3. 结论与建议",
                "",
            ]
        )

        if metrics.avg_recall_rate >= 0.8:
            lines.append("- 记忆召回率表现优秀 (>=80%)")
        elif metrics.avg_recall_rate >= 0.6:
            lines.append("- 记忆召回率表现良好 (>=60%)")
        else:
            lines.append("- 记忆召回率需要优化 (<60%)，建议检查 Embedding 质量或调整检索参数")

        if metrics.avg_query_time_ms < 100:
            lines.append("- 查询延迟表现优秀 (<100ms)")
        elif metrics.avg_query_time_ms < 500:
            lines.append("- 查询延迟可接受 (<500ms)")
        else:
            lines.append("- 查询延迟较高 (>500ms)，建议优化索引或使用缓存")

        if speed_results and speed_results["speedup_ratio"] > 1.5:
            lines.append("- 重复查询有明显速度提升 (>1.5x)，缓存机制有效")
        elif speed_results:
            lines.append("- 重复查询速度提升有限，缓存机制可能需要优化")

        report = "\n".join(lines)

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report, encoding="utf-8")
            logger.info(f"报告已保存: {output_file}")

        return report


async def main() -> int:
    parser = argparse.ArgumentParser(description="长期记忆模块测试")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="scripts/dataset/preset_memories.json",
        help="预置记忆 JSON 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./data/memory_test_db",
        help="ChromaDB 持久化目录",
    )
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default="memory_test",
        help="集合名称",
    )
    parser.add_argument(
        "--embedding",
        "-e",
        type=str,
        choices=["mock", "real"],
        default="mock",
        help="Embedding 提供商",
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=10,
        help="速度测试迭代次数",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="生成测试报告",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default="eval_data/memory_test_report.md",
        help="报告输出路径",
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="跳过数据导入（使用已有数据）",
    )

    args = parser.parse_args()

    input_path = (
        PROJECT_ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    )

    if not input_path.exists():
        print(f"❌ 错误: 文件不存在 - {input_path}")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        preset_data = json.load(f)

    tester = MemoryRecallTester(
        persist_directory=args.output_dir,
        collection_name=args.collection,
        embedding_provider=args.embedding,
    )

    if not args.skip_import:
        print("📦 导入预置记忆数据...")
        from scripts.dataset.memory_loader import MemoryLoader

        loader = MemoryLoader(
            persist_directory=args.output_dir,
            collection_name=args.collection,
            embedding_provider=args.embedding,
        )
        loader.import_memories(preset_data, clear_existing=True)

    print("\n🔍 运行记忆命中率测试...")
    tester.run_recall_test(preset_data)
    metrics = tester.calculate_metrics()

    print("\n⚡ 运行响应速度测试...")
    speed_results = tester.run_speed_test(preset_data, iterations=args.iterations)

    print("\n" + "=" * 60)
    print("📊 测试结果摘要")
    print("=" * 60)
    print(f"总查询数: {metrics.total_queries}")
    print(f"平均召回率: {metrics.avg_recall_rate:.2%}")
    print(f"平均查询延迟: {metrics.avg_query_time_ms:.2f} ms")
    print(f"首次查询延迟: {speed_results['first_query_avg_ms']:.2f} ms")
    print(f"后续查询延迟: {speed_results['subsequent_avg_ms']:.2f} ms")
    print(f"速度提升比: {speed_results['speedup_ratio']:.2f}x")

    if args.report:
        print("\n📝 生成测试报告...")
        report = tester.generate_report(
            metrics,
            speed_results,
            output_path=str(PROJECT_ROOT / args.report_output),
        )
        print(f"\n报告已保存: {args.report_output}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
