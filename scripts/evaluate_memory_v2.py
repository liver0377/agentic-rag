"""Memory System Evaluator v2.

基于 LLM-as-Judge 的记忆系统评估器，支持：
1. 短期记忆（滑动窗口）评估
2. 长期记忆（向量召回）评估
3. 记忆命中率（LLM 判定）
4. 上下文准确率

Usage:
    python scripts/evaluate_memory_v2.py --test-set data/test_memory_cases.json
    python scripts/evaluate_memory_v2.py --test-set data/test_memory_cases.json --output results/memory_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
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
class MemoryTestCase:
    """记忆系统测试用例。"""

    scenario_id: int
    turn: int
    query: str
    ground_truth: str
    memory_type: Optional[str] = None
    refer_to_previous: Optional[str] = None
    memory_relevance_prompt: Optional[str] = None
    expected_keywords: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryTestCase":
        return cls(
            scenario_id=data.get("scenario_id", 0),
            turn=data.get("turn", 1),
            query=data.get("query", ""),
            ground_truth=data.get("ground_truth", ""),
            memory_type=data.get("memory_type"),
            refer_to_previous=data.get("refer_to_previous"),
            memory_relevance_prompt=data.get("memory_relevance_prompt"),
            expected_keywords=data.get("expected_keywords", []),
        )


@dataclass
class MemoryHitResult:
    """记忆命中评估结果。"""

    is_hit: bool
    relevance: str
    confidence: float
    matched_keywords: List[str]
    reasoning: str


@dataclass
class EvaluationResult:
    """单条评估结果。"""

    scenario_id: int
    turn: int
    query: str
    memory_type: Optional[str]
    is_hit: bool
    relevance: str
    confidence: float
    matched_keywords: List[str]
    reasoning: str
    recalled_count: int
    latency_ms: float = 0.0
    recall_success: bool = True
    first_relevant_rank: Optional[int] = None
    expected_keywords: List[str] = field(default_factory=list)


@dataclass
class LongTermMemoryConfig:
    """长期记忆配置。"""

    collection: str = "long_term_memory"
    top_k: int = 5
    enable_hybrid_search: bool = True


class LongTermMemoryStore:
    """长期记忆存储（模拟实现）。

    在实际使用中，应该连接到 MCP Server 的向量库。
    这里提供一个模拟实现，用于测试和演示。
    """

    def __init__(self, config: Optional[LongTermMemoryConfig] = None):
        self._config = config or LongTermMemoryConfig()
        self._memories: Dict[str, List[Dict[str, Any]]] = {}
        self._all_memories: List[Dict[str, Any]] = []

    def save(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """保存记忆到长期存储。"""
        import time

        memory_id = f"ltm_{int(time.time() * 1000)}_{len(self._all_memories)}"
        memory = {
            "id": memory_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        if session_id not in self._memories:
            self._memories[session_id] = []
        self._memories[session_id].append(memory)
        self._all_memories.append(memory)

        logger.debug(f"LongTermMemory.save: {memory_id}")
        return memory_id

    def recall_by_query(
        self,
        query: str,
        limit: int = 5,
        session_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """基于查询召回记忆（模拟向量搜索）。

        实际实现应该：
        1. 将 query 转换为 embedding
        2. 在 ChromaDB 中进行向量搜索
        3. 应用 RRF 融合（如果启用 hybrid）

        这里使用简单的关键词匹配模拟。
        """
        import time

        start_time = time.time()

        candidates = self._all_memories
        if session_filter:
            candidates = [m for m in candidates if m["session_id"] == session_filter]

        scored_memories = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for memory in candidates:
            content_lower = memory["content"].lower()
            score = 0.0

            for word in query_words:
                if word in content_lower:
                    score += 1.0

            for keyword in query_words:
                if keyword in content_lower:
                    score += 0.5

            if score > 0:
                scored_memories.append((memory, score))

        scored_memories.sort(key=lambda x: x[1], reverse=True)
        results = [m[0] for m in scored_memories[:limit]]

        latency = (time.time() - start_time) * 1000
        logger.debug(
            f"LongTermMemory.recall_by_query: query='{query[:30]}...', "
            f"returned={len(results)}, latency={latency:.1f}ms"
        )

        return results

    def get_all_sessions(self) -> List[str]:
        """获取所有会话 ID。"""
        return list(self._memories.keys())


class MemoryEvaluatorV2:
    """记忆系统评估器 v2。"""

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._llm_client = llm_client
        self._config = config or {}
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}
        self._long_term_store = LongTermMemoryStore()

    def _get_llm_client(self) -> Any:
        if self._llm_client is None:
            from src.core.llm_client import create_llm_client

            self._llm_client = create_llm_client()
        return self._llm_client

    async def evaluate_memory_hit(
        self,
        query: str,
        recalled_memories: List[Dict[str, Any]],
        memory_relevance_prompt: str,
        expected_keywords: List[str],
    ) -> MemoryHitResult:
        """使用 LLM 评估记忆是否命中。"""
        if not recalled_memories:
            return MemoryHitResult(
                is_hit=False,
                relevance="not_relevant",
                confidence=1.0,
                matched_keywords=[],
                reasoning="未召回任何记忆",
            )

        if not memory_relevance_prompt:
            return self._fallback_keyword_match(recalled_memories, expected_keywords)

        memories_text = "\n\n".join(
            [f"[{m.get('role', 'unknown')}] {m.get('content', '')}" for m in recalled_memories]
        )

        prompt = MEMORY_HIT_EVALUATION_PROMPT.format(
            query=query,
            recalled_memories=memories_text,
            memory_relevance_prompt=memory_relevance_prompt,
        )

        try:
            llm = self._get_llm_client()
            response = llm.chat(prompt, temperature=0.1, max_tokens=500)

            result = self._parse_llm_response(response)

            if result:
                return MemoryHitResult(
                    is_hit=result.get("is_hit", False),
                    relevance=result.get("relevance", "not_relevant"),
                    confidence=result.get("confidence", 0.5),
                    matched_keywords=result.get("matched_keywords", []),
                    reasoning=result.get("reasoning", ""),
                )
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}, falling back to keyword match")

        return self._fallback_keyword_match(recalled_memories, expected_keywords)

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 响应为 JSON。"""
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(response)
        except json.JSONDecodeError:
            return None

    def _fallback_keyword_match(
        self,
        recalled_memories: List[Dict[str, Any]],
        expected_keywords: List[str],
    ) -> MemoryHitResult:
        """降级：关键词匹配。"""
        if not expected_keywords:
            return MemoryHitResult(
                is_hit=True,
                relevance="relevant",
                confidence=0.5,
                matched_keywords=[],
                reasoning="无期望关键词，默认命中",
            )

        recalled_content = " ".join([m.get("content", "") for m in recalled_memories]).lower()

        matched = [kw for kw in expected_keywords if kw.lower() in recalled_content]

        coverage = len(matched) / len(expected_keywords) if expected_keywords else 0

        return MemoryHitResult(
            is_hit=coverage > 0,
            relevance="relevant"
            if coverage >= 0.8
            else "partially_relevant"
            if coverage >= 0.3
            else "not_relevant",
            confidence=coverage,
            matched_keywords=matched,
            reasoning=f"关键词覆盖率: {coverage:.1%}",
        )

    def save_to_session(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """保存对话到会话缓存。"""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append(
            {
                "role": role,
                "content": content,
            }
        )

    def recall_from_session(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """从会话缓存召回记忆。"""
        memories = self._sessions.get(session_id, [])
        return memories[-limit:] if limit else memories

    async def run_evaluation(
        self,
        test_cases: List[MemoryTestCase],
    ) -> Dict[str, Any]:
        """运行完整评估。"""
        sorted_cases = sorted(test_cases, key=lambda x: (x.scenario_id, x.turn))

        results: List[EvaluationResult] = []
        long_term_rank_results: List[int] = []

        for tc in sorted_cases:
            session_id = f"scenario_{tc.scenario_id}"

            recalled_memories = []
            recall_success = True
            first_relevant_rank = None

            if tc.turn > 1 and tc.memory_type:
                if tc.memory_type == "short_term":
                    recalled_memories = self.recall_from_session(
                        session_id,
                        limit=self._config.get("short_term_limit", 5),
                    )
                elif tc.memory_type == "long_term":
                    recalled_memories = self._long_term_store.recall_by_query(
                        query=tc.query,
                        limit=self._config.get("long_term_limit", 5),
                    )
                    recall_success = len(recalled_memories) > 0

                    first_relevant_rank = self._find_first_relevant_rank(
                        recalled_memories,
                        tc.expected_keywords,
                    )
                    if first_relevant_rank:
                        long_term_rank_results.append(first_relevant_rank)

            hit_result = await self.evaluate_memory_hit(
                query=tc.query,
                recalled_memories=recalled_memories,
                memory_relevance_prompt=tc.memory_relevance_prompt or "",
                expected_keywords=tc.expected_keywords,
            )

            results.append(
                EvaluationResult(
                    scenario_id=tc.scenario_id,
                    turn=tc.turn,
                    query=tc.query,
                    memory_type=tc.memory_type,
                    is_hit=hit_result.is_hit,
                    relevance=hit_result.relevance,
                    confidence=hit_result.confidence,
                    matched_keywords=hit_result.matched_keywords,
                    reasoning=hit_result.reasoning,
                    recalled_count=len(recalled_memories),
                    recall_success=recall_success,
                    first_relevant_rank=first_relevant_rank,
                    expected_keywords=tc.expected_keywords,
                )
            )

            self.save_to_session(session_id, "user", tc.query)
            self.save_to_session(session_id, "assistant", tc.ground_truth)

            if tc.memory_type == "long_term" or tc.turn == 1:
                self._long_term_store.save(
                    session_id=session_id,
                    role="user",
                    content=tc.query,
                )
                self._long_term_store.save(
                    session_id=session_id,
                    role="assistant",
                    content=tc.ground_truth,
                )

        return self._calculate_metrics(results, long_term_rank_results)

    def _find_first_relevant_rank(
        self,
        recalled_memories: List[Dict[str, Any]],
        expected_keywords: List[str],
    ) -> Optional[int]:
        """找到第一个相关记忆的排名（用于 MRR 计算）。"""
        if not expected_keywords:
            return None

        for rank, memory in enumerate(recalled_memories, start=1):
            content = memory.get("content", "").lower()
            for keyword in expected_keywords:
                if keyword.lower() in content:
                    return rank

        return None

    def _calculate_metrics(
        self,
        results: List[EvaluationResult],
        long_term_rank_results: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """计算评估指标。"""
        memory_queries = [r for r in results if r.memory_type is not None and r.turn > 1]

        short_term = [r for r in memory_queries if r.memory_type == "short_term"]
        long_term = [r for r in memory_queries if r.memory_type == "long_term"]

        def calc_short_term_metrics(group: List[EvaluationResult]) -> Dict[str, Any]:
            """短期记忆评估指标。"""
            if not group:
                return {
                    "hit_rate": 0.0,
                    "avg_confidence": 0.0,
                    "avg_recalled": 0.0,
                    "total": 0,
                    "note": "召回成功率不适用（滑动窗口必然 100%）",
                }

            return {
                "hit_rate": round(sum(r.is_hit for r in group) / len(group), 4),
                "avg_confidence": round(sum(r.confidence for r in group) / len(group), 4),
                "avg_recalled": round(sum(r.recalled_count for r in group) / len(group), 2),
                "total": len(group),
                "note": "召回成功率不适用（滑动窗口必然 100%）",
            }

        def calc_long_term_metrics(
            group: List[EvaluationResult],
            rank_results: Optional[List[int]] = None,
        ) -> Dict[str, Any]:
            """长期记忆评估指标（包含 MRR 和 Hit@K）。"""
            if not group:
                return {
                    "recall_success_rate": 0.0,
                    "hit_rate": 0.0,
                    "avg_confidence": 0.0,
                    "avg_recalled": 0.0,
                    "mrr": 0.0,
                    "hit_at_1": 0.0,
                    "hit_at_3": 0.0,
                    "hit_at_5": 0.0,
                    "total": 0,
                }

            recall_success = sum(1 for r in group if r.recall_success)
            recall_success_rate = recall_success / len(group) if group else 0

            mrr = 0.0
            hit_at_1 = 0.0
            hit_at_3 = 0.0
            hit_at_5 = 0.0

            if rank_results:
                mrr = sum(1.0 / r for r in rank_results) / len(group)
                hit_at_1 = sum(1 for r in rank_results if r == 1) / len(group)
                hit_at_3 = sum(1 for r in rank_results if r <= 3) / len(group)
                hit_at_5 = sum(1 for r in rank_results if r <= 5) / len(group)

            return {
                "recall_success_rate": round(recall_success_rate, 4),
                "hit_rate": round(sum(r.is_hit for r in group) / len(group), 4),
                "avg_confidence": round(sum(r.confidence for r in group) / len(group), 4),
                "avg_recalled": round(sum(r.recalled_count for r in group) / len(group), 2),
                "mrr": round(mrr, 4),
                "hit_at_1": round(hit_at_1, 4),
                "hit_at_3": round(hit_at_3, 4),
                "hit_at_5": round(hit_at_5, 4),
                "total": len(group),
            }

        by_relevance = {"relevant": 0, "partially_relevant": 0, "not_relevant": 0}
        for r in memory_queries:
            rel_key = r.relevance if r.relevance in by_relevance else "not_relevant"
            by_relevance[rel_key] = by_relevance.get(rel_key, 0) + 1

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "overall_hit_rate": round(
                    sum(r.is_hit for r in memory_queries) / len(memory_queries), 4
                )
                if memory_queries
                else 0.0,
                "total_memory_queries": len(memory_queries),
            },
            "by_type": {
                "short_term": calc_short_term_metrics(short_term),
                "long_term": calc_long_term_metrics(long_term, long_term_rank_results),
            },
            "by_relevance": by_relevance,
            "details": [asdict(r) for r in results],
        }


MEMORY_HIT_EVALUATION_PROMPT = """你是一个记忆系统评估专家。

## 任务
判断召回的记忆内容是否能够支持回答当前问题。

## 当前问题
{query}

## 召回的记忆
{recalled_memories}

## 期望召回的内容特征
{memory_relevance_prompt}

## 评估标准

### 相关性等级
1. **relevant（相关）**: 召回的记忆完整包含回答问题所需的关键信息
2. **partially_relevant（部分相关）**: 召回的记忆包含部分有用信息，但不够完整
3. **not_relevant（不相关）**: 召回的记忆与当前问题无关

### 置信度
- 0.9-1.0: 非常确定
- 0.7-0.9: 比较确定
- 0.5-0.7: 有一定把握
- < 0.5: 不太确定

## 输出格式（JSON）
```json
{{
    "is_hit": true或false,
    "relevance": "relevant"或"partially_relevant"或"not_relevant",
    "confidence": 0.0到1.0之间的数值,
    "matched_keywords": ["匹配到的关键词列表"],
    "reasoning": "简要说明判断依据（1-2句话）"
}}
```

请严格按照 JSON 格式输出，不要包含其他内容。"""


def load_test_cases(file_path: str) -> List[MemoryTestCase]:
    """加载测试用例。"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("test_cases", data) if isinstance(data, dict) else data
    return [MemoryTestCase.from_dict(c) for c in cases]


def save_report(report: Dict[str, Any], output_path: str) -> None:
    """保存评估报告。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved to: {output_path}")


def print_report(report: Dict[str, Any]) -> None:
    """打印评估报告摘要。"""
    print("\n" + "=" * 60)
    print("Memory System Evaluation Report")
    print("=" * 60)

    summary = report.get("summary", {})
    by_type = report.get("by_type", {})
    by_relevance = report.get("by_relevance", {})

    print(f"\nTimestamp: {report.get('timestamp', 'N/A')}")
    print(f"\nOverall Hit Rate: {summary.get('overall_hit_rate', 0):.2%}")
    print(f"Total Memory Queries: {summary.get('total_memory_queries', 0)}")

    print("\n" + "-" * 40)
    print("By Memory Type:")
    print("-" * 40)
    print(f"{'Type':<15} {'Hit Rate':<12} {'Avg Conf':<12} {'Avg Recalled':<12} {'Total':<8}")
    print("-" * 60)

    for mem_type, metrics in by_type.items():
        if mem_type == "short_term":
            print(
                f"{mem_type:<15} {metrics.get('hit_rate', 0):.2%}      "
                f"{metrics.get('avg_confidence', 0):.2f}        "
                f"{metrics.get('avg_recalled', 0):.1f}         "
                f"{metrics.get('total', 0)}"
            )
        else:
            print(
                f"{mem_type:<15} {metrics.get('hit_rate', 0):.2%}      "
                f"{metrics.get('avg_confidence', 0):.2f}        "
                f"{metrics.get('avg_recalled', 0):.1f}         "
                f"{metrics.get('total', 0)}"
            )

    long_term_metrics = by_type.get("long_term", {})
    if long_term_metrics.get("total", 0) > 0:
        print("\n" + "-" * 40)
        print("Long-term Memory Ranking Quality:")
        print("-" * 40)
        print(f"  Recall Success Rate: {long_term_metrics.get('recall_success_rate', 0):.2%}")
        print(f"  MRR:                 {long_term_metrics.get('mrr', 0):.4f}")
        print(f"  Hit@1:               {long_term_metrics.get('hit_at_1', 0):.2%}")
        print(f"  Hit@3:               {long_term_metrics.get('hit_at_3', 0):.2%}")
        print(f"  Hit@5:               {long_term_metrics.get('hit_at_5', 0):.2%}")

    print("\n" + "-" * 40)
    print("By Relevance:")
    print("-" * 40)
    for rel, count in by_relevance.items():
        print(f"  {rel}: {count}")

    print("\n" + "=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Memory System Evaluator v2")
    parser.add_argument("--test-set", required=True, help="Path to test cases JSON file")
    parser.add_argument("--output", "-o", help="Output report path")
    parser.add_argument("--config", help="Evaluation config file")
    parser.add_argument("--report", help="Display existing report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.report:
        with open(args.report, "r", encoding="utf-8") as f:
            report = json.load(f)
        print_report(report)
        return

    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    logger.info(f"Loading test cases from: {args.test_set}")
    test_cases = load_test_cases(args.test_set)
    logger.info(f"Loaded {len(test_cases)} test cases")

    evaluator = MemoryEvaluatorV2(config=config)

    logger.info("Running evaluation...")
    report = await evaluator.run_evaluation(test_cases)

    if args.output:
        save_report(report, args.output)
    else:
        default_output = PROJECT_ROOT / "results" / "memory_evaluation_report.json"
        save_report(report, str(default_output))

    print_report(report)

    if args.verbose:
        print("\nDetailed Results:")
        print("-" * 80)
        for detail in report.get("details", []):
            if detail.get("memory_type"):
                print(
                    f"Scenario {detail['scenario_id']} Turn {detail['turn']}: "
                    f"{'HIT' if detail['is_hit'] else 'MISS'} "
                    f"({detail['relevance']}, conf={detail['confidence']:.2f})"
                )
                print(f"  Query: {detail['query'][:60]}...")
                print(f"  Reason: {detail['reasoning']}")
                print()


if __name__ == "__main__":
    asyncio.run(main())
