"""Memory 测试数据集生成器。

生成用于评估长期记忆系统的测试用例数据集。

支持两种模式：
1. JSON 模式：生成 JSON 格式的测试用例（原有模式）
2. ChromaDB 模式：将预置记忆导入到 ChromaDB 并生成测试查询

Usage:
    # JSON 模式
    python scripts/dataset/build_memory_test_dataset.py --output memory_test_cases.json

    # ChromaDB 模式
    python scripts/dataset/build_memory_test_dataset.py --mode chromadb --input preset_memories.json

    # 完整测试流程
    python scripts/dataset/build_memory_test_dataset.py --mode chromadb --input preset_memories.json --test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset.test_case_schema import MemoryTestCase


class MemoryTestDatasetGenerator:
    """Memory 系统测试数据集生成器。"""

    def __init__(self) -> None:
        self.knowledge_templates = {
            "报销": "报销流程：1.提交发票 2.领导审批 3.财务审核 4.出纳打款（3-5 工作日）",
            "请假": "请假制度：年假 5-15 天，病假需医院证明，事假无薪",
            "入职": "入职流程：签合同→领设备→培训→试用期 3 个月→转正考核",
            "考勤": "考勤规定：9:00-18:00，迟到扣 100 元/次，月累计 3 次警告",
        }

        self.preference_templates = [
            "用户偏好：回答要简洁，不要废话",
            "用户偏好：喜欢表格形式展示",
            "用户偏好：需要详细解释和示例",
            "用户偏好：所有输出用 PDF 格式",
            "用户偏好：优先说结论，再说原因",
        ]

        self.fact_templates = [
            "用户部门：财务部",
            "用户级别：经理",
            "用户入职时间：2025 年 1 月",
        ]

    def generate_enhance_case(self, case_id: int) -> MemoryTestCase:
        """生成记忆增强类用例。

        测试系统是否能够结合用户偏好回答问题。
        """
        topic = random.choice(list(self.knowledge_templates.keys()))
        preference = random.choice(self.preference_templates)

        return MemoryTestCase(
            id=f"TC_ENHANCE_{case_id:03d}",
            scenario="enhance",
            priority="P0",
            user_id=f"user_{random.randint(1, 10):03d}",
            cross_session=False,
            user_memories=[{"content": preference, "type": "preference"}],
            knowledge_context=self.knowledge_templates[topic],
            query=f"{topic}相关政策是什么？",
            expected_behavior=f"结合偏好呈现{topic}政策",
            should_use_memory=True,
            should_use_knowledge=True,
            memory_keywords=["简洁"]
            if "简洁" in preference
            else ["表格"]
            if "表格" in preference
            else [],
            knowledge_keywords=[topic],
            evaluation_criteria=["准确引用知识库", "体现用户偏好"],
        )

    def generate_cross_session_case(self, case_id: int) -> MemoryTestCase:
        """生成跨会话用例。

        测试系统是否能够跨会话召回记忆。
        """
        topic = random.choice(list(self.knowledge_templates.keys()))
        preference = random.choice(self.preference_templates)
        fact = random.choice(self.fact_templates)

        return MemoryTestCase(
            id=f"TC_CROSS_{case_id:03d}",
            scenario="cross_session",
            priority="P0",
            user_id=f"user_{random.randint(1, 10):03d}",
            cross_session=True,
            user_memories=[
                {"content": preference, "type": "preference"},
                {"content": fact, "type": "fact"},
            ],
            knowledge_context=self.knowledge_templates[topic],
            query=f"上次讨论的{topic}政策，现在还有效吗？",
            expected_behavior="结合历史记忆回答",
            should_use_memory=True,
            should_use_knowledge=True,
            memory_keywords=["上次", preference.split("：")[-1]],
            knowledge_keywords=[topic],
            evaluation_criteria=["召回历史记忆", "准确回答当前问题"],
        )

    def generate_conflict_case(self, case_id: int) -> MemoryTestCase:
        """生成冲突检测用例。

        测试系统是否能检测到用户问题与知识库的冲突。
        """
        return MemoryTestCase(
            id=f"TC_CONFLICT_{case_id:03d}",
            scenario="conflict",
            priority="P1",
            user_id=f"user_{random.randint(1, 10):03d}",
            cross_session=False,
            user_memories=[{"content": "我记得报销不需要审批", "type": "fact"}],
            knowledge_context=self.knowledge_templates["报销"],
            query="3000 元报销需要审批吗？",
            expected_behavior="礼貌纠正，以知识库为准",
            should_use_memory=True,
            should_use_knowledge=True,
            memory_keywords=[],
            knowledge_keywords=["均需审批", "无豁免"],
            evaluation_criteria=["明确指出需要审批", "语气礼貌", "引用规定"],
        )

    def generate_baseline_case(self, case_id: int) -> MemoryTestCase:
        """生成基线用例。

        测试系统在无记忆情况下的基本问答能力。
        """
        topic = random.choice(list(self.knowledge_templates.keys()))

        return MemoryTestCase(
            id=f"TC_BASELINE_{case_id:03d}",
            scenario="baseline",
            priority="P1",
            user_id=f"user_new_{case_id:03d}",
            cross_session=False,
            user_memories=[],
            knowledge_context=self.knowledge_templates[topic],
            query=f"{topic}相关怎么弄？",
            expected_behavior="准确回答知识库内容",
            should_use_memory=False,
            should_use_knowledge=True,
            memory_keywords=[],
            knowledge_keywords=[topic],
            evaluation_criteria=["准确引用知识库", "无幻觉"],
        )

    def generate_dataset(self, count: int = 30) -> List[MemoryTestCase]:
        """生成完整数据集。

        按比例分布:
        - enhance: 40%
        - cross_session: 30%
        - conflict: 20%
        - baseline: 10%
        """
        distribution = [
            ("enhance", int(count * 0.4), self.generate_enhance_case),
            ("cross_session", int(count * 0.3), self.generate_cross_session_case),
            ("conflict", int(count * 0.2), self.generate_conflict_case),
            (
                "baseline",
                count - int(count * 0.4) - int(count * 0.3) - int(count * 0.2),
                self.generate_baseline_case,
            ),
        ]

        cases: List[MemoryTestCase] = []
        case_id = 1

        for scenario, scenario_count, generator in distribution:
            for _ in range(scenario_count):
                cases.append(generator(case_id))
                case_id += 1

        random.shuffle(cases)
        return cases

    def save_to_json(self, cases: List[MemoryTestCase], filepath: str) -> None:
        """保存为 JSON 文件。"""
        data = {
            "metadata": {
                "total_cases": len(cases),
                "scenarios": dict(Counter(c.scenario for c in cases)),
                "description": "Memory 系统测试用例",
                "generated_at": datetime.now().isoformat(),
            },
            "test_cases": [c.to_dict() for c in cases],
        }

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ 数据集已保存: {output_path}")
        print(f"   总用例数: {len(cases)}")
        for scenario, count in data["metadata"]["scenarios"].items():
            print(f"   - {scenario}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 Memory 测试数据集")
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["json", "chromadb"],
        default="json",
        help="生成模式: json (生成 JSON 测试用例) 或 chromadb (导入到 ChromaDB)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="eval_data/memory_test_cases.json",
        help="输出文件路径 (default: eval_data/memory_test_cases.json)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="scripts/dataset/preset_memories.json",
        help="输入文件路径 (ChromaDB 模式下的预置记忆 JSON)",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=30,
        help="生成用例数量 (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子 (用于复现)",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="./data/memory_test_db",
        help="ChromaDB 持久化目录 (default: ./data/memory_test_db)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="memory_test",
        help="ChromaDB 集合名称 (default: memory_test)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="导入后立即运行测试",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        choices=["mock", "real"],
        default="mock",
        help="Embedding 提供商 (mock 用于快速测试)",
    )

    args = parser.parse_args()

    if args.mode == "chromadb":
        return _run_chromadb_mode(args)
    else:
        return _run_json_mode(args)


def _run_json_mode(args) -> int:
    """运行 JSON 模式。"""
    if args.seed is not None:
        random.seed(args.seed)

    generator = MemoryTestDatasetGenerator()
    cases = generator.generate_dataset(args.count)

    output_path = (
        PROJECT_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    )
    generator.save_to_json(cases, str(output_path))

    return 0


def _run_chromadb_mode(args) -> int:
    """运行 ChromaDB 模式。"""
    input_path = (
        PROJECT_ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    )

    if not input_path.exists():
        print(f"❌ 错误: 输入文件不存在 - {input_path}")
        return 1

    print(f"📦 加载预置记忆数据: {input_path}")

    from scripts.dataset.memory_loader import MemoryLoader

    loader = MemoryLoader(
        persist_directory=args.db_dir,
        collection_name=args.collection,
        embedding_provider=args.embedding,
    )

    with open(input_path, "r", encoding="utf-8") as f:
        preset_data = json.load(f)

    stats = loader.import_memories(preset_data, clear_existing=True)

    print(f"\n✅ 导入完成:")
    print(f"   总记忆数: {stats.get('total_memories', 0)}")
    print(f"   用户数: {stats.get('users', 0)}")
    print(f"   按类型分布:")
    for mem_type, count in stats.get("by_type", {}).items():
        print(f"     - {mem_type}: {count}")

    print(f"\n📍 ChromaDB 配置:")
    print(f"   持久化目录: {args.db_dir}")
    print(f"   集合名称: {args.collection}")

    if args.test:
        print("\n🧪 运行测试...")
        from scripts.dataset.test_memory_recall import MemoryRecallTester

        tester = MemoryRecallTester(
            persist_directory=args.db_dir,
            collection_name=args.collection,
            embedding_provider=args.embedding,
        )

        tester.run_recall_test(preset_data)
        metrics = tester.calculate_metrics()

        print("\n📊 测试结果:")
        print(f"   总查询数: {metrics.total_queries}")
        print(f"   平均召回率: {metrics.avg_recall_rate:.2%}")
        print(f"   平均延迟: {metrics.avg_query_time_ms:.2f} ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
