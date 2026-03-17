# Agentic RAG Assistant 评估体系

> 本文档介绍项目的双层评估架构、数据收集方式、评估指标及使用方法。

## 目录

- [整体架构](#整体架构)
- [层级一：Agent 实时评估](#层级一agent-实时评估)
- [层级二：Ragas 离线评估](#层级二ragas-离线评估)
- [数据获取](#数据获取)
- [评估指标详解](#评估指标详解)
- [评估执行](#评估执行)
- [评估报告](#评估报告)
- [典型工作流](#典型工作流)

---

## 整体架构

本项目采用**双层评估架构**，分离实时决策与质量评估职责：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        评估体系（双层）                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │         层级1: Agent 实时评估（流程控制）                       │  │
│  │                                                               │  │
│  │   触发: 每次检索后自动执行                                     │  │
│  │   位置: src/agent/nodes/evaluator.py                          │  │
│  │   目的: 决定是否改写查询重试                                   │  │
│  │   方法: 基于 RAG 返回的 chunk scores 阈值判断                  │  │
│  │   延迟: <10ms，不调用 LLM                                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │         层级2: Ragas 离线评估（质量评估）                       │  │
│  │                                                               │  │
│  │   触发: 手动执行 CLI 命令                                      │  │
│  │   位置: src/evaluation/ragas_evaluator.py                     │  │
│  │   目的: 评估端到端 RAG 质量                                    │  │
│  │   方法: LLM-as-Judge（需要调用 LLM）                           │  │
│  │   延迟: 1-3秒/指标                                             │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 为什么需要双层评估？

| 问题 | 答案 |
|------|------|
| RAG 服务返回的 score 不是评估吗？ | score 是单个 chunk 的相关性评分，不是整体检索质量的评估 |
| 为什么不在实时评估中调用 LLM？ | 实时决策需要低延迟（<10ms），LLM 调用延迟高（1-3秒/指标） |
| 两层评估是否重复？ | 不重复。实时评估做流程控制决策，离线评估做质量度量 |

---

## 层级一：Agent 实时评估

### 职责

实时评估是一个**决策节点**，不是质量评估。其职责是：

1. 判断检索结果是否足够回答用户问题
2. 决定是否触发查询改写重试

### 实现位置

```
src/agent/nodes/evaluator.py
```

### 评估逻辑

```python
def evaluate_retrieval(
    query: str, 
    chunks: List[Chunk], 
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    基于 RAG 返回的 scores 判断检索结果是否足够。
    
    简化逻辑：
    1. 计算 top5 chunk 的平均 score
    2. 检查数量是否足够（>=3）
    3. 返回决策结果
    """
    if not chunks:
        return {
            "is_sufficient": False,
            "reason": "未检索到任何相关文档",
            "score": 0.0,
        }
    
    avg_score = sum(c.score for c in chunks[:5]) / min(5, len(chunks))
    is_sufficient = avg_score >= threshold and len(chunks) >= 3
    
    return {
        "is_sufficient": is_sufficient,
        "reason": f"检索结果{'充分' if is_sufficient else '不足'} (avg_score={avg_score:.2f})",
        "score": avg_score,
    }
```

### 配置参数

在 `config/settings.yaml` 中配置：

```yaml
agent:
  sufficiency_threshold: 0.7   # 检索充分性阈值
  max_rewrite_attempts: 2      # 最大改写次数
```

---

## 层级二：Ragas 离线评估

### 职责

Ragas 离线评估使用 **LLM-as-Judge** 方式，评估端到端 RAG 质量用于：

- 系统调优
- 回归测试
- 模型对比

### 与 RAG MCP Server 的关键区别

| 对比项 | RAG MCP Server | 本项目 |
|--------|----------------|--------|
| **Answer 来源** | chunk 拼接（fallback） | **LLM 生成** |
| **Faithfulness** | 失真（答案=context，恒高分） | **有意义**（检测幻觉） |
| **Answer Relevancy** | 失真（拼接文本未必回答问题） | **有意义**（评估答案质量） |

**核心问题**：RAG Server 的 Ragas 评估直接拼接 chunks 作为 answer，导致：
- Faithfulness 恒高（答案就是 context 的子集）
- Answer Relevancy 失真（拼接文本未必回答用户问题）

**解决方案**：在 Agent 端执行 Ragas 评估，使用 LLM 生成的真实答案。

### 实现位置

```
src/evaluation/ragas_evaluator.py  # 核心评估逻辑
src/evaluation/data_collector.py   # 数据收集
scripts/evaluate_ragas.py          # CLI 入口
```

---

## 数据获取

### 数据来源

| 来源 | 方式 | 适用场景 |
|------|------|----------|
| **Session 收集** | 运行时自动收集问答对 | 开发测试阶段 |
| **LangFuse 导出** | 从 LangFuse 追踪中提取 | 生产环境 |
| **手动标注** | 创建 JSON 测试集 | 黄金测试集 |

### 数据格式

测试集采用 JSON 格式，存储在 `eval_data/` 目录：

```json
{
  "test_cases": [
    {
      "query": "用户问题",
      "contexts": ["检索到的 chunk1", "检索到的 chunk2"],
      "answer": "LLM 生成的答案",
      "reference_answer": "人工标注的标准答案（可选）",
      "metadata": {
        "session_id": "xxx",
        "timestamp": "2024-01-15T10:30:00Z"
      }
    }
  ],
  "created_at": "2024-01-15T10:00:00Z",
  "source": "manual",
  "description": "测试集描述"
}
```

### 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `query` | 是 | 用户问题 |
| `contexts` | 是 | 检索到的上下文列表 |
| `answer` | 是 | LLM 生成的答案 |
| `reference_answer` | 否 | 人工标注的标准答案 |
| `metadata` | 否 | 元数据（session_id、timestamp 等） |

### 收集方式

#### 方式一：Session 收集

在 Agent 运行过程中自动收集：

```python
from src.evaluation import DataCollector

collector = DataCollector()

# 在查询完成后添加数据
collector.add_session_data(
    query="什么是 RAG？",
    contexts=["RAG 是一种技术...", "RAG 的主要组件..."],
    answer="RAG 是检索增强生成...",
    metadata={"session_id": "session_001"}
)

# 获取收集的数据
test_set = collector.get_session_data()

# 保存到文件
test_set.save("eval_data/raw/session_001.json")
```

#### 方式二：从 LangFuse 收集

```python
from src.evaluation import create_data_collector
from src.core.config import load_settings

settings = load_settings()
collector = create_data_collector(settings.langfuse)

# 收集最近 7 天的追踪数据
test_set = collector.collect_from_langfuse(
    days=7,
    limit=100,
    tags=["production"]  # 可选：按标签过滤
)

test_set.save("eval_data/raw/langfuse_export.json")
```

#### 方式三：导入 JSON 文件

```python
from src.evaluation import TestSet

# 加载已有测试集
test_set = TestSet.load("eval_data/annotated/test_set.json")
```

---

## 评估指标详解

### 三大核心指标

| 指标 | 英文名 | 计算方式 | 意义 | 取值范围 |
|------|--------|----------|------|----------|
| **忠实度** | Faithfulness | LLM 判断答案中的每个 claim 是否能从 contexts 推导 | 答案是否"幻觉" | 0-1 |
| **答案相关性** | Answer Relevancy | LLM 生成可能问题，计算与原问题的语义相似度 | 答案是否"答非所问" | 0-1 |
| **上下文精确度** | Context Precision | LLM 判断每个 context 是否与问题相关 | 检索是否"精准" | 0-1 |

### Faithfulness（忠实度）

**定义**：衡量生成的答案是否忠实于检索到的上下文，不包含"幻觉"。

**计算方式**：
1. LLM 从答案中提取所有 claims（事实陈述）
2. 对每个 claim，判断是否能从 contexts 推导出来
3. 计算可推导 claims 的比例

**示例**：
```
Context: "RAG 是一种结合检索和生成的技术，由 Meta 在 2020 年提出。"
Answer: "RAG 是一种技术，由 Google 在 2019 年提出。"

问题: "Google" 和 "2019" 与 Context 不符
结果: Faithfulness = 0.33（3个claims中1个正确）
```

**解读**：
- `>= 0.8`：优秀，答案高度忠实于上下文
- `0.6-0.8`：良好，存在少量不准确
- `< 0.6`：需要关注，可能存在明显幻觉

### Answer Relevancy（答案相关性）

**定义**：衡量答案是否真正回答了用户的问题。

**计算方式**：
1. LLM 基于答案反向生成多个可能的问题
2. 计算这些问题与原始问题的语义相似度
3. 取平均值

**示例**：
```
Question: "什么是 RAG？"
Answer: "RAG 是 Retrieval-Augmented Generation 的缩写，是一种结合检索和生成的技术..."

LLM 反向生成的问题：
- "RAG 代表什么？" （相似度 0.9）
- "RAG 技术是如何工作的？" （相似度 0.85）

结果: Answer Relevancy = 0.875
```

**解读**：
- `>= 0.8`：答案高度相关
- `0.6-0.8`：答案部分相关
- `< 0.6`：答案可能跑题

### Context Precision（上下文精确度）

**定义**：衡量检索到的上下文是否与问题相关，是否包含噪声。

**计算方式**：
1. LLM 对每个 context 判断是否与问题相关
2. 计算相关 context 的比例，考虑排序权重

**示例**：
```
Question: "RAG 的主要组件有哪些？"
Contexts:
  1. "RAG 的主要组件包括检索器、生成器..." （相关）
  2. "向量数据库用于存储嵌入向量..." （相关）
  3. "Python 是一种编程语言..." （不相关）

结果: Context Precision = 0.67（3个context中2个相关）
```

**解读**：
- `>= 0.8`：检索精准，噪声少
- `0.6-0.8`：检索质量尚可
- `< 0.6`：检索噪声多，需要优化

---

## 评估执行

### 安装依赖

```bash
pip install ragas datasets
```

### CLI 命令

```bash
# 基础评估
python scripts/evaluate_ragas.py --test-set eval_data/annotated/test_set.json

# 指定输出目录
python scripts/evaluate_ragas.py --test-set test_set.json --output-dir eval_reports/

# 从 LangFuse 收集并评估
python scripts/evaluate_ragas.py --collect-from-langfuse --days 7

# 仅评估指定指标
python scripts/evaluate_ragas.py --test-set test_set.json \
    --metrics faithfulness answer_relevancy

# JSON 输出（便于管道处理）
python scripts/evaluate_ragas.py --test-set test_set.json --json

# 详细输出
python scripts/evaluate_ragas.py --test-set test_set.json --verbose
```

### 命令参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--test-set` | 测试集 JSON 文件路径 | - |
| `--output-dir` | 报告输出目录 | `eval_reports` |
| `--metrics` | 要评估的指标 | 全部 |
| `--collect-from-langfuse` | 从 LangFuse 收集数据 | - |
| `--days` | LangFuse 回溯天数 | 7 |
| `--limit` | LangFuse 最大条数 | 100 |
| `--json` | JSON 格式输出 | - |
| `--verbose` | 详细输出 | - |

### 代码调用

```python
from src.evaluation import RagasEvaluator, TestSet
from src.core.config import load_settings

# 加载配置
settings = load_settings()

# 创建评估器
evaluator = RagasEvaluator(
    llm_config=settings.llm,
    metrics=["faithfulness", "answer_relevancy", "context_precision"]
)

# 加载测试集
test_set = TestSet.load("eval_data/annotated/test_set.json")

# 执行评估
report = evaluator.evaluate_batch(
    [tc.to_dict() for tc in test_set.test_cases]
)

# 查看结果
print(f"总测试数: {report.total_cases}")
print(f"总耗时: {report.total_elapsed_ms:.0f}ms")
print("聚合指标:")
for metric, value in report.aggregate_metrics.items():
    print(f"  {metric}: {value:.4f}")
```

---

## 评估报告

### 输出位置

```
eval_reports/
├── 2024-01-15/
│   └── report.json          # 按日期归档
├── 2024-01-16/
│   └── report.json
└── latest.json              # 最新报告
```

### 报告格式

```json
{
  "evaluator": "RagasEvaluator",
  "test_set_path": "eval_data/annotated/test_set.json",
  "total_cases": 3,
  "aggregate_metrics": {
    "faithfulness": 0.8500,
    "answer_relevancy": 0.7800,
    "context_precision": 0.9200
  },
  "per_case_results": [
    {
      "query": "什么是 RAG？",
      "metrics": {
        "faithfulness": 0.9000,
        "answer_relevancy": 0.8500,
        "context_precision": 0.9500
      },
      "contexts": ["RAG 是一种技术..."],
      "answer": "RAG 是检索增强生成...",
      "elapsed_ms": 2345.6,
      "error": null
    }
  ],
  "total_elapsed_ms": 7234.5
}
```

### 控制台输出示例

```
============================================================
  RAGAS EVALUATION REPORT
============================================================
  Evaluator: RagasEvaluator
  Test Set:  eval_data/annotated/test_set.json
  Test Cases: 3
  Total Time: 7234 ms

------------------------------------------------------------
  AGGREGATE METRICS
------------------------------------------------------------
  answer_relevancy      ██████████████░░░░░░ 0.7800
  context_precision     ███████████████████░░ 0.9200
  faithfulness          █████████████████░░░░ 0.8500

------------------------------------------------------------
  PER-CASE RESULTS
------------------------------------------------------------

  [1] ✓ 什么是 RAG？它有哪些主要组件？
      answer_relevancy: 0.8500
      context_precision: 0.9500
      faithfulness: 0.9000
      Time: 2345 ms

  [2] ✓ LangGraph 和 LangChain 有什么区别？
      answer_relevancy: 0.7200
      context_precision: 0.8800
      faithfulness: 0.8000
      Time: 2456 ms

  [3] ✓ 如何评估 RAG 系统的质量？
      answer_relevancy: 0.7700
      context_precision: 0.9300
      faithfulness: 0.8500
      Time: 2433 ms

============================================================
  Report saved to: eval_reports/2024-01-15/report.json
============================================================
```

---

## 典型工作流

### 日常评估流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         评估工作流                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 收集数据                                                         │
│     ├── 运行 Agent 收集问答历史                                      │
│     ├── 或从 LangFuse 导出追踪                                       │
│     └── 保存到 eval_data/raw/                                        │
│                                                                      │
│  2. 标注/整理                                                        │
│     ├── 筛选高质量问答对                                             │
│     ├── 可选添加 reference_answer                                    │
│     └── 保存到 eval_data/annotated/test_set.json                     │
│                                                                      │
│  3. 执行评估                                                         │
│     └── python scripts/evaluate_ragas.py --test-set test_set.json    │
│                                                                      │
│  4. 分析结果                                                         │
│     ├── 查看 aggregate_metrics 整体质量                              │
│     ├── 识别低分案例，分析原因                                       │
│     └── 调整检索/生成策略                                            │
│                                                                      │
│  5. 迭代优化                                                         │
│     └── 修改配置后重新评估                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 问题诊断指南

| 现象 | 可能原因 | 优化方向 |
|------|----------|----------|
| Faithfulness 低 | 答案包含上下文没有的信息 | 优化生成 prompt，减少幻觉 |
| Answer Relevancy 低 | 答案跑题或信息不足 | 改进检索策略，增加相关上下文 |
| Context Precision 低 | 检索噪声多 | 调整检索参数，增加 reranker |

### 配置调优

在 `config/settings.yaml` 中调整：

```yaml
agent:
  retrieval_top_k: 10          # 检索数量
  sufficiency_threshold: 0.7   # 充分性阈值

llm:
  temperature: 0.0             # 降低温度减少随机性
```

---

## 文件索引

| 文件 | 说明 |
|------|------|
| `src/evaluation/ragas_evaluator.py` | Ragas 评估核心逻辑 |
| `src/evaluation/data_collector.py` | 数据收集器 |
| `src/evaluation/metrics.py` | Agent 自定义指标 |
| `scripts/evaluate_ragas.py` | 评估 CLI 脚本 |
| `config/evaluation.yaml` | 评估配置 |
| `eval_data/annotated/test_set.json` | 示例测试集 |
| `eval_reports/` | 评估报告目录 |

---

## 参考资料

- [Ragas 官方文档](https://docs.ragas.io/)
- [DEV_SPEC.md - 评估体系架构](../DEV_SPEC.md#9-评估体系架构)
- [LangFuse 文档](https://langfuse.com/docs)