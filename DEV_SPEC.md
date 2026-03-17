# Agentic RAG 知识助手 - 开发规格文档

> 版本：1.1 — 完整开发规格

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 核心特点](#2-核心特点)
- [3. 技术选型](#3-技术选型)
- [4. 系统架构](#4-系统架构)
- [5. 模块设计](#5-模块设计)
- [6. 配置设计](#6-配置设计)
- [7. 项目排期](#7-项目排期)
- [8. 可扩展性](#8-可扩展性)
- [9. 评估体系架构](#9-评估体系架构)
- [10. 关键设计决策](#10-关键设计决策)
- [11. 面试亮点](#11-面试亮点)

---

## 1. 项目概述

### 1.1 项目定位

面向企业内部知识管理场景，构建基于 **Agentic RAG** 的智能知识助手。为企业员工提供对内部文档、数据库等私有知识的自然语言访问能力。

### 1.2 与现有项目的关系

- **独立仓库**：本项目作为独立 Git 仓库
- **MCP 协议连接**：通过 MCP Client 调用现有 RAG MCP Server
- **复用能力**：复用 RAG Server 的混合检索引擎，专注于 Agent 推理层

### 1.3 核心价值

```
传统 RAG:  Query → Retrieve → Generate

Agentic RAG:  Query → [Analyze → Retrieve → Evaluate → Rewrite? → Sub-query?] → Generate with Citations
```

### 1.4 设计理念

- **Agent-First**：以 Agent 为核心，检索只是工具
- **可观测性**：全链路追踪，决策过程透明
- **可扩展性**：模块化设计，易于添加新能力
- **面试导向**：代码质量高，架构清晰，便于讲解

---

## 2. 核心特点

### 2.1 Agentic RAG 能力

| 能力 | 说明 | 实现方式 |
|------|------|----------|
| **子问题拆分** | 复杂问题拆分为子问题分别检索 | LangGraph 条件分支 + Decomposition Prompt |
| **查询改写** | 根据评估结果改写查询重新检索 | Rewrite Prompt + Re-retrieve Loop |
| **引用溯源** | 回答中标注来源文档和段落 | Chunk ID 追踪 + Citation Formatter |

### 2.2 Agent 决策流程

```
用户问题
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph State Machine                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐                                           │
│  │ QUERY_       │  分析问题复杂度                            │
│  │ ANALYZER     │  ↓ 简单/复杂                               │
│  └──────┬───────┘                                           │
│         │                                                    │
│    ┌────┴────┐                                              │
│    ▼         ▼                                              │
│  简单查询   复杂查询                                         │
│    │         │                                              │
│    │    ┌────┴────┐                                         │
│    │    ▼         │                                         │
│    │  SUB_QUERY   │  子问题拆分                              │
│    │  DECOMPOSER  │  ↓ [Q1, Q2, Q3...]                      │
│    │    │         │                                         │
│    └────┼─────────┘                                         │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │ RETRIEVER    │  MCP Client 调用 RAG Server               │
│  │ (MCP Call)   │  ↓ Chunks + Scores                        │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │ RETRIEVAL_   │  评估检索结果是否充分                      │
│  │ EVALUATOR    │  ↓ Sufficient / Insufficient              │
│  └──────┬───────┘                                           │
│         │                                                    │
│    ┌────┴────┐                                              │
│    ▼         ▼                                              │
│  充分      不充分                                            │
│    │         │                                              │
│    │    ┌────┴────┐                                         │
│    │    ▼         │                                         │
│    │  QUERY_      │  改写查询                                │
│    │  REWRITER    │  ↓ 新查询                                │
│    │    │         │                                         │
│    │    └────┐    │  (最多重试 N 次)                         │
│    └─────────┼────┘                                         │
│              │                                               │
│              ▼                                               │
│  ┌──────────────┐                                           │
│  │ RESPONSE_    │  生成回答 + 引用标注                        │
│  │ GENERATOR    │  ↓ Final Answer with Citations            │
│  └──────────────┘                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 MCP 集成

作为 **MCP Client** 调用 RAG Server 的工具：

| Tool | 说明 |
|------|------|
| `query_knowledge_hub` | 混合检索，返回 Chunks |
| `list_collections` | 列出可用知识库 |
| `get_document_summary` | 获取文档摘要 |

### 2.4 评估体系

本项目的评估体系由**两个独立但互补的层级**组成：

| 层级 | 位置 | 触发方式 | 职责 | 工具 |
|------|------|----------|------|------|
| **RAG离线评估** | RAG MCP Server | Dashboard 手动触发 | 评估检索系统整体质量 | Ragas / Custom |
| **Agent实时评估** | Agent端 | 每次检索后实时 | 决定是否改写查询重试 | evaluate_node |

#### 2.4.1 两层评估的区别

```
┌─────────────────────────────────────────────────────────────────────┐
│                        完整评估体系架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │               RAG MCP Server (离线评估)                        │  │
│  │                                                               │  │
│  │   黄金测试集 + Dashboard触发                                   │  │
│  │         ↓                                                     │  │
│  │   EvalRunner → Ragas/Custom Evaluator                        │  │
│  │         ↓                                                     │  │
│  │   输出: 评估报告 (faithfulness, answer_relevancy, hit_rate)   │  │
│  │                                                               │  │
│  │   用途: 系统调优、回归测试、质量监控                            │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              │ query_knowledge_hub                  │
│                              │ 返回: chunks + scores               │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │               Agent端 (实时决策)                               │  │
│  │                                                               │  │
│  │   retrieve → evaluate_node → sufficient?                      │  │
│  │                    │           │                              │  │
│  │                    │      ┌────┴────┐                         │  │
│  │                    │      ▼         ▼                         │  │
│  │                    │   generate   rewrite                      │  │
│  │                    │              │                           │  │
│  │                    └──────────────┘                           │  │
│  │                                                               │  │
│  │   输入: chunks + scores (来自RAG)                             │  │
│  │   输出: 是否充分决策                                          │  │
│  │                                                               │  │
│  │   用途: 实时判断是否改写查询重试                               │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 2.4.2 为什么需要两层评估？

| 问题 | 答案 |
|------|------|
| RAG服务返回的score不是评估吗？ | score是单个chunk的相关性评分，不是整体检索质量的评估 |
| 为什么不在RAG服务中做实时评估？ | Ragas评估需要调用LLM，延迟高(1-3秒/指标)，不适合在线检索 |
| Agent端评估是否重复？ | 不重复。Agent评估基于RAG返回的scores做实时决策，不是重新评估 |

#### 2.4.3 RAG离线评估详情

RAG MCP Server 提供的评估能力：

| 评估器 | 指标 | 输入需求 | 适用场景 |
|--------|------|----------|----------|
| CustomEvaluator | hit_rate, MRR | Ground Truth 标签 | 快速回归测试、CI/CD |
| RagasEvaluator | faithfulness, answer_relevancy, context_precision | query + response + contexts | 质量评估、模型对比 |

**注意**：Ragas评估需要`generated_answer`，但RAG检索服务不生成答案。因此需要：
- 通过Dashboard手动输入答案
- 或由Agent端生成答案后传回评估

#### 2.4.4 Agent实时评估详情

Agent端的`evaluate_node`是一个**决策节点**，不是质量评估：

```python
def evaluate_retrieval(query: str, chunks: List[Chunk], threshold: float = 0.5):
    """
    基于RAG返回的chunk scores，判断检索结果是否足够回答问题。
    
    简化逻辑：
    1. 计算top5 chunk的平均score
    2. 判断是否达到阈值且数量足够
    
    返回：
    - is_sufficient: 是否充分
    - reason: 决策原因
    """
    if not chunks:
        return {"is_sufficient": False, "reason": "未检索到任何相关文档"}
    
    avg_score = sum(c.score for c in chunks[:5]) / min(5, len(chunks))
    is_sufficient = avg_score >= threshold and len(chunks) >= 3
    
    return {"is_sufficient": is_sufficient, "score": avg_score}
```

#### 2.4.5 可观测性指标

| 维度 | 指标 | 工具 |
|------|------|------|
| **Agent可追溯性** | 决策路径、工具调用链、Token消耗 | LangFuse |
| **性能指标** | 端到端延迟、各阶段耗时 | LangFuse |

---

## 3. 技术选型

| 层级 | 技术栈 | 说明 |
|------|--------|------|
| **Agent 框架** | LangGraph | 状态机 + 条件分支，适合复杂推理流程 |
| **MCP Client** | mcp Python SDK | 标准 MCP 协议通信 |
| **LLM** | Azure OpenAI / DeepSeek / Qwen | 可插拔，复用配置模式 |
| **Web UI** | Streamlit | 快速构建 Chat 界面 |
| **评估追踪** | LangFuse | 开源 LLM 可观测平台 |
| **离线评估** | RAGAS | RAG 专用评估框架 |

### 3.1 依赖清单

```txt
# Agent
langgraph>=0.2.0
langchain-core>=0.3.0
langchain-openai>=0.2.0

# MCP
mcp>=1.0.0

# LLM Providers
openai>=1.0.0
httpx>=0.27.0

# UI
streamlit>=1.35.0

# Evaluation
langfuse>=2.0.0
ragas>=0.1.0

# Utils
pyyaml>=6.0
pydantic>=2.0
python-dotenv>=1.0.0
```

---

## 4. 系统架构

### 4.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Presentation Layer                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Streamlit Chat UI                             │   │
│  │    - 会话管理  - 流式输出  - 引用展示  - 追踪可视化               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────┤
│                             Agent Layer                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    LangGraph State Machine                       │   │
│  │  Nodes: Analyzer → Retriever → Evaluator → Rewriter → Generator │   │
│  │  State: query, sub_queries, chunks, evaluations, response       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────┤
│                          Integration Layer                               │
│  ┌───────────────────────┐  ┌───────────────────────────────────┐      │
│  │     MCP Client        │  │         LangFuse Client           │      │
│  │  - connect_server()   │  │  - trace.span()                   │      │
│  │  - call_tool()        │  │  - trace.generation()             │      │
│  └───────────┬───────────┘  └───────────────────────────────────┘      │
│              │                                                            │
├──────────────┼────────────────────────────────────────────────────────────┤
│              │  MCP Protocol (stdio/SSE)                                 │
│              ▼                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │              [External] RAG MCP Server (现有项目)                  │  │
│  │    BM25 + Dense + RRF Fusion + Cross-Encoder Rerank               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 目录结构

```
agentic-rag-assistant/
├── src/
│   ├── agent/                    # Agent 核心
│   │   ├── __init__.py
│   │   ├── graph.py              # LangGraph 状态机定义
│   │   ├── state.py              # Agent State 定义
│   │   ├── nodes/                # 各节点实现
│   │   │   ├── __init__.py
│   │   │   ├── analyzer.py       # 问题分析
│   │   │   ├── decomposer.py     # 子问题拆分
│   │   │   ├── retriever.py      # 检索调用
│   │   │   ├── evaluator.py      # 检索评估
│   │   │   ├── rewriter.py       # 查询改写
│   │   │   └── generator.py      # 回答生成
│   │   └── prompts/              # Prompt 模板
│   │       ├── __init__.py
│   │       ├── analyze.py
│   │       ├── decompose.py
│   │       ├── evaluate.py
│   │       └── rewrite.py
│   │
│   ├── mcp_client/               # MCP 客户端
│   │   ├── __init__.py
│   │   ├── client.py             # MCP 连接管理
│   │   └── tools.py              # Tool 封装
│   │
│   ├── ui/                       # Web 界面
│   │   ├── __init__.py
│   │   ├── app.py                # Streamlit 主入口
│   │   ├── components/           # UI 组件
│   │   │   ├── __init__.py
│   │   │   ├── chat.py           # 聊天界面
│   │   │   ├── citations.py      # 引用展示
│   │   │   └── trace_viewer.py   # 追踪可视化
│   │   └── styles/               # 样式文件
│   │       └── main.css
│   │
│   ├── evaluation/               # 评估模块
│   │   ├── __init__.py
│   │   ├── langfuse_client.py    # LangFuse 集成
│   │   ├── ragas_evaluator.py    # Ragas 评估器
│   │   ├── data_collector.py     # 数据收集器
│   │   └── metrics.py            # 自定义指标
│   │
│   └── core/                     # 基础设施
│       ├── __init__.py
│       ├── config.py             # 配置管理
│       ├── types.py              # 类型定义
│       └── utils.py              # 工具函数
│
├── config/
│   ├── settings.yaml             # 主配置文件
│   ├── evaluation.yaml           # 评估配置
│   └── prompts/                  # Prompt 文件
│       ├── analyze.txt
│       ├── decompose.txt
│       ├── evaluate.txt
│       └── rewrite.txt
│
├── eval_data/                    # 评估数据
│   ├── raw/                      # 原始收集数据
│   └── annotated/                # 标注后数据
│       └── test_set.json
│
├── eval_reports/                 # 评估报告
│   └── .gitkeep
│
├── tests/
│   ├── unit/
│   │   └── __init__.py
│   └── integration/
│       └── __init__.py
│
├── scripts/
│   ├── start_agent.py            # 启动 Agent CLI
│   ├── start_ui.py               # 启动 Web UI
│   └── evaluate_ragas.py         # Ragas 评估 CLI
│
├── docs/
│   └── ARCHITECTURE.md
│
├── .env.example                  # 环境变量模板
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 5. 模块设计

### 5.1 Agent State 定义

```python
# src/agent/state.py
from typing import TypedDict, List, Optional, Annotated
from operator import add

class Chunk(TypedDict):
    """检索结果块。"""
    id: str
    text: str
    score: float
    metadata: dict

class AgentState(TypedDict):
    """Agent 状态机状态。"""
    # 输入
    original_query: str
    rewritten_query: Optional[str]
    sub_queries: Annotated[List[str], add]
    
    # 检索结果
    chunks: Annotated[List[Chunk], add]
    retrieval_score: Optional[float]
    
    # 评估结果
    is_sufficient: Optional[bool]
    evaluation_reason: Optional[str]
    rewrite_count: int  # 改写次数，防止无限循环
    
    # 输出
    final_response: Optional[str]
    citations: Optional[List[dict]]
    
    # 追踪
    trace_id: Optional[str]
    decision_path: Annotated[List[str], add]
```

### 5.2 LangGraph 节点接口

每个节点遵循统一接口：

```python
from src.agent.state import AgentState

def node_name(state: AgentState) -> AgentState:
    """
    节点处理函数。
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态（部分更新，LangGraph 会自动合并）
    """
    # 1. 从 state 提取输入
    # 2. 执行节点逻辑
    # 3. 返回状态更新
    return {"key": "value"}
```

### 5.3 MCP Client 接口

```python
# src/mcp_client/client.py

class RAGMCPClient:
    """MCP Client for RAG Server."""
    
    async def connect(self) -> None:
        """连接到 RAG MCP Server。"""
        pass
    
    async def query_knowledge_hub(
        self, 
        query: str, 
        collection: str = "default",
        top_k: int = 10
    ) -> List[dict]:
        """调用 RAG Server 的检索工具。"""
        pass
    
    async def list_collections(self) -> List[str]:
        """列出可用知识库。"""
        pass
    
    async def close(self) -> None:
        """关闭连接。"""
        pass
```

### 5.4 配置管理接口

```python
# src/core/config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0

@dataclass
class AgentConfig:
    max_rewrite_attempts: int = 2
    retrieval_top_k: int = 10
    sufficiency_threshold: float = 0.7

@dataclass
class Settings:
    llm: LLMConfig
    agent: AgentConfig
    rag_server: dict
    langfuse: Optional[dict] = None
    
    @classmethod
    def load(cls, path: str = "config/settings.yaml") -> "Settings":
        """从 YAML 文件加载配置。"""
        pass
```

---

## 6. 配置设计

### 6.1 settings.yaml 结构

```yaml
# config/settings.yaml

# RAG MCP Server 配置
rag_server:
  command: ["python", "-m", "src.mcp_server.server"]
  working_dir: "../MODULAR-RAG-MCP-SERVER"  # RAG Server 项目路径
  collection: "knowledge_hub"

# Agent 配置
agent:
  max_rewrite_attempts: 2      # 最大改写次数
  retrieval_top_k: 10          # 检索数量
  sufficiency_threshold: 0.7   # 检索充分性阈值

# LLM 配置
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  api_key: "${DEEPSEEK_API_KEY}"
  base_url: "https://api.deepseek.com/v1"
  temperature: 0.0

# LangFuse 配置
langfuse:
  enabled: true
  public_key: "${LANGFUSE_PUBLIC_KEY}"
  secret_key: "${LANGFUSE_SECRET_KEY}"
  host: "https://cloud.langfuse.com"

# Streamlit UI 配置
ui:
  title: "企业知识助手"
  theme: "light"
  show_trace: true             # 显示追踪信息
```

### 6.2 环境变量

```bash
# .env.example

# LLM API Keys
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key

# LangFuse
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
```

---

## 7. 项目排期

### A. 项目基础 [A1-A3]

| 任务 | 说明 | 产物 | 状态 |
|------|------|------|------|
| A1. 项目初始化 | 创建仓库结构、依赖配置 | 目录结构、pyproject.toml | [ ] |
| A2. 配置管理 | 实现 settings.yaml 加载 | src/core/config.py | [ ] |
| A3. MCP Client | 实现 MCP 协议连接 | src/mcp_client/client.py | [ ] |

### B. Agent 核心 [B1-B7]

| 任务 | 说明 | 产物 | 状态 |
|------|------|------|------|
| B1. Agent State | 定义状态机状态 | src/agent/state.py | [ ] |
| B2. 问题分析节点 | 复杂度判断 | src/agent/nodes/analyzer.py | [ ] |
| B3. 子问题拆分节点 | Decomposition Prompt | src/agent/nodes/decomposer.py | [ ] |
| B4. 检索节点 | MCP 调用封装 | src/agent/nodes/retriever.py | [ ] |
| B5. 检索评估节点 | 充分性评估 | src/agent/nodes/evaluator.py | [ ] |
| B6. 查询改写节点 | Rewrite Prompt | src/agent/nodes/rewriter.py | [ ] |
| B7. 回答生成节点 | 带引用的回答 | src/agent/nodes/generator.py | [ ] |
| B8. LangGraph 组装 | 完整状态机 | src/agent/graph.py | [ ] |

### C. 评估体系 [C1-C5]

| 任务 | 说明 | 产物 | 状态 |
|------|------|------|------|
| C1. LangFuse 集成 | 追踪 SDK 接入 | src/evaluation/langfuse_client.py | [ ] |
| C2. Ragas 评估器 | Ragas 核心评估逻辑 | src/evaluation/ragas_evaluator.py | [ ] |
| C3. 数据收集器 | 问答历史收集与标注 | src/evaluation/data_collector.py | [ ] |
| C4. 评估 CLI | 离线评估脚本 | scripts/evaluate_ragas.py | [ ] |
| C5. 自定义指标 | Agent 特有指标 | src/evaluation/metrics.py | [ ] |

### D. Web UI [D1-D3]

| 任务 | 说明 | 产物 | 状态 |
|------|------|------|------|
| D1. Chat 界面 | 基础聊天组件 | src/ui/components/chat.py | [ ] |
| D2. 引用展示 | Citation 组件 | src/ui/components/citations.py | [ ] |
| D3. 追踪可视化 | 决策路径展示 | src/ui/components/trace_viewer.py | [ ] |

### E. 测试与文档 [E1-E3]

| 任务 | 说明 | 产物 | 状态 |
|------|------|------|------|
| E1. 单元测试 | 各节点测试 | tests/unit/ | [ ] |
| E2. 集成测试 | 端到端测试 | tests/integration/ | [ ] |
| E3. 文档完善 | README + 架构文档 | docs/ | [ ] |

---

## 8. 可扩展性

### 8.1 Agent 能力扩展点

| 扩展方向 | 说明 | 实现方式 |
|----------|------|----------|
| **工具扩展** | 新增 Tool Calling | 在 LangGraph 中添加新节点 |
| **记忆扩展** | 对话历史、用户偏好 | 添加 Memory 节点 |
| **多 Agent 协作** | Agent 间通信 | 使用 LangGraph 的 subgraph |

### 8.2 评估扩展点

| 扩展方向 | 说明 | 实现方式 |
|----------|------|----------|
| **A/B Testing** | 对比不同策略 | LangFuse Sessions |
| **在线学习** | 用户反馈优化 | 添加 Feedback 节点 |

---

## 9. 评估体系架构

### 9.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           完整评估体系                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    RAG MCP Server                                │   │
│   │                                                                 │   │
│   │   ┌─────────────────────────────────────────────────────────┐   │   │
│   │   │              在线检索路径 (Online Retrieval)             │   │   │
│   │   │                                                         │   │   │
│   │   │   query_knowledge_hub(query, top_k)                     │   │   │
│   │   │         ↓                                               │   │   │
│   │   │   HybridSearch (Dense + Sparse + RRF)                   │   │   │
│   │   │         ↓                                               │   │   │
│   │   │   Reranker (CrossEncoder / LLM)                         │   │   │
│   │   │         ↓                                               │   │   │
│   │   │   MCPToolResponse {                                      │   │   │
│   │   │     content: str,                                        │   │   │
│   │   │     citations: List,                                     │   │   │
│   │   │     metadata: { query, result_count, collection },      │   │   │
│   │   │     chunks: [{ id, text, score, metadata }]             │   │   │
│   │   │   }                                                      │   │   │
│   │   │                                                         │   │   │
│   │   │   ⚠️ 不包含评估指标，只返回检索结果                       │   │   │
│   │   │                                                         │   │   │
│   │   └─────────────────────────────────────────────────────────┘   │   │
│   │                                                                 │   │
│   │   ┌─────────────────────────────────────────────────────────┐   │   │
│   │   │              离线评估路径 (Offline Evaluation)           │   │   │
│   │   │                                                         │   │   │
│   │   │   Dashboard 触发                                         │   │   │
│   │   │         ↓                                               │   │   │
│   │   │   EvalRunner.load_test_set(golden_test_set.json)        │   │   │
│   │   │         ↓                                               │   │   │
│   │   │   For each test_case:                                   │   │   │
│   │   │     - HybridSearch.search(query) → chunks               │   │   │
│   │   │     - Reranker.rerank(query, chunks)                    │   │   │
│   │   │     - Evaluator.evaluate(query, chunks, answer, truth)  │   │   │
│   │   │         ↓                                               │   │   │
│   │   │   EvalReport {                                          │   │   │
│   │   │     faithfulness: 0.85,                                  │   │   │
│   │   │     answer_relevancy: 0.92,                              │   │   │
│   │   │     hit_rate: 0.75,                                      │   │   │
│   │   │     mrr: 0.68                                            │   │   │
│   │   │   }                                                      │   │   │
│   │   │                                                         │   │   │
│   │   └─────────────────────────────────────────────────────────┘   │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    │ MCP Protocol                       │
│                                    ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    Agent (本项目)                                │   │
│   │                                                                 │   │
│   │   ┌─────────────────────────────────────────────────────────┐   │   │
│   │   │              实时评估路径 (Real-time Decision)           │   │   │
│   │   │                                                         │   │   │
│   │   │   MCP Client.query_knowledge_hub()                      │   │   │
│   │   │         ↓                                               │   │   │
│   │   │   RetrievalResult { chunks, scores }                    │   │   │
│   │   │         ↓                                               │   │   │
│   │   │   evaluate_node(chunks, threshold)                      │   │   │
│   │   │         ↓                                               │   │   │
│   │   │   Decision { is_sufficient, reason }                    │   │   │
│   │   │         ↓                                               │   │   │
│   │   │   ┌────┴────┐                                           │   │   │
│   │   │   ▼         ▼                                           │   │   │
│   │   │ generate   rewrite → retrieve (循环)                    │   │   │
│   │   │                                                         │   │   │
│   │   │   ⚠️ 这是流程控制决策，不是质量评估                       │   │   │
│   │   │                                                         │   │   │
│   │   └─────────────────────────────────────────────────────────┘   │   │
│   │                                                                 │   │
│   │   ┌─────────────────────────────────────────────────────────┐   │   │
│   │   │              可观测性 (Observability)                    │   │   │
│   │   │                                                         │   │   │
│   │   │   LangFuse Client                                        │   │   │
│   │   │     - trace.span() → 记录每个节点执行                    │   │   │
│   │   │     - trace.generation() → 记录LLM调用                   │   │   │
│   │   │     - 自定义指标 → decision_path, rewrite_count          │   │   │
│   │   │                                                         │   │   │
│   │   └─────────────────────────────────────────────────────────┘   │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 评估层级对照表

| 维度 | RAG离线评估 | Agent实时评估 |
|------|------------|---------------|
| **触发方式** | Dashboard手动触发 | 每次检索后自动 |
| **执行时机** | 批量 | 实时 |
| **数据来源** | 黄金测试集 | 用户实时查询 |
| **输入** | query + expected_chunks + generated_answer | chunks + scores |
| **输出** | 评估指标报告 | 是否充分的决策 |
| **调用LLM** | 是 (Ragas需要) | 否 |
| **延迟** | 1-3秒/指标 | <10ms |
| **用途** | 系统调优、回归测试 | 流程控制决策 |

### 9.3 数据流详解

#### 9.3.1 RAG返回数据结构

```python
@dataclass
class MCPToolResponse:
    content: str                    # Markdown格式内容
    citations: List[Citation]       # 引用列表
    metadata: Dict[str, Any]        # 元数据
    is_empty: bool                  # 是否为空
    image_contents: List[Image]     # 多模态图片

@dataclass  
class Chunk:
    id: str
    text: str
    score: float                    # ⬅️ RAG检索相关性评分
    metadata: Dict[str, Any]
```

**关键点**：
- `score`是单个chunk与query的相关性评分（由Dense/Sparse/Reranker计算）
- 不包含"检索结果是否足够"的判断
- 不包含Ragas评估指标

#### 9.3.2 Agent评估逻辑

```python
def evaluate_retrieval(
    query: str, 
    chunks: List[Chunk], 
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    基于RAG返回的scores判断检索结果是否足够。
    
    这是一个简化的决策逻辑，不重新计算相关性：
    1. 计算top5 chunk的平均score
    2. 检查数量是否足够(>=3)
    3. 返回决策结果
    
    Args:
        query: 原始查询（用于日志）
        chunks: RAG返回的chunks（已包含score）
        threshold: 平均分阈值
        
    Returns:
        {
            "is_sufficient": bool,    # 是否充分
            "reason": str,            # 决策原因
            "score": float            # 平均分
        }
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

### 9.4 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Agent评估是否调用LLM | 否 | 实时决策需要低延迟，避免额外LLM调用 |
| 是否移除evaluate_node | 否 | 这是流程控制决策，与RAG离线评估职责不同 |
| 简化评估逻辑 | 是 | 移除冗余的关键词匹配，直接基于scores判断 |
| Ragas评估位置 | RAG Server端 | 需要黄金测试集和用户输入答案，适合离线执行 |

### 9.5 Agent端 Ragas 离线评估

#### 9.5.1 为什么在 Agent 端做 Ragas 评估？

| 对比项 | RAG MCP Server | Agent 端（本项目） |
|--------|----------------|-------------------|
| **Answer 来源** | chunk 拼接（fallback） | **LLM 生成**（generator.py） |
| **Faithfulness** | 失真（答案=context，恒高分） | **有意义**（评估是否幻觉） |
| **Answer Relevancy** | 失真（拼接文本未必回答问题） | **有意义**（评估答案质量） |
| **评估场景** | 检索质量评估 | **端到端 RAG 质量评估** |

**核心问题**：RAG Server 的 Ragas 评估直接拼接 chunks 作为 answer，导致：
- Faithfulness 恒高（答案就是 context 的子集）
- Answer Relevancy 失真（拼接文本未必回答用户问题）

**解决方案**：在 Agent 端执行 Ragas 评估，使用 LLM 生成的真实答案。

#### 9.5.2 Ragas 评估架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Ragas Evaluation Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐     ┌──────────────────┐                      │
│  │ 数据收集模块      │     │ 评估执行模块      │                      │
│  │                  │     │                  │                      │
│  │ - 问答历史收集    │────▶│ - RagasEvaluator │                      │
│  │ - 手动标注接口    │     │ - 指标计算       │                      │
│  │ - 测试集导入      │     │ - 报告生成       │                      │
│  └──────────────────┘     └──────────────────┘                      │
│         │                          │                                 │
│         ▼                          ▼                                 │
│  ┌──────────────────┐     ┌──────────────────┐                      │
│  │ eval_data/       │     │ eval_reports/    │                      │
│  │ ├── raw/         │     │ ├── YYYY-MM-DD/  │                      │
│  │ │   session_xxx  │     │ │   report.json   │                      │
│  │ └── annotated/   │     │ └── latest.json   │                      │
│  │     test_set.json│     │                  │                      │
│  └──────────────────┘     └──────────────────┘                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 9.5.3 评估指标说明

| 指标 | 英文名 | 计算方式 | 意义 |
|------|--------|----------|------|
| **忠实度** | Faithfulness | LLM 判断答案 claims 是否都能从 contexts 推导 | 答案是否"幻觉" |
| **答案相关性** | Answer Relevancy | LLM 生成可能的问题，计算与原问题的相似度 | 答案是否"答非所问" |
| **上下文精确度** | Context Precision | LLM 判断每个 context 是否与问题相关 | 检索结果是否"精准" |

#### 9.5.4 数据格式

**测试集格式** (`eval_data/annotated/test_set.json`):

```json
{
  "test_cases": [
    {
      "query": "用户问题",
      "contexts": ["chunk1 text", "chunk2 text"],
      "answer": "LLM 生成的答案",
      "reference_answer": "人工标注的标准答案（可选，用于对比）",
      "metadata": {
        "session_id": "xxx",
        "timestamp": "2024-01-15T10:30:00Z"
      }
    }
  ]
}
```

**评估报告格式** (`eval_reports/2024-01-15/report.json`):

```json
{
  "evaluator": "RagasEvaluator",
  "timestamp": "2024-01-15T10:30:00Z",
  "test_set_path": "eval_data/annotated/test_set.json",
  "total_cases": 10,
  "aggregate_metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.78,
    "context_precision": 0.92
  },
  "per_case_results": [
    {
      "query": "什么是 RAG？",
      "metrics": {
        "faithfulness": 0.9,
        "answer_relevancy": 0.85,
        "context_precision": 0.95
      }
    }
  ]
}
```

#### 9.5.5 CLI 使用方式

```bash
# 使用标注后的测试集评估
python scripts/evaluate_ragas.py --test-set eval_data/annotated/test_set.json

# 指定输出目录
python scripts/evaluate_ragas.py --test-set test_set.json --output-dir eval_reports/

# 从 LangFuse 收集最近 7 天的问答历史
python scripts/evaluate_ragas.py --collect-from-langfuse --days 7 --output eval_data/raw/

# 仅评估指定指标
python scripts/evaluate_ragas.py --test-set test_set.json --metrics faithfulness answer_relevancy

# JSON 输出（便于管道处理）
python scripts/evaluate_ragas.py --test-set test_set.json --json
```

#### 9.5.6 核心文件

| 文件 | 职责 |
|------|------|
| `src/evaluation/ragas_evaluator.py` | Ragas 评估核心逻辑 |
| `src/evaluation/data_collector.py` | 问答数据收集（从 LangFuse/Session 提取） |
| `scripts/evaluate_ragas.py` | CLI 入口，执行离线评估 |
| `config/evaluation.yaml` | 评估配置（指标、LLM、阈值） |

### 9.6 未来扩展

| 扩展方向 | 说明 | 实现方式 |
|----------|------|----------|
| **在线质量评估** | 在Agent端实时评估生成质量 | 调用轻量级评估模型 |
| **评估结果缓存** | 缓存相似查询的评估结果 | Redis / 本地缓存 |
| **A/B Testing** | 对比不同评估策略 | LangFuse Sessions |
| **用户反馈闭环** | 根据用户反馈调整阈值 | Feedback节点 + 参数调优 |

---

## 10. 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Agent 框架 | LangGraph | 状态机模型适合条件分支，面试展示技术深度 |
| 检索方式 | MCP Client | 解耦架构，可独立部署 RAG Server |
| 评估架构 | 两层评估 | RAG离线评估质量，Agent实时决策流程 |
| UI 框架 | Streamlit | 快速迭代，适合 Demo 展示 |

---

## 11. 面试亮点

### 11.1 技术深度

1. **LangGraph 状态机设计**：展示对 Agent 架构的理解
2. **条件分支与循环**：展示复杂业务逻辑的建模能力
3. **MCP 协议**：展示对 LLM 生态标准的掌握
4. **两层评估架构**：展示对系统设计的深入思考

### 11.2 工程能力

1. **可插拔架构**：抽象接口 + 工厂模式
2. **配置驱动**：零代码切换 LLM Provider
3. **可观测性**：全链路追踪
4. **关注点分离**：RAG评估与Agent决策职责清晰

### 11.3 业务理解

1. **Agentic RAG**：展示对 RAG 发展趋势的理解
2. **企业场景**：展示对 B 端需求的理解 |