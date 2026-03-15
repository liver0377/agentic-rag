# Agentic RAG 知识助手 - 开发规格文档

> 版本：1.0 — 完整开发规格

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 核心特点](#2-核心特点)
- [3. 技术选型](#3-技术选型)
- [4. 系统架构](#4-系统架构)
- [5. 模块设计](#5-模块设计)
- [6. 配置设计](#6-配置设计)
- [7. 项目排期](#7-项目排期)
- [8. 可扩展性](#8-可扩展性)

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

| 维度 | 指标 | 工具 |
|------|------|------|
| **检索质量** | Hit Rate, MRR, Recall@K | RAGAS |
| **生成质量** | Faithfulness, Relevancy | RAGAS + LLM-as-Judge |
| **Agent 可追溯性** | 决策路径、工具调用链、Token 消耗 | LangFuse |
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
│   │   ├── ragas_evaluator.py    # RAGAS 评估
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
│   └── prompts/                  # Prompt 文件
│       ├── analyze.txt
│       ├── decompose.txt
│       ├── evaluate.txt
│       └── rewrite.txt
│
├── tests/
│   ├── unit/
│   │   └── __init__.py
│   └── integration/
│       └── __init__.py
│
├── scripts/
│   ├── start_agent.py            # 启动 Agent CLI
│   └── start_ui.py               # 启动 Web UI
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

### C. 评估体系 [C1-C3]

| 任务 | 说明 | 产物 | 状态 |
|------|------|------|------|
| C1. LangFuse 集成 | 追踪 SDK 接入 | src/evaluation/langfuse_client.py | [ ] |
| C2. RAGAS 评估 | 离线评估脚本 | src/evaluation/ragas_evaluator.py | [ ] |
| C3. 自定义指标 | Agent 特有指标 | src/evaluation/metrics.py | [ ] |

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

## 9. 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Agent 框架 | LangGraph | 状态机模型适合条件分支，面试展示技术深度 |
| 检索方式 | MCP Client | 解耦架构，可独立部署 RAG Server |
| 评估工具 | LangFuse + RAGAS | 覆盖在线追踪和离线评估双重需求 |
| UI 框架 | Streamlit | 快速迭代，适合 Demo 展示 |

---

## 10. 面试亮点

### 10.1 技术深度

1. **LangGraph 状态机设计**：展示对 Agent 架构的理解
2. **条件分支与循环**：展示复杂业务逻辑的建模能力
3. **MCP 协议**：展示对 LLM 生态标准的掌握

### 10.2 工程能力

1. **可插拔架构**：抽象接口 + 工厂模式
2. **配置驱动**：零代码切换 LLM Provider
3. **可观测性**：全链路追踪

### 10.3 业务理解

1. **Agentic RAG**：展示对 RAG 发展趋势的理解
2. **企业场景**：展示对 B 端需求的理解