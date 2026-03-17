# Modular RAG MCP Server 项目知识库

> 本文档为面试准备的核心知识库，包含项目架构、核心模块、技术决策和代码位置。

---

## 1. 项目概述

### 1.1 项目定位

**Agentic RAG（检索增强生成）知识助手**，面向企业知识管理场景。相比传统 RAG，核心差异是引入了 **LangGraph 状态机** 实现自主决策能力。

```
传统 RAG:  Query → Retrieve → Generate

Agentic RAG:  Query → [Analyze → Retrieve → Evaluate → Rewrite? → Sub-query?] → Generate with Citations
```

### 1.2 核心能力

| 能力 | 描述 | 实现方式 |
|------|------|----------|
| 子查询分解 | 将复杂问题拆分为子问题并行检索 | LangGraph 条件分支 + Decomposition Prompt |
| 查询改写 | 根据检索评估结果改写查询 | Rewrite Prompt + 重检索循环 |
| 引用追溯 | 在回答中标注来源文档和段落 | Chunk ID 追踪 + 引用格式化 |

### 1.3 设计理念

| 原则 | 描述 |
|------|------|
| Agent-First | Agent 是核心，检索只是工具 |
| 可观测性 | 全链路追踪，决策过程透明 |
| 可扩展性 | 模块化设计，易于添加能力 |
| 面试导向 | 代码质量高，架构清晰便于演示 |

---

## 2. 系统架构

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              展示层 (Presentation)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Streamlit Chat UI                           │   │
│  │        会话管理  ·  流式输出  ·  引用展示  ·  决策路径可视化          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                              Agent 层                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       LangGraph 状态机                               │   │
│  │   Nodes: Analyzer → Retriever → Evaluator → Rewriter → Generator   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                              集成层                                          │
│  ┌────────────────────────┐  ┌────────────────────────────────┐            │
│  │      MCP Client        │  │        LangFuse Client          │            │
│  │  connect_server()      │  │  trace.span()                   │            │
│  │  call_tool()           │  │  trace.generation()            │            │
│  └────────────────────────┘  └────────────────────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│                     [外部] RAG MCP Server                                    │
│              BM25 + Dense + RRF Fusion + Cross-Encoder Rerank               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
agentic-rag-assistant/
├── src/
│   ├── agent/                    # Agent 核心
│   │   ├── graph.py              # LangGraph 状态机定义
│   │   ├── state.py              # Agent State 定义
│   │   └── nodes/                # 节点实现
│   │       ├── analyzer.py       # 查询分析
│   │       ├── decomposer.py     # 子查询分解
│   │       ├── retriever.py      # 检索调用
│   │       ├── evaluator.py      # 检索评估
│   │       ├── rewriter.py       # 查询改写
│   │       └── generator.py      # 响应生成
│   ├── mcp_client/               # MCP 客户端
│   │   ├── client.py             # MCP 连接管理
│   │   └── tools.py              # 工具包装
│   ├── ui/                       # Web 界面
│   │   ├── app.py                # Streamlit 主入口
│   │   └── components/           # UI 组件
│   ├── evaluation/               # 评估模块
│   │   ├── langfuse_client.py    # LangFuse 集成
│   │   └── metrics.py            # 自定义指标
│   └── core/                     # 基础设施
│       ├── config.py             # 配置管理
│       ├── types.py               # 类型定义
│       ├── llm_client.py         # LLM 客户端
│       └── utils.py              # 工具函数
├── config/
│   └── settings.yaml             # 配置文件
├── tests/
│   ├── unit/                     # 单元测试
│   └── integration/              # 集成测试
└── scripts/
    ├── start_agent.py            # CLI 入口
    └── start_ui.py               # Web UI 入口
```

---

## 3. 核心模块详解

### 3.1 Agent 状态机 (LangGraph)

**位置:** `src/agent/graph.py`, `src/agent/state.py`

#### 状态定义

```python
@dataclass(kw_only=True)
class AgentState:
    # 输入
    original_query: str = ""
    rewritten_query: Optional[str] = None
    sub_queries: Annotated[List[str], reduce_strings] = field(default_factory=list)
    
    # 检索结果
    chunks: Annotated[List[Chunk], reduce_chunks] = field(default_factory=list)
    retrieval_score: Optional[float] = None
    
    # 评估结果
    is_sufficient: Optional[bool] = None
    evaluation_reason: Optional[str] = None
    rewrite_count: int = 0
    
    # 输出
    final_response: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    
    # 追踪
    trace_id: Optional[str] = None
    decision_path: Annotated[List[str], reduce_strings] = field(default_factory=list)
```

#### 图流程 (完整 Agent 模式)

```
START → analyze
         ↓
    ┌────┴────┐
    ↓         ↓
decompose   retrieve (简单查询)
    ↓         ↓
    └────┬────┘
         ↓
      retrieve
         ↓
      evaluate
         ↓
    ┌────┴────┐
    ↓         ↓
generate   rewrite (不充分)
    ↓         ↓
   END    retrieve (重试)
```

### 3.2 节点实现

#### Analyzer 节点 (`src/agent/nodes/analyzer.py`)

**职责:** 分析查询复杂度和类型，决定处理路径。

**复杂度指标:**
- 关键词: `和`、`以及`、`比较`、`区别`、`如何`、`为什么`
- 字数 > 30
- 多个问号

**查询类型:**

| 类型 | 关键词 |
|------|--------|
| `factual` | 是什么, 什么是, 定义, 概念 |
| `procedural` | 如何, 怎么, 步骤, 流程 |
| `analytical` | 为什么, 原因, 导致, 影响 |
| `comparative` | 比较, 区别, 不同, 相同 |
| `general` | 其他 |

#### Evaluator 节点 (`src/agent/nodes/evaluator.py`)

**职责:** 评估检索结果是否充分（流程控制决策，非质量评估）。

**关键设计决策:**
- RAG Server 负责 **离线评估** (Ragas: faithfulness, answer_relevancy)
- Agent 负责 **实时决策** (是否改写查询)
- 两者 **互补，非重复**

**评估逻辑:**
```python
def evaluate_retrieval(query: str, chunks: List[Chunk], threshold: float = 0.5):
    if not chunks:
        return {"is_sufficient": False, "reason": "未检索到任何相关文档"}
    
    avg_score = sum(c.score for c in chunks[:5]) / min(5, len(chunks))
    is_sufficient = avg_score >= threshold and len(chunks) >= 3
    
    return {"is_sufficient": is_sufficient, "score": avg_score}
```

#### Rewriter 节点 (`src/agent/nodes/rewriter.py`)

**职责:** 改写查询以改善检索结果。

**改写策略:**

| 问题类型 | 改写策略 |
|----------|----------|
| 相关性低 | 添加 "详细信息", "说明文档" |
| 数量不足 | 添加 "步骤和方法", "教程指南" |
| 匹配度低 | 添加 "定义", "概念", 或改为 "什么是X" |

#### Generator 节点 (`src/agent/nodes/generator.py`)

**职责:** 生成带引用的最终回答。

**流程:**
1. 格式化 chunks 为上下文 (限制 4000 字符)
2. 调用 LLM 生成回答
3. 提取前 5 个 chunks 作为引用

### 3.3 MCP Client (`src/mcp_client/client.py`)

**职责:** 通过 HTTP transport 连接 RAG MCP Server。

**关键方法:**
```python
class HTTPMCPClient:
    async def connect(self) -> None          # 初始化 MCP 连接
    async def call_tool(tool_name, args)     # 调用 MCP 工具
    async def query_knowledge_hub(query)     # 主检索工具
    async def list_collections()             # 列出可用集合
```

**协议:** MCP 2025-03-26 规范, JSON-RPC 2.0

### 3.4 LLM Client (`src/core/llm_client.py`)

**职责:** 统一 LLM 客户端，支持多 Provider。

**支持的 Provider:**
- OpenAI
- Azure OpenAI
- DeepSeek
- Ollama (本地模型)

**关键特性:** 配置驱动，零代码切换 Provider。

---

## 4. 关键技术决策

### 4.1 为什么选择 LangGraph？

| 决策 | 理由 |
|------|------|
| **状态机模型** | 适合复杂推理，有条件和循环 |
| **条件分支** | 支持分解和改写决策 |
| **面试价值** | 展示 Agent 架构理解能力 |

### 4.2 为什么用混合检索 (Dense + Sparse)？

| Dense Retrieval (向量) | Sparse Retrieval (BM25) |
|------------------------|-------------------------|
| 捕捉语义相似性 | 捕捉精确关键词匹配 |
| 解决"不同词同义"问题 | 解决专有名词查找问题 |
| 对同义词友好 | 对技术术语友好 |

**融合算法:** RRF (Reciprocal Rank Fusion)
```
Score = 1/(k + Rank_Dense) + 1/(k + Rank_Sparse)
```

**为什么选 RRF？**
- 不依赖绝对分数值（归一化问题）
- 基于排名位置（更稳定）
- 平滑单模态失效

### 4.3 为什么需要两层评估架构？

| 维度 | RAG 离线评估 | Agent 实时评估 |
|------|-------------|----------------|
| **位置** | RAG MCP Server | Agent 端 |
| **触发** | Dashboard 手动 | 每次检索后 |
| **输入** | Golden 测试集 | 实时查询 chunks |
| **输出** | 质量指标 | 决策 (充分/不充分) |
| **LLM 调用** | 是 (Ragas 需要) | 否 |
| **延迟** | 1-3 秒/指标 | <10ms |
| **目的** | 系统调优、回归测试 | 流程控制决策 |

**关键洞察:** RAG 返回的分数是 **单个 chunk 相关性分数**，不是整体检索质量评估。

### 4.4 为什么选择 MCP 协议？

| 优势 | 描述 |
|------|------|
| **标准接口** | 可与 GitHub Copilot、Claude Desktop 集成 |
| **工具调用** | 标准 JSON-RPC 2.0 协议 |
| **零前端开发** | 复用现有 AI 助手 |
| **上下文共享** | Copilot 同时看到代码和知识库 |

---

## 5. 重要代码位置

### 5.1 核心文件

| 文件 | 职责 | 关键函数 |
|------|------|----------|
| `src/agent/graph.py` | LangGraph 定义 | `build_agent_graph()`, `KnowledgeAssistant` |
| `src/agent/state.py` | 状态定义 | `AgentState`, `create_initial_state()` |
| `src/core/config.py` | 配置管理 | `load_settings()`, `Settings` |
| `src/core/types.py` | 类型定义 | `Chunk`, `Citation`, `RetrievalResult` |
| `src/core/llm_client.py` | LLM 抽象 | `LLMClient`, `create_llm_client()` |

### 5.2 节点文件

| 文件 | 关键函数 |
|------|----------|
| `src/agent/nodes/analyzer.py` | `analyze_query()`, `should_decompose()` |
| `src/agent/nodes/decomposer.py` | `decompose_query_with_llm()` |
| `src/agent/nodes/retriever.py` | `retrieve_node_sync()` |
| `src/agent/nodes/evaluator.py` | `evaluate_retrieval()`, `should_rewrite()` |
| `src/agent/nodes/rewriter.py` | `rewrite_query()`, `rewrite_query_with_llm()` |
| `src/agent/nodes/generator.py` | `generate_response_with_llm()` |

### 5.3 集成文件

| 文件 | 职责 |
|------|------|
| `src/mcp_client/client.py` | MCP HTTP 客户端 |
| `src/evaluation/langfuse_client.py` | LangFuse 追踪 |
| `src/ui/app.py` | Streamlit 应用 |

---

## 6. 配置设计

### 6.1 配置结构 (`config/settings.yaml`)

```yaml
# LLM 配置
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  api_key: "${DEEPSEEK_API_KEY}"
  base_url: "https://api.deepseek.com/v1"
  temperature: 0.0

# Agent 配置
agent:
  max_rewrite_attempts: 2      # 最大改写尝试次数
  retrieval_top_k: 10          # 检索数量
  sufficiency_threshold: 0.7   # 充分性阈值
  enable_sub_query: true       # 启用子查询分解
  enable_query_rewrite: true   # 启用查询改写

# RAG MCP Server 配置
rag_server:
  url: "http://127.0.0.1:8080"
  collection: "knowledge_hub"
  timeout: 60

# LangFuse 配置
langfuse:
  enabled: true
  public_key: "${LANGFUSE_PUBLIC_KEY}"
  secret_key: "${LANGFUSE_SECRET_KEY}"
  host: "https://cloud.langfuse.com"
```

---

## 7. 高频面试问题参考答案

### Q1: 介绍一下这个项目的架构？

**参考答案:**

这个项目是一个 Agentic RAG 知识助手，架构分四层。

最上层是展示层，用 Streamlit 实现了一个聊天界面，支持流式输出和引用展示。

核心是 Agent 层，用 LangGraph 构建了一个状态机。包含分析、检索、评估、改写、生成五个节点，通过条件分支实现复杂推理流程。

下面是集成层，主要是 MCP Client 和 LangFuse Client。MCP Client 负责调用外部 RAG Server，LangFuse 负责全链路追踪。

最底层是外部 RAG MCP Server，实现了混合检索和重排序。

设计上最大的亮点是状态机模型，能根据查询复杂度自动选择处理路径，简单问题直接检索，复杂问题走分解流程。

### Q2: 为什么选择 LangGraph 而不是 LangChain Chain？

**参考答案:**

主要有三点考虑。

第一是复杂度。LangChain Chain 是线性的，适合简单流程。但我们的场景需要条件分支和循环，比如检索不充分时要改写查询重试，这是 Chain 不擅长表达的。

第二是状态管理。LangGraph 有显式的 State 定义，每个节点可以读取和更新状态，调试时能清楚看到状态变化。Chain 的状态管理比较隐式。

第三是可观测性。LangGraph 自带 trace 功能，每一步决策都能记录，配合 LangFuse 能做完整的链路追踪。

当然 LangGraph 学习成本稍高，但考虑到面试演示价值，这个投入是值得的。

### Q3: 混合检索是怎么做的？为什么用 RRF？

**参考答案:**

我们做了两路召回，一路用向量做语义匹配，一路用 BM25 做关键词匹配，然后用 RRF 算法融合。

向量检索的优势是能捕捉语义相似性，比如"机器学习"和"ML"会被识别为相关。但它对专有名词、技术术语不太友好。

BM25 刚好相反，它做精确关键词匹配，对专有名词效果好，但无法理解语义。

RRF 融合的公式是 `Score = 1/(k + Rank)`，k 通常是 60。它的好处是不依赖绝对分数值，只看排名位置。

为什么这很重要？因为向量和 BM25 的分数范围不同，归一化很麻烦。RRF 直接用排名，归一化问题自然解决，而且对单个模态失效更鲁棒。

### Q4: Evaluator 是怎么评估检索结果的？

**参考答案:**

Evaluator 的职责是判断检索结果是否充分，用来做流程控制决策，不是做质量评估。

具体逻辑是取前 5 个 chunks，计算平均相关性分数，如果分数达到阈值（默认 0.7）且数量不少于 3 个，就认为充分。

这里有个设计要点：RAG Server 已经有离线评估，用 Ragas 算 faithfulness、answer_relevancy 这些指标。Agent 这层评估是为了实时决策，不需要 LLM 调用，延迟在 10ms 以内。

两层评估是互补的：离线评估用于系统调优和回归测试，实时评估用于判断是否需要改写查询重试。

### Q5: 如果检索结果一直不充分怎么办？

**参考答案:**

我们做了几层保护机制。

第一是最大重试次数限制，默认是 2 次。超过次数就不再改写，直接用现有结果生成回答。

第二是改写策略会根据失败原因调整。如果相关性低，会加"详细信息"这类限定词；如果数量不足，会加"步骤方法"扩展查询。

第三是 Generator 有兜底逻辑。即使 chunks 不充分，也会基于有限信息生成回答，同时在回答中标注引用不充分。

从产品角度，这种情况可以触发人工介入或者反馈收集，帮助优化知识库。

### Q6: MCP 协议是什么？为什么用它？

**参考答案:**

MCP 是 Model Context Protocol，是 Anthropic 推出的开放协议，让 LLM 能通过标准化接口调用外部工具。

我们用它主要是为了解耦。RAG 能力（索引、检索、重排序）是一个独立服务，通过 MCP 暴露工具接口。Agent 作为 MCP Client 调用这些工具。

好处有几个：

一是标准接口。支持 MCP 的客户端都能用，比如 Claude Desktop、GitHub Copilot。这意味着我们不需要开发前端，直接复用现有 AI 助手。

二是协议规范。JSON-RPC 2.0，消息格式统一，调试方便。

三是扩展性。RAG Server 可以独立迭代，Agent 侧不需要改动。

### Q7: 怎么保证回答的可追溯性？

**参考答案:**

我们在三个层面实现了可追溯性。

第一是 Chunk ID 追踪。每个检索结果都带着 source document ID 和 chunk ID，生成回答时把这些 ID 传给 LLM，让它引用。

第二是引用展示。UI 层会把 chunks 和回答对应起来，用户点击引用能看到原文。

第三是全链路追踪。LangFuse 记录了每一步的输入输出，包括决策路径。如果回答有问题，可以通过 trace ID 回溯整个流程。

设计上有个权衡：引用数量我们限制在 5 个以内，太多了用户看不过来，也会影响回答的聚焦度。

### Q8: 项目中的依赖注入是怎么做的？

**参考答案:**

主要是通过工厂模式和配置驱动。

LLM Client 这块，我们定义了抽象接口，然后用 `create_llm_client()` 工厂函数根据配置创建具体实现。切换 Provider 只需要改配置文件，代码不需要动。

MCP Client 也是类似，通过配置注入 RAG Server 地址和超时参数。

Agent 层的依赖是通过构造函数注入的，`KnowledgeAssistant` 类接收 `mcp_client` 和 `llm_client` 作为参数，方便测试时 mock。

这套设计的好处是解耦和可测试性，代价是增加了一些抽象层，对新人理解代码可能稍有门槛。

---

## 8. 代码路径速查表

| 模块 | 关键文件 |
|------|----------|
| Agent 状态机 | `src/agent/graph.py` |
| 状态定义 | `src/agent/state.py` |
| 查询分析 | `src/agent/nodes/analyzer.py` |
| 子查询分解 | `src/agent/nodes/decomposer.py` |
| 检索节点 | `src/agent/nodes/retriever.py` |
| 评估节点 | `src/agent/nodes/evaluator.py` |
| 查询改写 | `src/agent/nodes/rewriter.py` |
| 响应生成 | `src/agent/nodes/generator.py` |
| MCP 客户端 | `src/mcp_client/client.py` |
| LLM 客户端 | `src/core/llm_client.py` |
| 配置管理 | `src/core/config.py` |
| 类型定义 | `src/core/types.py` |
| LangFuse | `src/evaluation/langfuse_client.py` |
| Web UI | `src/ui/app.py` |

---

## 9. 运行命令

### CLI 模式

```bash
python scripts/start_agent.py "什么是机器学习?"
```

### Web UI 模式

```bash
python scripts/start_ui.py
# 访问 http://localhost:8501
```

### 测试

```bash
pytest tests/
```