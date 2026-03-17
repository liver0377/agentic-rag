# Agentic RAG 知识助手 - 项目知识点

## 1. 项目概述

### 1.1 项目定位
面向企业内部知识管理场景，构建基于 **Agentic RAG** 的智能知识助手。为企业员工提供对内部文档、数据库等私有知识的自然语言访问能力。

### 1.2 核心价值

**传统 RAG vs Agentic RAG**:
- 传统 RAG: Query → Retrieve → Generate
- Agentic RAG: Query → [Analyze → Retrieve → Evaluate → Rewrite? → Sub-query?] → Generate with Citations

### 1.3 设计理念
- **Agent-First**: 以 Agent 为核心，检索只是工具
- **可观测性**: 全链路追踪，决策过程透明
- **可扩展性**: 模块化设计，易于添加新能力
- **面试导向**: 代码质量高，架构清晰，便于讲解

## 2. 核心特点

### 2.1 Agentic RAG 能力

| 能力 | 说明 | 实现方式 |
|------|------|----------|
| **子问题拆分** | 复杂问题拆分为子问题分别检索 | LangGraph 条件分支 + Decomposition Prompt |
| **查询改写** | 根据评估结果改写查询重新检索 | Rewrite Prompt + Re-retrieve Loop |
| **引用溯源** | 回答中标注来源文档和段落 | Chunk ID 追踪 + Citation Formatter |

### 2.2 Agent 决策流程

**完整流程**:
1. **QUERY_ANALYZER**: 分析问题复杂度（简单/复杂）
2. **SUB_QUERY_DECOMPOSER**: 复杂查询拆分为子问题
3. **RETRIEVER**: 通过 MCP Client 调用 RAG Server
4. **RETRIEVAL_EVALUATOR**: 评估检索结果是否充分
5. **QUERY_REWRITER**: 不充分时改写查询
6. **RESPONSE_GENERATOR**: 生成回答 + 引用标注

### 2.3 MCP 集成

作为 **MCP Client** 调用 RAG Server 的工具：
- `query_knowledge_hub`: 混合检索，返回 Chunks
- `list_collections`: 列出可用知识库
- `get_document_summary`: 获取文档摘要

## 3. 技术选型

| 层级 | 技术栈 | 说明 |
|------|--------|------|
| **Agent 框架** | LangGraph | 状态机 + 条件分支，适合复杂推理流程 |
| **MCP Client** | mcp Python SDK | 标准 MCP 协议通信 |
| **LLM** | Azure OpenAI / DeepSeek / Qwen | 可插拔，复用配置模式 |
| **Web UI** | Streamlit | 快速构建 Chat 界面 |
| **评估追踪** | LangFuse | 开源 LLM 可观测平台 |
| **离线评估** | RAGAS | RAG 专用评估框架 |

## 4. 系统架构

### 4.1 整体架构分层

```
Presentation Layer (Streamlit Chat UI)
    ↓
Agent Layer (LangGraph State Machine)
    ↓
Integration Layer (MCP Client + LangFuse Client)
    ↓
External RAG MCP Server
```

### 4.2 目录结构

```
src/
├── agent/           # Agent 核心（graph.py, state.py, nodes/）
├── mcp_client/      # MCP 客户端（client.py, tools.py）
├── ui/              # Web 界面（app.py, components/）
├── evaluation/      # 评估模块（langfuse_client.py, ragas_evaluator.py）
└── core/            # 基础设施（config.py, types.py）
```

## 5. 模块设计

### 5.1 Agent State 定义

AgentState 包含：
- **输入**: original_query, rewritten_query, sub_queries
- **检索结果**: chunks, retrieval_score
- **评估结果**: is_sufficient, evaluation_reason, rewrite_count
- **输出**: final_response, citations
- **追踪**: trace_id, decision_path

### 5.2 LangGraph 节点接口

每个节点遵循统一接口：
```python
def node_name(state: AgentState) -> AgentState:
    # 1. 从 state 提取输入
    # 2. 执行节点逻辑
    # 3. 返回状态更新
    return {"key": "value"}
```

## 6. 评估体系架构

### 6.1 两层评估体系

| 层级 | 位置 | 触发方式 | 职责 |
|------|------|----------|------|
| **RAG离线评估** | RAG MCP Server | Dashboard 手动触发 | 评估检索系统整体质量 |
| **Agent实时评估** | Agent端 | 每次检索后实时 | 决定是否改写查询重试 |

### 6.2 为什么需要两层评估？

1. **RAG服务返回的score**: 是单个chunk的相关性评分，不是整体检索质量的评估
2. **为什么不在RAG服务中做实时评估**: Ragas评估需要调用LLM，延迟高(1-3秒/指标)，不适合在线检索
3. **Agent端评估是否重复**: 不重复。Agent评估基于RAG返回的scores做实时决策，不是重新评估

### 6.3 Agent端 Ragas 离线评估

**为什么在 Agent 端做 Ragas 评估？**

| 对比项 | RAG MCP Server | Agent 端 |
|--------|----------------|----------|
| **Answer 来源** | chunk 拼接（fallback） | **LLM 生成**（generator.py） |
| **Faithfulness** | 失真（答案=context，恒高分） | **有意义**（评估是否幻觉） |
| **Answer Relevancy** | 失真（拼接文本未必回答问题） | **有意义**（评估答案质量） |

**评估指标**:
- **忠实度 (Faithfulness)**: 答案是否"幻觉"
- **答案相关性 (Answer Relevancy)**: 答案是否"答非所问"
- **上下文精确度 (Context Precision)**: 检索结果是否"精准"

## 7. 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Agent 框架 | LangGraph | 状态机模型适合条件分支，面试展示技术深度 |
| 检索方式 | MCP Client | 解耦架构，可独立部署 RAG Server |
| 评估架构 | 两层评估 | RAG离线评估质量，Agent实时决策流程 |
| UI 框架 | Streamlit | 快速迭代，适合 Demo 展示 |

## 8. 面试亮点

### 8.1 技术深度
1. **LangGraph 状态机设计**: 展示对 Agent 架构的理解
2. **条件分支与循环**: 展示复杂业务逻辑的建模能力
3. **MCP 协议**: 展示对 LLM 生态标准的掌握
4. **两层评估架构**: 展示对系统设计的深入思考

### 8.2 工程能力
1. **可插拔架构**: 抽象接口 + 工厂模式
2. **配置驱动**: 零代码切换 LLM Provider
3. **可观测性**: 全链路追踪
4. **关注点分离**: RAG评估与Agent决策职责清晰

### 8.3 业务理解
1. **Agentic RAG**: 展示对 RAG 发展趋势的理解
2. **企业场景**: 展示对 B 端需求的理解

## 9. 可扩展性

### 9.1 Agent 能力扩展点
- **工具扩展**: 在 LangGraph 中添加新节点
- **记忆扩展**: 添加 Memory 节点
- **多 Agent 协作**: 使用 LangGraph 的 subgraph

### 9.2 评估扩展点
- **A/B Testing**: 对比不同策略
- **在线学习**: 用户反馈优化