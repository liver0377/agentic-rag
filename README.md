# Agentic RAG 知识助手

面向企业内部知识管理场景的智能知识助手，基于 **Agentic RAG** 架构实现。

## 特性

- **Agentic RAG**: 自主评估检索结果、改写查询、子问题拆分
- **LangGraph 状态机**: 清晰的决策流程可视化
- **MCP 协议**: 标准 Tool Calling 接口，可对接 RAG Server
- **可观测性**: LangFuse 追踪，决策路径透明

## 快速开始

### 1. 安装依赖

```bash
cd agentic-rag-assistant
pip install -r requirements.txt
```

### 2. 配置

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env` 填入 API Key：

```bash
DEEPSEEK_API_KEY=your_api_key
```

### 3. 运行

**CLI 模式**：

```bash
python scripts/start_agent.py "什么是机器学习?"
```

**Web UI 模式**：

```bash
python scripts/start_ui.py
```

访问 http://localhost:8501

## 项目结构

```
agentic-rag-assistant/
├── src/
│   ├── agent/           # LangGraph Agent
│   │   ├── graph.py     # 状态机定义
│   │   ├── state.py     # Agent State
│   │   └── nodes/       # 各节点实现
│   ├── mcp_client/      # MCP 客户端
│   ├── ui/              # Streamlit UI
│   ├── evaluation/      # 评估模块
│   └── core/            # 基础设施
├── config/
│   └── settings.yaml    # 配置文件
├── tests/               # 测试
└── scripts/             # 启动脚本
```

## Agent 决策流程

```
用户问题 → 分析问题
    ↓
    ├── 简单问题 → 检索 → 评估 → 生成回答
    │                           ↑
    └── 复杂问题 → 拆分子问题 → 检索 → 评估 → 改写? ──┘
```

## 配置说明

```yaml
# config/settings.yaml

agent:
  max_rewrite_attempts: 2      # 最大改写次数
  enable_sub_query: true       # 启用子问题拆分
  enable_query_rewrite: true   # 启用查询改写
```

## 开发

运行测试：

```bash
pytest tests/
```

## 技术栈

- **Agent**: LangGraph
- **UI**: Streamlit
- **MCP**: mcp Python SDK
- **评估**: LangFuse + RAGAS

## License

MIT