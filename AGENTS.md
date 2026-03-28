# AGENTS.md

Guidelines for agentic coding assistants working in the Modular RAG MCP Server codebase.

## Build/Lint/Test Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Tests
```bash
# Run all tests
pytest tests/

# Run all tests with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/unit/test_nodes.py

# Run a single test by name
pytest tests/unit/test_nodes.py::TestAnalyzerNode::test_analyze_simple_query

# Run with specific markers (if any)
pytest tests/ -m "not slow"
```

### Linting and Formatting
```bash
# Run ruff linter
ruff check .

# Run ruff formatter
ruff format .

# Run both check and format
ruff check . && ruff format .
```

### Run Application
```bash
# CLI mode
python scripts/start_agent.py "What is machine learning?"

# Web UI mode
python scripts/start_ui.py
```

## Project Structure

```
agentic-rag-assistant/
├── src/
│   ├── agent/           # LangGraph Agent
│   │   ├── graph.py     # State machine definition
│   │   ├── state.py     # AgentState dataclass
│   │   └── nodes/       # Node implementations (analyzer, decomposer, etc.)
│   ├── mcp_client/      # MCP protocol client
│   ├── ui/              # Streamlit UI components
│   ├── evaluation/      # RAGAS evaluation, LangFuse integration
│   └── core/            # Config, types, LLM client, utilities
├── config/
│   └── settings.yaml    # YAML configuration with env var substitution
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
└── scripts/             # Entry point scripts
```

## Code Style Guidelines

### Python Version
- Target: Python 3.10+
- Use `from __future__ import annotations` at the top of files for modern type hints

### Imports
```python
# Standard library first
from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Optional

# Third-party next
from langgraph.graph import END, StateGraph
import yaml

# Local imports last (use explicit src. prefix)
from src.agent.state import AgentState
from src.core.config import load_settings
from src.core.types import Chunk
```

### Type Hints
- Always use type hints for function parameters and return types
- Use `Optional[T]` for optional parameters
- Use `Dict[str, Any]` for flexible dictionaries
- Use `List[T]` for lists

```python
def analyze_query(query: str) -> Dict[str, Any]:
    ...

def create_initial_state(query: str, trace_id: Optional[str] = None) -> AgentState:
    ...
```

### Naming Conventions
- **Functions**: `snake_case` (e.g., `analyze_query`, `create_initial_state`)
- **Classes**: `PascalCase` (e.g., `AgentState`, `KnowledgeAssistant`, `LLMClient`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_REWRITE_ATTEMPTS`)
- **Private functions**: Prefix with `_` (e.g., `_substitute_env_vars`)
- **Node functions**: Suffix with `_node` (e.g., `analyze_node`, `retrieve_node`)

### Dataclasses
- Use `@dataclass` for data containers
- Use `kw_only=True` for keyword-only initialization
- Use `field(default_factory=list)` for mutable defaults

```python
@dataclass(kw_only=True)
class AgentState:
    original_query: str = ""
    sub_queries: List[str] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
```

### Docstrings
- Use triple-quoted docstrings for modules, classes, and public functions
- Include Args and Returns sections

```python
def analyze_query(query: str) -> Dict[str, Any]:
    """Analyze query characteristics.

    Args:
        query: The user's query.

    Returns:
        Analysis result with complexity, type, etc.
    """
```

### Error Handling
- Return error information in state rather than raising exceptions in nodes
- Use `Optional` types for values that may be None
- Check for None/empty values before processing

```python
if not chunks:
    return {"is_sufficient": False, "reason": "No chunks retrieved"}

if config.api_key is None:
    return None
```

### Configuration
- Load settings via `load_settings()` from `src.core.config`
- Access config values: `settings.agent.max_rewrite_attempts`
- Use environment variable substitution in YAML: `${ENV_VAR_NAME}`

### Agent Nodes
- Node functions take `AgentState` and return `Dict[str, Any]`
- Use `functools.partial` for parameterized nodes
- Add decisions to `decision_path` for traceability

```python
def analyze_node(state: AgentState) -> Dict[str, Any]:
    return {
        "decision_path": [f"analyze: {result}"],
        "sub_queries": [],
    }
```

### Testing
- Use pytest with class-based test organization
- Group related tests in classes (e.g., `TestAnalyzerNode`)
- Use descriptive test names: `test_analyze_simple_query`
- Place fixtures in test files or conftest.py

```python
class TestAnalyzerNode:
    def test_analyze_simple_query(self):
        result = analyze_query("What is machine learning?")
        assert "is_complex" in result
```

## Formatting Rules (Ruff)
- Line length: 100 characters
- Target version: py310
- Enabled rules: E, F, I, W (pycodestyle, pyflakes, isort, warnings)
- E501 (line too long) is ignored

## Key Patterns

### State Machine Flow
```
START -> analyze -> [decompose?] -> retrieve -> evaluate -> [rewrite?] -> generate -> END
```

### Priority for Query Retrieval
```
rewritten_sub_queries > sub_queries > rewritten_query > original_query
```

### Creating LLM Client
```python
from src.core.llm_client import create_llm_client
llm_client = create_llm_client()  # Loads from settings
```

### Running Agent
```python
from src.agent.graph import KnowledgeAssistant
assistant = KnowledgeAssistant()
result = assistant.ask("Your question here")
```
