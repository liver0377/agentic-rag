"""LangGraph graph definition for Agentic RAG Agent.

This module defines the state machine that orchestrates the agent's behavior.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from src.agent.nodes.analyzer import analyze_node, should_decompose
from src.agent.nodes.decomposer import decompose_node
from src.agent.nodes.evaluator import evaluate_node, should_rewrite
from src.agent.nodes.generator import generate_node
from src.agent.nodes.retriever import retrieve_node_sync
from src.agent.nodes.rewriter import rewrite_node
from src.agent.state import AgentState, create_initial_state
from src.core.config import Settings, load_settings
from src.core.llm_client import LLMClient, create_llm_client
from src.evaluation.langfuse_client import LangFuseTracer, init_langfuse


def build_simple_graph(llm_client: Optional[LLMClient] = None) -> StateGraph:
    """Build a simple linear graph for basic RAG flow.

    This is a simplified version without decomposition and rewrite loops.
    Good for testing and simple queries.

    Flow:
        START -> analyze -> retrieve -> evaluate -> generate -> END

    Args:
        llm_client: Optional LLM client for generation.

    Returns:
        Compiled StateGraph.
    """
    builder = StateGraph(AgentState)

    builder.add_node("analyze", analyze_node)
    builder.add_node("retrieve", retrieve_node_sync)
    builder.add_node("evaluate", partial(evaluate_node, llm_client=llm_client))
    builder.add_node("generate", partial(generate_node, llm_client=llm_client))

    builder.set_entry_point("analyze")
    builder.add_edge("analyze", "retrieve")
    builder.add_edge("retrieve", "evaluate")
    builder.add_edge("evaluate", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


def build_agent_graph(
    enable_decomposition: bool = True,
    enable_rewrite: bool = True,
    max_rewrite_attempts: int = 2,
    llm_client: Optional[LLMClient] = None,
) -> StateGraph:
    """Build the full agent graph with conditional branches.

    Flow:
        START -> analyze
        analyze -> decompose (if complex) OR retrieve (if simple)
        decompose -> retrieve
        retrieve -> evaluate
        evaluate -> rewrite (if insufficient) OR generate (if sufficient)
        rewrite -> retrieve
        generate -> END

    Args:
        enable_decomposition: Enable query decomposition.
        enable_rewrite: Enable query rewriting.
        max_rewrite_attempts: Maximum rewrite attempts.
        llm_client: Optional LLM client for LLM-based operations.

    Returns:
        Compiled StateGraph.
    """
    builder = StateGraph(AgentState)

    builder.add_node("analyze", analyze_node)
    builder.add_node("decompose", partial(decompose_node, llm_client=llm_client))
    builder.add_node("retrieve", retrieve_node_sync)
    builder.add_node("evaluate", partial(evaluate_node, llm_client=llm_client))
    builder.add_node("rewrite", partial(rewrite_node, llm_client=llm_client))
    builder.add_node("generate", partial(generate_node, llm_client=llm_client))

    builder.set_entry_point("analyze")

    if enable_decomposition:
        builder.add_conditional_edges(
            "analyze",
            should_decompose,
            {
                "decompose": "decompose",
                "retrieve": "retrieve",
            },
        )
        builder.add_edge("decompose", "retrieve")
    else:
        builder.add_edge("analyze", "retrieve")

    builder.add_edge("retrieve", "evaluate")

    if enable_rewrite:
        builder.add_conditional_edges(
            "evaluate",
            lambda state: should_rewrite(state, max_rewrite_attempts),
            {
                "rewrite": "rewrite",
                "generate": "generate",
            },
        )
        builder.add_edge("rewrite", "retrieve")
    else:
        builder.add_edge("evaluate", "generate")

    builder.add_edge("generate", END)

    return builder.compile()


def run_agent(query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the agent with a query.

    This is the main entry point for the agent.

    Args:
        query: User's question.
        config: Optional configuration.

    Returns:
        Agent output dictionary.
    """
    config = config or {}
    enable_decomposition = config.get("enable_decomposition", True)
    enable_rewrite = config.get("enable_rewrite", True)
    max_rewrite_attempts = config.get("max_rewrite_attempts", 2)
    use_llm = config.get("use_llm", True)

    llm_client = create_llm_client() if use_llm else None

    graph = build_agent_graph(
        enable_decomposition=enable_decomposition,
        enable_rewrite=enable_rewrite,
        max_rewrite_attempts=max_rewrite_attempts,
        llm_client=llm_client,
    )

    initial_state = create_initial_state(query)

    final_state = graph.invoke(initial_state)

    if isinstance(final_state, dict):
        return {
            "query": final_state.get("original_query", query),
            "response": final_state.get("final_response"),
            "citations": final_state.get("citations", []),
            "sub_queries": final_state.get("sub_queries", []),
            "rewritten_query": final_state.get("rewritten_query"),
            "decision_path": final_state.get("decision_path", []),
            "total_chunks": len(final_state.get("chunks", [])),
            "trace_id": final_state.get("trace_id"),
            "error": final_state.get("error"),
        }

    return final_state.to_output_dict()


async def run_agent_async(query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the agent asynchronously.

    Args:
        query: User's question.
        config: Optional configuration.

    Returns:
        Agent output dictionary.
    """
    return run_agent(query, config)


class KnowledgeAssistant:
    """High-level knowledge assistant class.

    This class provides a convenient interface for running the agent.

    Example:
        >>> assistant = KnowledgeAssistant()
        >>> result = assistant.ask("什么是机器学习？")
        >>> print(result["response"])
    """

    def __init__(
        self,
        enable_decomposition: bool = True,
        enable_rewrite: bool = True,
        max_rewrite_attempts: int = 2,
        llm_client: Optional[LLMClient] = None,
        use_llm: bool = True,
        settings: Optional[Settings] = None,
    ):
        """Initialize the knowledge assistant.

        Args:
            enable_decomposition: Enable query decomposition.
            enable_rewrite: Enable query rewriting.
            max_rewrite_attempts: Maximum rewrite attempts.
            llm_client: Optional LLM client instance.
            use_llm: Whether to use LLM for operations.
            settings: Optional settings object.
        """
        self.enable_decomposition = enable_decomposition
        self.enable_rewrite = enable_rewrite
        self.max_rewrite_attempts = max_rewrite_attempts
        self.use_llm = use_llm

        if settings is not None:
            self._settings = settings
        else:
            self._settings = load_settings()

        if llm_client is not None:
            self._llm_client = llm_client
        elif use_llm:
            self._llm_client = create_llm_client()
        else:
            self._llm_client = None

        self._graph = None
        self._tracer: Optional[LangFuseTracer] = None

        if self._settings.langfuse.enabled:
            self._tracer = init_langfuse(self._settings.langfuse)

    @property
    def graph(self):
        """Get or build the graph."""
        if self._graph is None:
            self._graph = build_agent_graph(
                enable_decomposition=self.enable_decomposition,
                enable_rewrite=self.enable_rewrite,
                max_rewrite_attempts=self.max_rewrite_attempts,
                llm_client=self._llm_client,
            )
        return self._graph

    def ask(self, query: str, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """Ask a question.

        Args:
            query: User's question.
            trace_id: Optional trace ID.

        Returns:
            Agent output dictionary.
        """
        initial_state = create_initial_state(query, trace_id)

        if self._tracer:
            with self._tracer.trace(name="agent_query", metadata={"query": query}):
                final_state = self.graph.invoke(initial_state)
        else:
            final_state = self.graph.invoke(initial_state)

        if isinstance(final_state, dict):
            return {
                "query": final_state.get("original_query", query),
                "response": final_state.get("final_response"),
                "citations": final_state.get("citations", []),
                "sub_queries": final_state.get("sub_queries", []),
                "rewritten_query": final_state.get("rewritten_query"),
                "decision_path": final_state.get("decision_path", []),
                "total_chunks": len(final_state.get("chunks", [])),
                "trace_id": final_state.get("trace_id"),
                "error": final_state.get("error"),
            }

        return final_state.to_output_dict()

    def __call__(self, query: str) -> Dict[str, Any]:
        """Make the assistant callable.

        Args:
            query: User's question.

        Returns:
            Agent output dictionary.
        """
        return self.ask(query)
