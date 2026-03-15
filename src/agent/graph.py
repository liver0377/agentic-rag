"""LangGraph graph definition for Agentic RAG Agent.

This module defines the state machine that orchestrates the agent's behavior.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from src.agent.nodes.analyzer import analyze_node, should_decompose
from src.agent.nodes.decomposer import decompose_node
from src.agent.nodes.evaluator import evaluate_node, should_rewrite
from src.agent.nodes.generator import generate_node
from src.agent.nodes.retriever import retrieve_node_sync
from src.agent.nodes.rewriter import rewrite_node
from src.agent.state import AgentState, create_initial_state


def build_simple_graph() -> StateGraph:
    """Build a simple linear graph for basic RAG flow.

    This is a simplified version without decomposition and rewrite loops.
    Good for testing and simple queries.

    Flow:
        START -> analyze -> retrieve -> evaluate -> generate -> END

    Returns:
        Compiled StateGraph.
    """
    builder = StateGraph(AgentState)

    builder.add_node("analyze", analyze_node)
    builder.add_node("retrieve", retrieve_node_sync)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("generate", generate_node)

    builder.set_entry_point("analyze")
    builder.add_edge("analyze", "retrieve")
    builder.add_edge("retrieve", "evaluate")
    builder.add_edge("evaluate", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


def build_agent_graph(
    enable_decomposition: bool = True, enable_rewrite: bool = True, max_rewrite_attempts: int = 2
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

    Returns:
        Compiled StateGraph.
    """
    builder = StateGraph(AgentState)

    builder.add_node("analyze", analyze_node)
    builder.add_node("decompose", decompose_node)
    builder.add_node("retrieve", retrieve_node_sync)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("generate", generate_node)

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

    graph = build_agent_graph(
        enable_decomposition=enable_decomposition,
        enable_rewrite=enable_rewrite,
        max_rewrite_attempts=max_rewrite_attempts,
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
    ):
        """Initialize the knowledge assistant.

        Args:
            enable_decomposition: Enable query decomposition.
            enable_rewrite: Enable query rewriting.
            max_rewrite_attempts: Maximum rewrite attempts.
        """
        self.enable_decomposition = enable_decomposition
        self.enable_rewrite = enable_rewrite
        self.max_rewrite_attempts = max_rewrite_attempts
        self._graph = None

    @property
    def graph(self):
        """Get or build the graph."""
        if self._graph is None:
            self._graph = build_agent_graph(
                enable_decomposition=self.enable_decomposition,
                enable_rewrite=self.enable_rewrite,
                max_rewrite_attempts=self.max_rewrite_attempts,
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
