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


def build_simple_graph(
    llm_client: Optional[LLMClient] = None,
    sufficiency_threshold: float = 0.5,
) -> StateGraph:
    """Build a simple linear graph for basic RAG flow.

    This is a simplified version without decomposition and rewrite loops.
    Good for testing and simple queries.

    Flow:
        START -> analyze -> retrieve -> evaluate -> generate -> END

    Args:
        llm_client: Optional LLM client for generation.
        sufficiency_threshold: Threshold for evaluating retrieval sufficiency.

    Returns:
        Compiled StateGraph.
    """
    builder = StateGraph(AgentState)

    builder.add_node("analyze", analyze_node)
    builder.add_node("retrieve", retrieve_node_sync)
    builder.add_node(
        "evaluate", partial(evaluate_node, threshold=sufficiency_threshold, llm_client=llm_client)
    )
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
    sufficiency_threshold: float = 0.5,
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
        sufficiency_threshold: Threshold for evaluating retrieval sufficiency.
        llm_client: Optional LLM client for LLM-based operations.

    Returns:
        Compiled StateGraph.
    """
    builder = StateGraph(AgentState)

    builder.add_node("analyze", analyze_node)
    builder.add_node("decompose", partial(decompose_node, llm_client=llm_client))
    builder.add_node("retrieve", retrieve_node_sync)
    builder.add_node(
        "evaluate", partial(evaluate_node, threshold=sufficiency_threshold, llm_client=llm_client)
    )
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
    settings = load_settings()

    config = config or {}
    enable_decomposition = config.get("enable_decomposition", settings.agent.enable_sub_query)
    enable_rewrite = config.get("enable_rewrite", settings.agent.enable_query_rewrite)
    max_rewrite_attempts = config.get("max_rewrite_attempts", settings.agent.max_rewrite_attempts)
    sufficiency_threshold = config.get(
        "sufficiency_threshold", settings.agent.sufficiency_threshold
    )
    use_llm = config.get("use_llm", True)

    llm_client = create_llm_client() if use_llm else None

    graph = build_agent_graph(
        enable_decomposition=enable_decomposition,
        enable_rewrite=enable_rewrite,
        max_rewrite_attempts=max_rewrite_attempts,
        sufficiency_threshold=sufficiency_threshold,
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
            "rewritten_sub_queries": final_state.get("rewritten_sub_queries"),
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
        enable_decomposition: Optional[bool] = None,
        enable_rewrite: Optional[bool] = None,
        max_rewrite_attempts: Optional[int] = None,
        sufficiency_threshold: Optional[float] = None,
        llm_client: Optional[LLMClient] = None,
        use_llm: bool = True,
        settings: Optional[Settings] = None,
    ):
        """Initialize the knowledge assistant.

        Args:
            enable_decomposition: Enable query decomposition (default from settings).
            enable_rewrite: Enable query rewriting (default from settings).
            max_rewrite_attempts: Maximum rewrite attempts (default from settings).
            sufficiency_threshold: Threshold for retrieval sufficiency (default from settings).
            llm_client: Optional LLM client instance.
            use_llm: Whether to use LLM for operations.
            settings: Optional settings object.
        """
        if settings is not None:
            self._settings = settings
        else:
            self._settings = load_settings()

        self.enable_decomposition = (
            enable_decomposition
            if enable_decomposition is not None
            else self._settings.agent.enable_sub_query
        )
        self.enable_rewrite = (
            enable_rewrite
            if enable_rewrite is not None
            else self._settings.agent.enable_query_rewrite
        )
        self.max_rewrite_attempts = (
            max_rewrite_attempts
            if max_rewrite_attempts is not None
            else self._settings.agent.max_rewrite_attempts
        )
        self.sufficiency_threshold = (
            sufficiency_threshold
            if sufficiency_threshold is not None
            else self._settings.agent.sufficiency_threshold
        )
        self.use_llm = use_llm

        self._graph = None
        self._tracer: Optional[LangFuseTracer] = None
        self._langfuse_client = None

        if self._settings.langfuse.enabled:
            self._tracer = init_langfuse(self._settings.langfuse)
            if self._tracer and self._tracer._client:
                self._langfuse_client = self._tracer._client

        if llm_client is not None:
            self._llm_client = llm_client
        elif use_llm:
            self._llm_client = create_llm_client(langfuse_client=self._langfuse_client)
        else:
            self._llm_client = None

    @property
    def graph(self):
        """Get or build the graph."""
        if self._graph is None:
            self._graph = build_agent_graph(
                enable_decomposition=self.enable_decomposition,
                enable_rewrite=self.enable_rewrite,
                max_rewrite_attempts=self.max_rewrite_attempts,
                sufficiency_threshold=self.sufficiency_threshold,
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

        try:
            if self._tracer:
                with self._tracer.trace(
                    name="agent_query",
                    input={"query": query},
                    metadata={"query": query},
                ) as trace_span:
                    final_state = self.graph.invoke(initial_state)
                    if isinstance(final_state, dict):
                        trace_span.metadata["output"] = final_state.get("final_response")
                        chunks = final_state.get("chunks", [])
                        if chunks:
                            contexts = [c.to_dict() if hasattr(c, "to_dict") else c for c in chunks]
                            trace_span.metadata["contexts"] = contexts
                            self._tracer.log_retrieval(query, contexts)
            else:
                final_state = self.graph.invoke(initial_state)
        finally:
            if self._tracer:
                self._tracer.flush()

        if isinstance(final_state, dict):
            return {
                "query": final_state.get("original_query", query),
                "response": final_state.get("final_response"),
                "citations": final_state.get("citations", []),
                "sub_queries": final_state.get("sub_queries", []),
                "rewritten_query": final_state.get("rewritten_query"),
                "rewritten_sub_queries": final_state.get("rewritten_sub_queries", []),
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
