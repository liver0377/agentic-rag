"""LangFuse client for tracing and observability.

This module provides integration with LangFuse for:
- Tracing agent execution
- Logging LLM calls
- Tracking retrieval results
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.config import LangFuseConfig


@dataclass
class Span:
    """Represents a traced span."""

    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata or {},
        }


class LangFuseTracer:
    """LangFuse tracer for agent observability.

    This class provides tracing capabilities for the agent.
    When LangFuse is disabled, it acts as a no-op tracer.

    Example:
        >>> tracer = LangFuseTracer(config)
        >>> with tracer.trace("query") as span:
        ...     span.metadata["query"] = "What is RAG?"
    """

    def __init__(self, config: LangFuseConfig):
        """Initialize the tracer.

        Args:
            config: LangFuse configuration.
        """
        self.config = config
        self.enabled = config.enabled
        self._client = None
        self._trace = None
        self._spans: List[Span] = []

    def init(self) -> None:
        """Initialize LangFuse client."""
        if not self.enabled:
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self.config.public_key,
                secret_key=self.config.secret_key,
                host=self.config.host,
            )
        except ImportError:
            self.enabled = False

    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start a new trace.

        Args:
            name: Trace name.
            metadata: Optional metadata.
        """
        if self.enabled and self._client:
            self._trace = self._client.trace(
                name=name,
                metadata=metadata or {},
            )
        self._spans = []

    def span(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Span:
        """Create a span.

        Args:
            name: Span name.
            metadata: Optional metadata.

        Returns:
            Span object.
        """
        span = Span(
            name=name,
            start_time=datetime.now(),
            metadata=metadata or {},
        )
        self._spans.append(span)

        if self.enabled and self._trace:
            self._trace.span(name=name, metadata=metadata or {})

        return span

    def end_span(self, span: Span) -> None:
        """End a span.

        Args:
            span: Span to end.
        """
        span.end_time = datetime.now()

    def log_generation(
        self,
        name: str,
        prompt: str,
        response: str,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an LLM generation.

        Args:
            name: Generation name.
            prompt: Input prompt.
            response: Output response.
            model: Model name.
            metadata: Optional metadata.
        """
        if self.enabled and self._trace:
            self._trace.generation(
                name=name,
                input=prompt,
                output=response,
                model=model,
                metadata=metadata or {},
            )

    def log_retrieval(
        self, query: str, chunks: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a retrieval operation.

        Args:
            query: Search query.
            chunks: Retrieved chunks.
            metadata: Optional metadata.
        """
        if self.enabled and self._trace:
            self._trace.span(
                name="retrieval",
                input=query,
                output={"chunks": chunks, "count": len(chunks)},
                metadata=metadata or {},
            )

    def end_trace(self) -> None:
        """End the current trace."""
        self._trace = None
        self._spans = []

    def get_trace_url(self) -> Optional[str]:
        """Get the URL to view the trace.

        Returns:
            Trace URL or None.
        """
        if self.enabled and self._trace:
            return self._trace.get_trace_url()
        return None


def init_langfuse(config: LangFuseConfig) -> LangFuseTracer:
    """Initialize LangFuse tracer.

    Args:
        config: LangFuse configuration.

    Returns:
        Initialized tracer.
    """
    tracer = LangFuseTracer(config)
    tracer.init()
    return tracer
