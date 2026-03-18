"""LangFuse client for tracing and observability.

This module provides integration with LangFuse for:
- Tracing agent execution
- Logging LLM calls
- Tracking retrieval results

Compatible with Langfuse SDK v4.0+
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from src.core.config import LangFuseConfig


@dataclass
class Span:
    """Represents a traced span."""

    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        >>> with tracer.trace("agent_query") as span:
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
        self._trace_url: Optional[str] = None
        self._trace_id: Optional[str] = None

    def init(self) -> None:
        """Initialize LangFuse client."""
        if not self.enabled:
            return

        try:
            if self.config.public_key:
                os.environ["LANGFUSE_PUBLIC_KEY"] = self.config.public_key
            if self.config.secret_key:
                os.environ["LANGFUSE_SECRET_KEY"] = self.config.secret_key
            if self.config.host:
                os.environ["LANGFUSE_HOST"] = self.config.host

            from langfuse import Langfuse

            self._client = Langfuse()
        except ImportError:
            print("Warning: langfuse package not installed. Tracing disabled.")
            self.enabled = False
        except Exception as e:
            print(f"Warning: Langfuse initialization failed: {e}. Tracing disabled.")
            self.enabled = False

    @contextmanager
    def trace(
        self,
        name: str,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Context manager for tracing.

        Args:
            name: Trace name.
            input: Input data (e.g., user query).
            output: Output data (e.g., response).
            metadata: Optional metadata.

        Yields:
            Span object.
        """
        span = Span(name=name, start_time=datetime.now(), metadata=metadata or {})
        span.metadata["_input"] = input
        self._trace_url = None
        self._trace_id = None

        if not self.enabled or not self._client:
            try:
                yield span
            finally:
                span.end_time = datetime.now()
            return

        try:
            with self._client.start_as_current_observation(
                as_type="span",
                name=name,
                input=input,
                metadata=metadata or {},
            ) as observation:
                self._trace_id = observation.trace_id
                try:
                    yield span
                finally:
                    span.end_time = datetime.now()
                    final_output = output if output is not None else span.metadata.get("output")
                    final_metadata = {
                        k: v for k, v in span.metadata.items() if not k.startswith("_")
                    }
                    update_kwargs = {}
                    if final_output is not None:
                        update_kwargs["output"] = final_output
                    if final_metadata:
                        update_kwargs["metadata"] = final_metadata
                    if update_kwargs:
                        observation.update(**update_kwargs)

            if self._trace_id:
                self._trace_url = f"{self.config.host}/trace/{self._trace_id}"
        except Exception as e:
            print(f"Warning: Trace failed: {e}")
            yield span
            span.end_time = datetime.now()

    @contextmanager
    def span(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Generator[Span, None, None]:
        """Context manager for creating a child span.

        Args:
            name: Span name.
            metadata: Optional metadata.

        Yields:
            Span object.
        """
        span = Span(name=name, start_time=datetime.now(), metadata=metadata or {})

        if not self.enabled or not self._client:
            try:
                yield span
            finally:
                span.end_time = datetime.now()
            return

        try:
            with self._client.start_as_current_observation(
                as_type="span", name=name, metadata=metadata or {}
            ):
                try:
                    yield span
                finally:
                    span.end_time = datetime.now()
        except Exception as e:
            print(f"Warning: Span failed: {e}")
            yield span
            span.end_time = datetime.now()

    @contextmanager
    def generation(
        self,
        name: str,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Context manager for logging an LLM generation.

        Args:
            name: Generation name.
            model: Model name.
            metadata: Optional metadata.

        Yields:
            Span object.
        """
        span = Span(name=name, start_time=datetime.now(), metadata=metadata or {})

        if not self.enabled or not self._client:
            try:
                yield span
            finally:
                span.end_time = datetime.now()
            return

        try:
            with self._client.start_as_current_observation(
                as_type="generation",
                name=name,
                model=model,
                metadata=metadata or {},
            ):
                try:
                    yield span
                finally:
                    span.end_time = datetime.now()
        except Exception as e:
            print(f"Warning: Generation logging failed: {e}")
            yield span
            span.end_time = datetime.now()

    def log_retrieval(
        self, query: str, chunks: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a retrieval operation. Must be called within a trace context.

        Args:
            query: Search query.
            chunks: Retrieved chunks.
            metadata: Optional metadata.
        """
        if not self.enabled or not self._client:
            return

        try:
            with self._client.start_as_current_observation(
                as_type="span",
                name="retrieval",
                input=query,
                output={"chunks": chunks, "count": len(chunks)},
                metadata=metadata or {},
            ):
                pass
        except Exception as e:
            print(f"Warning: Failed to log retrieval: {e}")

    def get_trace_url(self) -> Optional[str]:
        """Get the URL to view the trace.

        Returns:
            Trace URL or None.
        """
        return self._trace_url

    def get_trace_id(self) -> Optional[str]:
        """Get the trace ID.

        Returns:
            Trace ID or None.
        """
        return self._trace_id

    def flush(self) -> None:
        """Flush pending traces."""
        if self._client and self.enabled:
            try:
                self._client.flush()
            except Exception:
                pass


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
