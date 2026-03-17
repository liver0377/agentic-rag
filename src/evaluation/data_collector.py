"""Data collector for Ragas evaluation.

Collects query-context-answer triples from:
1. Session history (in-memory)
2. LangFuse traces (if configured)
3. JSON file import

Data Format:
{
  "test_cases": [
    {
      "query": "user question",
      "contexts": ["chunk1 text", "chunk2 text"],
      "answer": "LLM generated answer",
      "reference_answer": "human annotated (optional)",
      "metadata": {"session_id": "xxx", "timestamp": "..."}
    }
  ]
}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single evaluation test case."""

    query: str
    contexts: List[str] = field(default_factory=list)
    answer: str = ""
    reference_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "contexts": self.contexts,
            "answer": self.answer,
            "reference_answer": self.reference_answer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        contexts = data.get("contexts", [])
        if contexts and isinstance(contexts[0], dict):
            contexts = [c.get("text", str(c)) for c in contexts]

        return cls(
            query=data.get("query", ""),
            contexts=contexts,
            answer=data.get("answer", ""),
            reference_answer=data.get("reference_answer"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TestSet:
    """Collection of test cases for evaluation."""

    test_cases: List[TestCase] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "unknown"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "created_at": self.created_at,
            "source": self.source,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestSet":
        return cls(
            test_cases=[TestCase.from_dict(tc) for tc in data.get("test_cases", [])],
            created_at=data.get("created_at", datetime.now().isoformat()),
            source=data.get("source", "unknown"),
            description=data.get("description", ""),
        )

    def save(self, path: str | Path) -> None:
        """Save test set to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("Saved %d test cases to %s", len(self.test_cases), path)

    @classmethod
    def load(cls, path: str | Path) -> "TestSet":
        """Load test set from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Test set not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)


class DataCollector:
    """Collects evaluation data from various sources."""

    def __init__(self, langfuse_config: Optional[Any] = None):
        """Initialize data collector.

        Args:
            langfuse_config: Optional LangFuse configuration for trace collection.
        """
        self.langfuse_config = langfuse_config
        self._session_data: List[TestCase] = []

    def add_session_data(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add data from a session (in-memory collection).

        Args:
            query: User query.
            contexts: Retrieved context chunks.
            answer: LLM generated answer.
            metadata: Optional metadata (session_id, etc.).
        """
        tc = TestCase(
            query=query,
            contexts=contexts,
            answer=answer,
            metadata=metadata or {},
        )
        self._session_data.append(tc)
        logger.debug("Added session data: query='%s'", query[:50])

    def get_session_data(self) -> TestSet:
        """Get all collected session data as a TestSet."""
        return TestSet(
            test_cases=list(self._session_data),
            source="session",
            description="Collected from session history",
        )

    def clear_session_data(self) -> None:
        """Clear in-memory session data."""
        self._session_data.clear()

    def collect_from_langfuse(
        self,
        days: int = 7,
        limit: int = 100,
        tags: Optional[List[str]] = None,
    ) -> TestSet:
        """Collect data from LangFuse traces.

        Args:
            days: Number of days to look back.
            limit: Maximum number of traces to collect.
            tags: Optional tags to filter by.

        Returns:
            TestSet with collected data.
        """
        if not self.langfuse_config or not getattr(self.langfuse_config, "enabled", False):
            logger.warning("LangFuse not configured, cannot collect traces")
            return TestSet(source="langfuse")

        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=self.langfuse_config.public_key,
                secret_key=self.langfuse_config.secret_key,
                host=self.langfuse_config.host,
            )

            test_cases: List[TestCase] = []
            from_date = datetime.now() - timedelta(days=days)

            traces = client.get_traces(
                from_timestamp=from_date.isoformat(),
                limit=limit,
                tags=tags,
            )

            for trace in traces.data:
                tc = self._extract_test_case_from_trace(trace)
                if tc:
                    test_cases.append(tc)

            logger.info("Collected %d test cases from LangFuse", len(test_cases))
            return TestSet(
                test_cases=test_cases,
                source="langfuse",
                description=f"Collected from LangFuse (last {days} days)",
            )

        except ImportError:
            logger.warning("LangFuse SDK not installed")
            return TestSet(source="langfuse")
        except Exception as exc:
            logger.error("Failed to collect from LangFuse: %s", exc)
            return TestSet(source="langfuse")

    def _extract_test_case_from_trace(self, trace: Any) -> Optional[TestCase]:
        """Extract test case from a LangFuse trace."""
        try:
            query = ""
            contexts: List[str] = []
            answer = ""

            for observation in getattr(trace, "observations", []):
                obs_type = getattr(observation, "type", "")
                name = getattr(observation, "name", "").lower()

                if "query" in name or "question" in name or obs_type == "GENERATION":
                    query = getattr(observation, "input", "") or ""
                    if isinstance(query, dict):
                        query = query.get("query", "") or query.get("question", "") or str(query)

                if "context" in name or "retriev" in name or obs_type == "RETRIEVAL":
                    output = getattr(observation, "output", None)
                    if isinstance(output, list):
                        for item in output:
                            if isinstance(item, dict):
                                text = item.get("text", item.get("content", ""))
                                if text:
                                    contexts.append(text)
                            elif isinstance(item, str):
                                contexts.append(item)

                if "answer" in name or "response" in name or "generator" in name:
                    output = getattr(observation, "output", "")
                    if isinstance(output, dict):
                        answer = output.get(
                            "response", output.get("answer", output.get("content", ""))
                        )
                    elif isinstance(output, str):
                        answer = output

            if query and answer:
                return TestCase(
                    query=query,
                    contexts=contexts,
                    answer=answer,
                    metadata={
                        "trace_id": getattr(trace, "id", ""),
                        "timestamp": getattr(trace, "timestamp", ""),
                    },
                )

        except Exception as exc:
            logger.debug("Failed to extract test case from trace: %s", exc)

        return None

    def import_from_json(self, path: str | Path) -> TestSet:
        """Import test set from JSON file.

        Supports formats:
        - Native format (test_cases array)
        - Simple format (array of {query, answer, contexts})
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "test_cases" in data:
            return TestSet.from_dict(data)

        if isinstance(data, list):
            test_cases = [TestCase.from_dict(item) for item in data]
            return TestSet(
                test_cases=test_cases,
                source="import",
                description=f"Imported from {path.name}",
            )

        raise ValueError(f"Unknown JSON format in {path}")


def create_data_collector(langfuse_config: Optional[Any] = None) -> DataCollector:
    """Create data collector with optional LangFuse config.

    Args:
        langfuse_config: Optional LangFuse configuration.

    Returns:
        DataCollector instance.
    """
    if langfuse_config is None:
        try:
            from src.core.config import load_settings

            settings = load_settings()
            langfuse_config = settings.langfuse
        except Exception:
            pass

    return DataCollector(langfuse_config=langfuse_config)
