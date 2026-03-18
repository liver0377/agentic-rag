"""Data collector for Ragas evaluation.

Collects query-context-answer triples from:
1. Session history (in-memory)
2. LangFuse traces (if configured)
3. LangFuse Datasets (with human annotations)
4. JSON file import

Data Format:
{
  "test_cases": [
    {
      "query": "user question",
      "contexts": [{"id": "...", "text": "...", "score": 0.9, "metadata": {...}}],
      "answer": "LLM generated answer",
      "pass_rate": "Pass",
      "ragas_scores": {"faithfulness": 0.9, "answer_relevancy": 0.8},
      "metadata": {"trace_id": "xxx", "timestamp": "..."}
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
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single evaluation test case."""

    query: str
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    pass_rate: Optional[str] = None
    ragas_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "contexts": self.contexts,
            "answer": self.answer,
            "pass_rate": self.pass_rate,
            "ragas_scores": self.ragas_scores,
            "metadata": self.metadata,
        }

    def to_ragas_format(self) -> Dict[str, Any]:
        """Convert to RAGAS-compatible format (contexts as list of strings)."""
        contexts_text = []
        for c in self.contexts:
            if isinstance(c, dict):
                contexts_text.append(c.get("text", str(c)))
            else:
                contexts_text.append(str(c))
        return {
            "query": self.query,
            "contexts": contexts_text,
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        contexts = data.get("contexts", [])
        if contexts and isinstance(contexts[0], str):
            contexts = [{"text": c} for c in contexts]

        return cls(
            query=data.get("query", ""),
            contexts=contexts,
            answer=data.get("answer", ""),
            pass_rate=data.get("pass_rate"),
            ragas_scores=data.get("ragas_scores", {}),
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
        contexts: Union[List[str], List[Dict[str, Any]]],
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add data from a session (in-memory collection).

        Args:
            query: User query.
            contexts: Retrieved context chunks (strings or dicts).
            answer: LLM generated answer.
            metadata: Optional metadata (session_id, etc.).
        """
        normalized_contexts: List[Dict[str, Any]] = []
        for c in contexts:
            if isinstance(c, str):
                normalized_contexts.append({"text": c})
            else:
                normalized_contexts.append(c)

        tc = TestCase(
            query=query,
            contexts=normalized_contexts,
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

            traces = client.api.trace.list(
                from_timestamp=from_date,
                limit=limit,
                tags=tags,
            )

            for trace in traces.data:
                tc = self._extract_test_case_from_trace(client, trace)
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

    def _extract_test_case_from_trace(self, client: Any, trace: Any) -> Optional[TestCase]:
        """Extract test case from a LangFuse trace.

        The trace object from API has:
        - id, name, timestamp, input, output, metadata
        - observations fetched separately via api.observations.get_many()
        """
        try:
            query = ""
            contexts: List[Dict[str, Any]] = []
            answer = ""

            input_data = getattr(trace, "input", None)
            if input_data:
                if isinstance(input_data, dict):
                    query = input_data.get("query", input_data.get("question", str(input_data)))
                elif isinstance(input_data, str):
                    query = input_data

            output_data = getattr(trace, "output", None)
            if output_data:
                if isinstance(output_data, dict):
                    answer = output_data.get(
                        "response",
                        output_data.get("answer", output_data.get("content", str(output_data))),
                    )
                elif isinstance(output_data, str):
                    answer = output_data

            trace_id = getattr(trace, "id", "")
            if trace_id:
                try:
                    observations = client.api.observations.get_many(trace_id=trace_id, limit=50)
                    for obs in observations.data:
                        obs_name = getattr(obs, "name", "").lower()
                        obs_type = getattr(obs, "type", "")

                        if not query:
                            obs_input = getattr(obs, "input", None)
                            if obs_input:
                                if isinstance(obs_input, dict):
                                    query = obs_input.get(
                                        "query", obs_input.get("question", str(obs_input))
                                    )
                                elif isinstance(obs_input, str):
                                    query = obs_input

                        if "retrieval" in obs_name or obs_type == "RETRIEVAL":
                            obs_output = getattr(obs, "output", None)
                            if isinstance(obs_output, dict):
                                chunks = obs_output.get("chunks", [])
                                for chunk in chunks:
                                    if isinstance(chunk, dict):
                                        contexts.append(chunk)
                                    elif isinstance(chunk, str):
                                        contexts.append({"text": chunk})
                            elif isinstance(obs_output, list):
                                for item in obs_output:
                                    if isinstance(item, dict):
                                        contexts.append(item)
                                    elif isinstance(item, str):
                                        contexts.append({"text": item})

                        if not answer:
                            obs_output = getattr(obs, "output", None)
                            if obs_output:
                                if isinstance(obs_output, dict):
                                    answer = obs_output.get(
                                        "response",
                                        obs_output.get("answer", obs_output.get("content", "")),
                                    )
                                elif isinstance(obs_output, str):
                                    answer = obs_output
                except Exception as e:
                    logger.debug("Failed to fetch observations for trace %s: %s", trace_id, e)

            if not query or not answer:
                return None

            trace_meta = getattr(trace, "metadata", None) or {}
            if not contexts and "contexts" in trace_meta:
                ctx_data = trace_meta["contexts"]
                if isinstance(ctx_data, list):
                    for c in ctx_data:
                        if isinstance(c, dict):
                            contexts.append(c)
                        elif isinstance(c, str):
                            contexts.append({"text": c})

            pass_rate = None
            scores = getattr(trace, "scores", None) or []
            for score in scores:
                score_name = getattr(score, "name", "")
                if score_name.lower() in ("pass_rate", "passrate", "pass"):
                    val = getattr(score, "value", None)
                    if val is not None:
                        if isinstance(val, str):
                            pass_rate = val
                        elif isinstance(val, (int, float)):
                            pass_rate = "Pass" if val >= 0.5 else "Fail"

            return TestCase(
                query=query,
                contexts=contexts,
                answer=answer,
                pass_rate=pass_rate,
                metadata={
                    "trace_id": trace_id,
                    "timestamp": str(getattr(trace, "timestamp", "")),
                    "name": getattr(trace, "name", ""),
                },
            )

        except Exception as exc:
            logger.debug("Failed to extract test case from trace: %s", exc)

        return None

    def collect_from_dataset(self, dataset_name: str) -> TestSet:
        """Collect data from a Langfuse Dataset.

        Args:
            dataset_name: Name of the Langfuse dataset.

        Returns:
            TestSet with collected data including human annotations.
            Fetches contexts and answer from source trace if not in dataset item.
        """
        if not self.langfuse_config or not getattr(self.langfuse_config, "enabled", False):
            logger.warning("LangFuse not configured, cannot collect from dataset")
            return TestSet(source="langfuse_dataset")

        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=self.langfuse_config.public_key,
                secret_key=self.langfuse_config.secret_key,
                host=self.langfuse_config.host,
            )

            dataset = client.get_dataset(name=dataset_name)
            test_cases: List[TestCase] = []

            for item in dataset.items:
                input_data = getattr(item, "input", {}) or {}
                expected_output = getattr(item, "expected_output", None)
                source_trace_id = getattr(item, "source_trace_id", None)

                query = ""
                if isinstance(input_data, dict):
                    query = input_data.get("query", input_data.get("question", ""))

                contexts: List[Dict[str, Any]] = []
                ctx_data = input_data.get("contexts", [])
                if isinstance(ctx_data, list):
                    for c in ctx_data:
                        if isinstance(c, dict):
                            contexts.append(c)
                        elif isinstance(c, str):
                            contexts.append({"text": c})

                answer = input_data.get("answer", "")

                if source_trace_id and (not contexts or not answer):
                    try:
                        trace = client.api.trace.get(source_trace_id)

                        if not query:
                            trace_input = getattr(trace, "input", None)
                            if trace_input:
                                if isinstance(trace_input, dict):
                                    query = trace_input.get(
                                        "query", trace_input.get("question", "")
                                    )
                                elif isinstance(trace_input, str):
                                    query = trace_input

                        if not answer:
                            trace_output = getattr(trace, "output", None)
                            if trace_output:
                                if isinstance(trace_output, dict):
                                    answer = trace_output.get(
                                        "response",
                                        trace_output.get("answer", trace_output.get("content", "")),
                                    )
                                elif isinstance(trace_output, str):
                                    answer = trace_output

                        if not contexts:
                            trace_meta = getattr(trace, "metadata", {}) or {}
                            ctx_meta = trace_meta.get("contexts", [])
                            if isinstance(ctx_meta, list):
                                for c in ctx_meta:
                                    if isinstance(c, dict):
                                        contexts.append(c)
                                    elif isinstance(c, str):
                                        contexts.append({"text": c})

                    except Exception as e:
                        logger.debug("Failed to fetch source trace %s: %s", source_trace_id, e)

                pass_rate = None
                scores = getattr(item, "scores", None) or []
                for score in scores:
                    score_name = getattr(score, "name", "")
                    if score_name.lower() in ("pass_rate", "passrate", "pass"):
                        val = getattr(score, "value", None)
                        if val is not None:
                            if isinstance(val, str):
                                pass_rate = val
                            elif isinstance(val, (int, float)):
                                pass_rate = "Pass" if val >= 0.5 else "Fail"

                if query:
                    tc = TestCase(
                        query=query,
                        contexts=contexts,
                        answer=answer,
                        pass_rate=pass_rate,
                        metadata={
                            "dataset_item_id": getattr(item, "id", ""),
                            "dataset_name": dataset_name,
                            "source_trace_id": source_trace_id,
                        },
                    )
                    test_cases.append(tc)

            logger.info("Collected %d test cases from dataset '%s'", len(test_cases), dataset_name)
            return TestSet(
                test_cases=test_cases,
                source="langfuse_dataset",
                description=f"Collected from Langfuse dataset: {dataset_name}",
            )

        except ImportError:
            logger.warning("LangFuse SDK not installed")
            return TestSet(source="langfuse_dataset")
        except Exception as exc:
            logger.error("Failed to collect from Langfuse dataset: %s", exc)
            return TestSet(source="langfuse_dataset")

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
