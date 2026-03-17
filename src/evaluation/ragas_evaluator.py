"""Ragas-based evaluator for RAG quality assessment.

This evaluator uses the Ragas framework to compute LLM-as-Judge metrics:
- Faithfulness: Does the answer stick to the retrieved context?
- Answer Relevancy: Is the answer relevant to the query?
- Context Precision: Are the retrieved chunks relevant and well-ordered?

Key Difference from RAG MCP Server:
- RAG MCP Server uses chunk concatenation as answer (fallback), causing:
  - Faithfulness always high (answer is subset of context)
  - Answer Relevancy distorted (concatenated text may not answer the query)
- This evaluator uses LLM-generated answer for accurate evaluation.

Design Principles:
- Pluggable: Works with any OpenAI-compatible LLM provider.
- Config-Driven: LLM backend read from settings.yaml.
- Graceful Degradation: Clear ImportError if ragas not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"

SUPPORTED_METRICS = {FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION}


def _validate_ragas_import() -> None:
    """Validate that ragas is importable."""
    try:
        import ragas
    except ImportError as exc:
        raise ImportError(
            "The 'ragas' package is required for RagasEvaluator. "
            "Install it with: pip install ragas datasets"
        ) from exc


@dataclass
class EvaluationResult:
    """Result of evaluating a single test case."""

    query: str
    metrics: Dict[str, float] = field(default_factory=dict)
    contexts: List[str] = field(default_factory=list)
    answer: str = ""
    elapsed_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "contexts": self.contexts,
            "answer": self.answer[:500] + "..." if len(self.answer) > 500 else self.answer,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "error": self.error,
        }


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across all test cases."""

    evaluator: str = "RagasEvaluator"
    test_set_path: str = ""
    total_cases: int = 0
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    per_case_results: List[EvaluationResult] = field(default_factory=list)
    total_elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "test_set_path": self.test_set_path,
            "total_cases": self.total_cases,
            "aggregate_metrics": {k: round(v, 4) for k, v in self.aggregate_metrics.items()},
            "per_case_results": [r.to_dict() for r in self.per_case_results],
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
        }


class RagasEvaluator:
    """Evaluator that uses the Ragas framework for LLM-as-Judge metrics.

    Supported metrics:
        - faithfulness: Measures factual consistency with context.
        - answer_relevancy: Measures how relevant the answer is to the query.
        - context_precision: Measures relevance/ordering of retrieved chunks.

    Example::

        evaluator = RagasEvaluator(llm_config=settings.llm)
        result = evaluator.evaluate_single(
            query="What is RAG?",
            contexts=["RAG is a technique that combines retrieval with generation..."],
            answer="RAG stands for Retrieval-Augmented Generation...",
        )
        print(result.metrics)  # {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}
    """

    def __init__(
        self,
        llm_config: Any,
        metrics: Optional[Sequence[str]] = None,
        embedding_config: Optional[Any] = None,
    ) -> None:
        """Initialize RagasEvaluator.

        Args:
            llm_config: LLM configuration (LLMConfig from config.py).
            metrics: Metric names to compute. Defaults to all supported.
            embedding_config: Embedding configuration (for Answer Relevancy).

        Raises:
            ImportError: If ragas is not installed.
            ValueError: If unsupported metric names are requested.
        """
        _validate_ragas_import()

        self.llm_config = llm_config
        self.embedding_config = embedding_config

        if metrics is None:
            metrics = list(SUPPORTED_METRICS)

        normalized = [m.strip().lower() for m in (metrics or [])]
        if not normalized:
            normalized = sorted(SUPPORTED_METRICS)

        unsupported = [m for m in normalized if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )

        self._metric_names = normalized
        self._llm_wrapper = None
        self._embeddings_wrapper = None

    def evaluate_single(
        self,
        query: str,
        contexts: List[str],
        answer: str,
    ) -> EvaluationResult:
        """Evaluate a single query-context-answer triple.

        Args:
            query: The user query string.
            contexts: List of context strings (retrieved chunks).
            answer: The LLM-generated answer text.

        Returns:
            EvaluationResult with metrics.
        """
        import time

        t0 = time.monotonic()
        result = EvaluationResult(query=query, contexts=contexts, answer=answer)

        if not query or not query.strip():
            result.error = "Empty query"
            return result

        if not contexts:
            result.error = "Empty contexts"
            return result

        if not answer or not answer.strip():
            result.error = "Empty answer"
            return result

        try:
            metrics = self._run_ragas(query, contexts, answer)
            result.metrics = metrics
        except Exception as exc:
            logger.error("Ragas evaluation failed for query '%s': %s", query[:50], exc)
            result.error = str(exc)

        result.elapsed_ms = (time.monotonic() - t0) * 1000.0
        return result

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> EvaluationReport:
        """Evaluate a batch of test cases.

        Args:
            test_cases: List of dicts with 'query', 'contexts', 'answer' keys.

        Returns:
            EvaluationReport with aggregated metrics.
        """
        import time

        report = EvaluationReport(evaluator="RagasEvaluator", total_cases=len(test_cases))

        t0 = time.monotonic()

        for tc in test_cases:
            query = tc.get("query", "")
            contexts = tc.get("contexts", [])
            answer = tc.get("answer", "")

            if isinstance(contexts[0], dict) if contexts else False:
                contexts = [c.get("text", str(c)) for c in contexts]

            result = self.evaluate_single(query, contexts, answer)
            report.per_case_results.append(result)

        report.total_elapsed_ms = (time.monotonic() - t0) * 1000.0
        report.aggregate_metrics = self._aggregate_metrics(report.per_case_results)

        return report

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        answer: str,
    ) -> Dict[str, float]:
        """Execute Ragas metrics and return normalized scores."""
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision

        llm = self._get_llm_wrapper()
        embeddings = self._get_embeddings_wrapper()

        scores: Dict[str, float] = {}

        for metric_name in self._metric_names:
            try:
                if metric_name == FAITHFULNESS:
                    m = Faithfulness(llm=llm)
                    result = m.score(
                        user_input=query,
                        response=answer,
                        retrieved_contexts=contexts,
                    )
                elif metric_name == ANSWER_RELEVANCY:
                    m = AnswerRelevancy(llm=llm, embeddings=embeddings)
                    result = m.score(user_input=query, response=answer)
                elif metric_name == CONTEXT_PRECISION:
                    m = ContextPrecision(llm=llm)
                    result = m.score(
                        user_input=query,
                        response=answer,
                        retrieved_contexts=contexts,
                    )
                else:
                    continue

                scores[metric_name] = float(result) if result is not None else 0.0

            except Exception as exc:
                logger.warning("Metric '%s' failed: %s", metric_name, exc)
                scores[metric_name] = 0.0

        return scores

    def _get_llm_wrapper(self) -> Any:
        """Build Ragas LLM wrapper from config."""
        if self._llm_wrapper is not None:
            return self._llm_wrapper

        from openai import AsyncOpenAI
        from ragas.llms import OpenAILLM

        if self.llm_config is None:
            raise ValueError("LLM config required for Ragas evaluation")

        provider = getattr(self.llm_config, "provider", "openai").lower()
        api_key = getattr(self.llm_config, "api_key", None)
        base_url = getattr(self.llm_config, "base_url", None)
        model = getattr(self.llm_config, "model", "gpt-4o")

        if provider in ("openai", "deepseek") or base_url:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            raise ValueError(
                f"Unsupported LLM provider for Ragas: '{provider}'. "
                "Supported: openai, deepseek, or any OpenAI-compatible provider"
            )

        self._llm_wrapper = OpenAILLM(model=model, client=client)
        return self._llm_wrapper

    def _get_embeddings_wrapper(self) -> Any:
        """Build Ragas Embeddings wrapper from config."""
        if self._embeddings_wrapper is not None:
            return self._embeddings_wrapper

        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings

        if self.embedding_config is not None:
            api_key = getattr(self.embedding_config, "api_key", None)
            base_url = getattr(self.embedding_config, "base_url", None)
            model = getattr(self.embedding_config, "model", "text-embedding-3-small")
        else:
            api_key = getattr(self.llm_config, "api_key", None)
            base_url = getattr(self.llm_config, "base_url", None)
            model = "text-embedding-3-small"

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._embeddings_wrapper = OpenAIEmbeddings(model=model, client=client)
        return self._embeddings_wrapper

    @staticmethod
    def _aggregate_metrics(results: List[EvaluationResult]) -> Dict[str, float]:
        """Compute average metrics across all results."""
        if not results:
            return {}

        all_keys: set = set()
        for r in results:
            all_keys.update(r.metrics.keys())

        averages: Dict[str, float] = {}
        for key in sorted(all_keys):
            values = [r.metrics[key] for r in results if key in r.metrics and r.metrics[key] > 0]
            averages[key] = sum(values) / len(values) if values else 0.0

        return averages


def create_ragas_evaluator(
    llm_config: Optional[Any] = None,
    metrics: Optional[Sequence[str]] = None,
) -> Optional[RagasEvaluator]:
    """Create RagasEvaluator from configuration.

    Args:
        llm_config: LLM configuration. If None, loads from settings.
        metrics: Metrics to compute. Defaults to all supported.

    Returns:
        RagasEvaluator instance, or None if no valid config.
    """
    if llm_config is None:
        from src.core.config import load_settings

        settings = load_settings()
        llm_config = settings.llm

    if not getattr(llm_config, "api_key", None):
        logger.warning("No API key configured, cannot create RagasEvaluator")
        return None

    try:
        return RagasEvaluator(llm_config=llm_config, metrics=metrics)
    except ImportError as exc:
        logger.warning("Ragas not installed: %s", exc)
        return None
