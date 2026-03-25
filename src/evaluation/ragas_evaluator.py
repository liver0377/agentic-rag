"""Ragas-based evaluator for RAG quality assessment.

This evaluator uses the Ragas framework to compute LLM-as-Judge metrics:
- Faithfulness: Does the answer stick to the retrieved context?
- Answer Relevancy: Is the answer relevant to the query?
- Context Precision (without reference): Are the retrieved chunks relevant?

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


class EmbeddingsAdapter:
    """Adapter for ragas 0.4.x OpenAIEmbeddings to add embed_query method."""

    def __init__(self, embeddings: Any):
        self._embeddings = embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_texts(texts)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embeddings, name)


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
    pass_rate: Optional[str] = None
    elapsed_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "contexts": self.contexts,
            "answer": self.answer[:500] + "..." if len(self.answer) > 500 else self.answer,
            "pass_rate": self.pass_rate,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "error": self.error,
        }


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across all test cases."""

    evaluator: str = "RagasEvaluator"
    test_set_path: str = ""
    total_cases: int = 0
    pass_count: int = 0
    fail_count: int = 0
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    pass_group_metrics: Dict[str, float] = field(default_factory=dict)
    fail_group_metrics: Dict[str, float] = field(default_factory=dict)
    per_case_results: List[EvaluationResult] = field(default_factory=list)
    total_elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "test_set_path": self.test_set_path,
            "total_cases": self.total_cases,
            "pass_rate_summary": {
                "pass": self.pass_count,
                "fail": self.fail_count,
                "unlabeled": self.total_cases - self.pass_count - self.fail_count,
            },
            "aggregate_metrics": {k: round(v, 4) for k, v in self.aggregate_metrics.items()},
            "pass_group_metrics": {k: round(v, 4) for k, v in self.pass_group_metrics.items()},
            "fail_group_metrics": {k: round(v, 4) for k, v in self.fail_group_metrics.items()},
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
                        May also include 'pass_rate' for human annotations.

        Returns:
            EvaluationReport with aggregated metrics.
        """
        import time

        report = EvaluationReport(evaluator="RagasEvaluator", total_cases=len(test_cases))

        t0 = time.monotonic()
        total = len(test_cases)

        for i, tc in enumerate(test_cases, 1):
            query = tc.get("query", "")
            contexts = tc.get("contexts", [])
            answer = tc.get("answer", "")
            pass_rate = tc.get("pass_rate")

            if isinstance(contexts[0], dict) if contexts else False:
                contexts = [c.get("text", str(c)) for c in contexts]

            print(f"  [{i}/{total}] Evaluating: {query[:50]}{'...' if len(query) > 50 else ''}")
            result = self.evaluate_single(query, contexts, answer)
            result.pass_rate = pass_rate
            report.per_case_results.append(result)
            print(f"  [{i}/{total}] Done - elapsed: {result.elapsed_ms:.0f}ms")

            if pass_rate == "Pass":
                report.pass_count += 1
            elif pass_rate == "Fail":
                report.fail_count += 1

        report.total_elapsed_ms = (time.monotonic() - t0) * 1000.0
        report.aggregate_metrics = self._aggregate_metrics(report.per_case_results)

        pass_results = [r for r in report.per_case_results if r.pass_rate == "Pass"]
        fail_results = [r for r in report.per_case_results if r.pass_rate == "Fail"]

        report.pass_group_metrics = self._aggregate_metrics(pass_results)
        report.fail_group_metrics = self._aggregate_metrics(fail_results)

        return report

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        answer: str,
    ) -> Dict[str, float]:
        """Execute Ragas metrics and return normalized scores."""
        import asyncio
        from ragas import SingleTurnSample
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference

        llm = self._get_llm_wrapper()
        embeddings = self._get_embeddings_wrapper()

        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
        )

        scores: Dict[str, float] = {}

        for metric_name in self._metric_names:
            try:
                if metric_name == FAITHFULNESS:
                    m = Faithfulness(llm=llm)
                elif metric_name == ANSWER_RELEVANCY:
                    m = AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1)
                elif metric_name == CONTEXT_PRECISION:
                    m = LLMContextPrecisionWithoutReference(llm=llm)
                else:
                    continue

                try:
                    result = asyncio.run(m.single_turn_ascore(sample))
                except RuntimeError:
                    result = m.single_turn_score(sample)

                scores[metric_name] = float(result) if result is not None else 0.0

            except Exception as exc:
                logger.warning("Metric '%s' failed: %s", metric_name, exc)
                scores[metric_name] = 0.0

        return scores

    def _get_llm_wrapper(self) -> Any:
        if self._llm_wrapper is not None:
            return self._llm_wrapper

        from openai import OpenAI
        from ragas.llms import llm_factory

        if self.llm_config is None:
            raise ValueError("LLM config required for Ragas evaluation")

        api_key = getattr(self.llm_config, "api_key", None)
        base_url = getattr(self.llm_config, "base_url", None)
        model = getattr(self.llm_config, "model", "gpt-4o")
        max_tokens = getattr(self.llm_config, "max_tokens", 8192)

        if not api_key:
            raise ValueError("API key required for Ragas evaluation")

        client = OpenAI(api_key=api_key, base_url=base_url)
        self._llm_wrapper = llm_factory(model=model, client=client, max_tokens=max_tokens)
        return self._llm_wrapper

    def _get_embeddings_wrapper(self) -> Any:
        if self._embeddings_wrapper is not None:
            return self._embeddings_wrapper

        from openai import OpenAI
        from ragas.embeddings import OpenAIEmbeddings

        api_key = None
        base_url = None
        model = "text-embedding-3-small"

        if self.embedding_config is not None:
            api_key = getattr(self.embedding_config, "api_key", None)
            base_url = getattr(self.embedding_config, "base_url", None)
            model = getattr(self.embedding_config, "model", "text-embedding-3-small")

        if not api_key:
            api_key = getattr(self.llm_config, "api_key", None)
            base_url = getattr(self.llm_config, "base_url", None)

        if not api_key:
            raise ValueError("API key required for Ragas embeddings")

        client = OpenAI(api_key=api_key, base_url=base_url)
        base_embeddings = OpenAIEmbeddings(model=model, client=client)
        self._embeddings_wrapper = EmbeddingsAdapter(base_embeddings)
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
    embedding_config: Optional[Any] = None,
    metrics: Optional[Sequence[str]] = None,
) -> Optional[RagasEvaluator]:
    """Create RagasEvaluator from configuration.

    Args:
        llm_config: LLM configuration. If None, loads from settings.evaluation.llm.
        embedding_config: Embedding configuration. If None, loads from settings.evaluation.embedding.
        metrics: Metrics to compute. Defaults to all supported.

    Returns:
        RagasEvaluator instance, or None if no valid config.
    """
    if llm_config is None or embedding_config is None:
        from src.core.config import load_settings

        settings = load_settings()
        if llm_config is None:
            llm_config = settings.evaluation.llm
        if embedding_config is None:
            embedding_config = settings.evaluation.embedding

    if not getattr(llm_config, "api_key", None):
        logger.warning("No API key configured, cannot create RagasEvaluator")
        return None

    try:
        return RagasEvaluator(
            llm_config=llm_config, embedding_config=embedding_config, metrics=metrics
        )
    except ImportError as exc:
        logger.warning("Ragas not installed: %s", exc)
        return None
