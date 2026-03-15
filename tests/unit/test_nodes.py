"""Unit tests for agent nodes."""

from __future__ import annotations

import pytest

from src.agent.nodes.analyzer import analyze_query, analyze_node
from src.agent.nodes.decomposer import decompose_query_with_llm
from src.agent.nodes.evaluator import evaluate_retrieval
from src.agent.nodes.rewriter import rewrite_query
from src.agent.state import AgentState, create_initial_state


class TestAnalyzerNode:
    """Tests for the analyzer node."""

    def test_analyze_simple_query(self):
        """Test analyzing a simple query."""
        result = analyze_query("什么是机器学习?")

        assert "is_complex" in result
        assert "query_type" in result
        assert result["query_type"] in [
            "factual",
            "procedural",
            "analytical",
            "comparative",
            "general",
        ]

    def test_analyze_complex_query(self):
        """Test analyzing a complex query."""
        result = analyze_query("比较深度学习和机器学习的区别，以及它们各自的优缺点")

        assert result["is_complex"] is True
        assert result["query_type"] == "comparative"

    def test_analyze_procedural_query(self):
        """Test analyzing a procedural query."""
        result = analyze_query("如何训练一个机器学习模型？")

        assert result["query_type"] == "procedural"

    def test_analyze_node(self):
        """Test the analyze node."""
        state = create_initial_state("什么是RAG?")
        result = analyze_node(state)

        assert "decision_path" in result
        assert len(result["decision_path"]) > 0


class TestDecomposerNode:
    """Tests for the decomposer node."""

    def test_decompose_simple_query(self):
        """Test decomposing a simple query."""
        result = decompose_query_with_llm("什么是机器学习?")

        assert len(result) >= 1
        assert result[0] == "什么是机器学习?"

    def test_decompose_comparative_query(self):
        """Test decomposing a comparative query."""
        result = decompose_query_with_llm("比较Python和Java的区别")

        assert len(result) >= 1


class TestEvaluatorNode:
    """Tests for the evaluator node."""

    def test_evaluate_empty_chunks(self):
        """Test evaluating with no chunks."""
        result = evaluate_retrieval("测试问题", [])

        assert result["is_sufficient"] is False
        assert "未检索到" in result["reason"]
        assert result["score"] == 0.0

    def test_evaluate_low_score_chunks(self):
        """Test evaluating with low-score chunks."""
        from src.core.types import Chunk

        chunks = [
            Chunk(id="1", text="不相关内容", score=0.1, metadata={}),
            Chunk(id="2", text="另一个不相关内容", score=0.15, metadata={}),
            Chunk(id="3", text="第三个内容", score=0.2, metadata={}),
        ]

        result = evaluate_retrieval("测试问题", chunks, threshold=0.5)

        assert result["is_sufficient"] is False
        assert result["score"] < 0.5

    def test_evaluate_high_score_sufficient_chunks(self):
        """Test evaluating with high-score and sufficient chunks."""
        from src.core.types import Chunk

        chunks = [
            Chunk(id="1", text="相关内容1", score=0.85, metadata={}),
            Chunk(id="2", text="相关内容2", score=0.90, metadata={}),
            Chunk(id="3", text="相关内容3", score=0.88, metadata={}),
            Chunk(id="4", text="相关内容4", score=0.82, metadata={}),
        ]

        result = evaluate_retrieval("测试问题", chunks, threshold=0.5)

        assert result["is_sufficient"] is True
        assert result["score"] >= 0.5

    def test_evaluate_insufficient_chunk_count(self):
        """Test evaluating with high-score but insufficient chunk count."""
        from src.core.types import Chunk

        chunks = [
            Chunk(id="1", text="相关内容", score=0.90, metadata={}),
            Chunk(id="2", text="另一个内容", score=0.85, metadata={}),
        ]

        result = evaluate_retrieval("测试问题", chunks, threshold=0.5)

        assert result["is_sufficient"] is False
        assert "数量不足" in result["reason"]


class TestRewriterNode:
    """Tests for the rewriter node."""

    def test_rewrite_query(self):
        """Test rewriting a query."""
        result = rewrite_query("什么是机器学习", "检索结果相关性较低")

        assert result != "什么是机器学习"
        assert len(result) > 0


class TestAgentState:
    """Tests for agent state."""

    def test_create_initial_state(self):
        """Test creating initial state."""
        state = create_initial_state("测试问题")

        assert state.original_query == "测试问题"
        assert state.trace_id is not None
        assert state.decision_path == ["start"]

    def test_state_add_decision(self):
        """Test adding decision to state."""
        state = create_initial_state("测试问题")
        state.add_decision("test_decision")

        assert "test_decision" in state.decision_path
