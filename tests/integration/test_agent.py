"""Integration tests for the full agent flow."""

from __future__ import annotations

import pytest

from src.agent.graph import KnowledgeAssistant, run_agent, build_simple_graph


class TestAgentFlow:
    """Tests for the full agent flow."""

    def test_run_agent_simple(self):
        """Test running the agent with a simple query."""
        result = run_agent("什么是机器学习?")

        assert "response" in result
        assert "query" in result
        assert result["query"] == "什么是机器学习?"
        assert len(result["response"]) > 0

    def test_knowledge_assistant(self):
        """Test the KnowledgeAssistant class."""
        assistant = KnowledgeAssistant(
            enable_decomposition=False,
            enable_rewrite=False,
        )

        result = assistant.ask("测试问题")

        assert "response" in result
        assert "citations" in result
        assert "decision_path" in result

    def test_build_simple_graph(self):
        """Test building the simple graph."""
        graph = build_simple_graph()

        assert graph is not None

    def test_agent_with_config(self):
        """Test agent with custom configuration."""
        result = run_agent(
            "测试问题",
            config={
                "enable_decomposition": False,
                "enable_rewrite": False,
            },
        )

        assert "response" in result


class TestAgentOutput:
    """Tests for agent output format."""

    def test_output_has_required_fields(self):
        """Test that output has all required fields."""
        result = run_agent("测试问题")

        required_fields = ["query", "response", "citations", "decision_path"]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_output_trace_id(self):
        """Test that output has a trace ID."""
        result = run_agent("测试问题")

        assert result.get("trace_id") is not None
