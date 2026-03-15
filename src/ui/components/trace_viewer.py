"""Trace viewer component for Streamlit UI."""

from __future__ import annotations

from typing import Dict, List

import streamlit as st


def render_decision_path(decision_path: List[str]):
    """Render the agent's decision path.

    Args:
        decision_path: List of decision strings.
    """
    if not decision_path:
        return

    st.subheader("决策路径")

    for i, decision in enumerate(decision_path):
        col1, col2 = st.columns([1, 9])

        with col1:
            st.markdown(f"**{i + 1}**")

        with col2:
            st.markdown(decision)

        if i < len(decision_path) - 1:
            st.markdown("↓")


def render_trace_timeline(trace_data: Dict):
    """Render trace as a timeline.

    Args:
        trace_data: Trace data dictionary.
    """
    st.subheader("执行时间线")

    events = trace_data.get("events", [])

    if not events:
        st.info("暂无追踪数据")
        return

    for event in events:
        name = event.get("name", "Unknown")
        duration = event.get("duration_ms", 0)
        status = event.get("status", "success")

        color = "green" if status == "success" else "red"

        col1, col2, col3 = st.columns([5, 3, 2])

        with col1:
            st.markdown(f":{color}[{name}]")

        with col2:
            st.caption(f"{duration:.1f}ms")

        with col3:
            if status == "success":
                st.markdown("✅")
            else:
                st.markdown("❌")


def render_metrics_panel(metrics: Dict):
    """Render metrics panel.

    Args:
        metrics: Metrics dictionary.
    """
    st.subheader("性能指标")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("总耗时", f"{metrics.get('total_latency_ms', 0):.0f}ms")

    with col2:
        st.metric("检索次数", metrics.get("retrieval_count", 0))

    with col3:
        st.metric("引用数量", metrics.get("citation_count", 0))


def render_sub_queries_panel(sub_queries: List[str]):
    """Render sub-queries panel.

    Args:
        sub_queries: List of sub-queries.
    """
    if not sub_queries:
        return

    st.subheader("子问题拆分")

    for i, query in enumerate(sub_queries, 1):
        st.markdown(f"{i}. {query}")
