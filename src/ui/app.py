"""Main Streamlit application for Agentic RAG Assistant."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import load_settings
from src.agent.graph import KnowledgeAssistant


def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "assistant" not in st.session_state:
        st.session_state.assistant = None
    if "settings" not in st.session_state:
        st.session_state.settings = None


def get_assistant() -> KnowledgeAssistant:
    """Get or create the knowledge assistant."""
    if st.session_state.assistant is None:
        settings = load_settings()
        st.session_state.settings = settings
        st.session_state.assistant = KnowledgeAssistant(
            enable_decomposition=settings.agent.enable_sub_query,
            enable_rewrite=settings.agent.enable_query_rewrite,
            max_rewrite_attempts=settings.agent.max_rewrite_attempts,
        )
    return st.session_state.assistant


def display_message(message: Dict[str, str]):
    """Display a chat message."""
    role = message.get("role", "user")
    content = message.get("content", "")

    with st.chat_message(role):
        st.markdown(content)


def display_citations(citations: List[Dict]):
    """Display citations."""
    if not citations:
        return

    with st.expander("参考文献", expanded=False):
        for i, citation in enumerate(citations, 1):
            source = citation.get("source_path", "未知来源")
            page = citation.get("page_num", "")
            snippet = citation.get("text_snippet", "")

            st.markdown(f"**[{i}] {source}**")
            if page:
                st.caption(f"第 {page} 页")
            if snippet:
                st.text(snippet[:200] + "..." if len(snippet) > 200 else snippet)
            st.divider()


def display_decision_path(decision_path: List[str]):
    """Display the agent's decision path."""
    if not decision_path:
        return

    with st.expander("决策路径", expanded=False):
        for decision in decision_path:
            st.markdown(f"- {decision}")


def display_sub_queries(sub_queries: List[str]):
    """Display decomposed sub-queries."""
    if not sub_queries:
        return

    with st.expander("子问题", expanded=False):
        for i, query in enumerate(sub_queries, 1):
            st.markdown(f"{i}. {query}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="企业知识助手",
        page_icon="🤖",
        layout="wide",
    )

    init_session_state()

    st.title("🤖 企业知识助手")
    st.caption("基于 Agentic RAG 的智能知识问答系统")

    with st.sidebar:
        st.header("设置")

        enable_decomposition = st.checkbox(
            "启用问题拆分", value=True, help="将复杂问题拆分为子问题分别检索"
        )

        enable_rewrite = st.checkbox("启用查询改写", value=True, help="当检索结果不充分时改写查询")

        max_rewrite = st.slider(
            "最大改写次数", min_value=0, max_value=5, value=2, help="查询改写的最大尝试次数"
        )

        st.divider()

        if st.button("清空对话", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        display_message(message)

    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    assistant = get_assistant()
                    assistant.enable_decomposition = enable_decomposition
                    assistant.enable_rewrite = enable_rewrite
                    assistant.max_rewrite_attempts = max_rewrite

                    result = assistant.ask(prompt)

                    response = result.get("response", "抱歉，我无法回答这个问题。")
                    st.markdown(response)

                    citations = result.get("citations", [])
                    display_citations(citations)

                    if st.session_state.settings and st.session_state.settings.ui.show_trace:
                        sub_queries = result.get("sub_queries", [])
                        display_sub_queries(sub_queries)

                        decision_path = result.get("decision_path", [])
                        display_decision_path(decision_path)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "citations": citations,
                            "sub_queries": result.get("sub_queries", []),
                            "decision_path": result.get("decision_path", []),
                        }
                    )

                except Exception as e:
                    st.error(f"发生错误: {e}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"抱歉，处理您的请求时发生错误: {e}",
                        }
                    )


if __name__ == "__main__":
    main()
