"""Chat component for Streamlit UI."""

from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st


def render_chat_message(role: str, content: str, citations: Optional[List[Dict]] = None):
    """Render a chat message with optional citations.

    Args:
        role: Message role (user/assistant).
        content: Message content.
        citations: Optional list of citations.
    """
    with st.chat_message(role):
        st.markdown(content)

        if citations and role == "assistant":
            render_citation_badges(citations)


def render_citation_badges(citations: List[Dict]):
    """Render citation badges.

    Args:
        citations: List of citation dictionaries.
    """
    if not citations:
        return

    cols = st.columns(min(len(citations), 5))

    for i, (col, citation) in enumerate(zip(cols, citations[:5])):
        with col:
            source = citation.get("source_path", "未知")
            source_name = source.split("/")[-1] if "/" in source else source
            st.caption(f"[{i + 1}] {source_name}")


def render_chat_history(messages: List[Dict]):
    """Render the full chat history.

    Args:
        messages: List of message dictionaries.
    """
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        citations = message.get("citations", [])
        render_chat_message(role, content, citations)


def render_input_area(placeholder: str = "请输入您的问题...") -> Optional[str]:
    """Render the chat input area.

    Args:
        placeholder: Input placeholder text.

    Returns:
        User input or None.
    """
    return st.chat_input(placeholder)
