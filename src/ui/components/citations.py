"""Citations component for Streamlit UI."""

from __future__ import annotations

from typing import Dict, List

import streamlit as st


def render_citation_card(citation: Dict, index: int):
    """Render a single citation card.

    Args:
        citation: Citation dictionary.
        index: Citation index.
    """
    source = citation.get("source_path", "未知来源")
    page = citation.get("page_num")
    snippet = citation.get("text_snippet", "")
    score = citation.get("score", 0)

    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(f"**[{index}] {source}**")
            if page:
                st.caption(f"第 {page} 页")

        with col2:
            st.metric("相关性", f"{score:.2f}")

        if snippet:
            st.text(snippet[:150] + "..." if len(snippet) > 150 else snippet)

        st.divider()


def render_citations_panel(citations: List[Dict]):
    """Render the full citations panel.

    Args:
        citations: List of citation dictionaries.
    """
    if not citations:
        st.info("暂无引用文献")
        return

    st.subheader("参考文献")

    for i, citation in enumerate(citations, 1):
        render_citation_card(citation, i)


def render_inline_citation(citation: Dict, index: int) -> str:
    """Generate inline citation text.

    Args:
        citation: Citation dictionary.
        index: Citation index.

    Returns:
        Inline citation text.
    """
    source = citation.get("source_path", "未知来源")
    page = citation.get("page_num")

    if page:
        return f"[{index}] {source}, 第{page}页"
    return f"[{index}] {source}"
