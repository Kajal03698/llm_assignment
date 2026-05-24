"""Streamlit UI for financial RAG Q&A."""

import streamlit as st

from rag import ask, load_index

st.set_page_config(page_title="Financial RAG", layout="wide")
st.title("Financial RAG Assistant")

DEMO = [
    "What was the reported revenue trend in the latest two periods?",
    "Which factors were cited for profit increase/decrease?",
    "What risks were highlighted in management discussion?",
    "Compare operating margin across two selected years.",
    "What assumptions are stated for forward-looking guidance?",
    "Which segment contributed most to growth and why?",
]

with st.sidebar:
    st.subheader("Demo questions")
    for q in DEMO:
        if st.button(q, use_container_width=True):
            st.session_state["q"] = q

try:
    store = load_index()
except FileNotFoundError:
    st.error("Index not found")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.info("Add Azure keys")
    st.stop()

question = st.text_area("Your question", value=st.session_state.get("q", ""), height=90)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Thinking..."):
        result = ask(store, question.strip())

    if result.status != "ok":
        st.warning(result.text)
    else:
        st.markdown("### Answer")
        st.markdown(result.text)

    if len(result.rewritten_queries) > 1:
        with st.expander("Search queries used"):
            for q in result.rewritten_queries:
                st.write("-", q)

    if result.citations:
        st.markdown("### Sources")
        for c in result.citations:
            with st.expander(f"[{c['index']}] {c['company']} — {c['section']}"):
                st.text(c["excerpt"][:1200])
                st.caption(c["file"])
