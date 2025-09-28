import streamlit as st
from typing import List
from rag_chain import build_chain


st.set_page_config(page_title="SafeDrop AI", page_icon="📦", layout="centered")
st.title("📦 SafeDrop AI — Policy-Aware Delivery Support")

# Build the chain once and keep it in session
if "chain" not in st.session_state:
    try:
        st.session_state.chain = build_chain()
    except Exception as e:
        st.error(
            "Could not start the RAG backend.\n\n"
            f"**Reason:** {e}\n\n"
            "Fixes:\n"
            "1) Run `python app\\ingest.py` once to create the Chroma index.\n"
            "2) Install & run **Ollama**, then pull a model: `ollama pull llama3`.\n"
            "3) Ensure `prompts\\policy_aware.txt` exists."
        )
        st.stop()

if "history" not in st.session_state:
    st.session_state.history: List[tuple] = []

with st.sidebar:
    st.markdown("### How to use")
    st.write(
        "- Ask delivery policy questions (Safe Drop / ATL / collection).\n"
        "- Answers are restricted to policy context.\n"
        "- If policy is unclear, the bot will **escalate**."
    )
    st.divider()
    answered = sum(1 for _, a, _ in st.session_state.history if not a.lower().startswith("escalate:"))
    escalated = sum(1 for _, a, _ in st.session_state.history if a.lower().startswith("escalate:"))
    st.metric("Answered", answered)
    st.metric("Escalated", escalated)

# Chat input
user_q = st.chat_input("Type your question about delivery, Safe Drop, ATL, etc.")
if user_q:
    out = st.session_state.chain(user_q)
    answer = out["result"]
    sources = out.get("source_documents", [])
    st.session_state.history.append((user_q, answer, sources))

# Render chat history (newest last)
for q, a, srcs in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        with st.expander("Show sources"):
            if not srcs:
                st.caption("No sources returned.")
            else:
                for i, doc in enumerate(srcs, start=1):
                    meta = doc.metadata
                    src = meta.get("source") or meta.get("file_path") or "unknown"
                    snippet = doc.page_content[:350].strip().replace("\n", " ")
                    st.markdown(f"**{i}.** `{src}` — {snippet} …")
