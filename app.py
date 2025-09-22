#!/usr/bin/env python3
import os
from pathlib import Path
import streamlit as st
from typing import List

# LangChain pieces
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from langchain_ollama import ChatOllama

# Our retriever helpers from scripts/
from scripts.retriever import get_retriever, preprocess_query


APP_TITLE = "TramMate (offline)"

SYSTEM = (
    "You are TramMate, a Melbourne tram helper focused on the CBD. "
    "Use ONLY the provided context. Do not claim live times or disruptions. "
    "If a question needs live data, say you are offline-only. "
    "Always cite source filenames in [brackets]."
)

# -------------------- UI SETUP --------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🚌", layout="wide")

st.title("🚋 TramMate — offline RAG")
st.caption("Static knowledge base (FAISS + LangChain + Ollama). No live data.")

# Sidebar controls
with st.sidebar:
    st.subheader("Model & Retrieval")
    model_name = st.text_input(
        "Ollama model",
        value=os.environ.get("TRAMMATE_MODEL", "mistral"),
        help="Run `ollama pull mistral` or switch to another local model you've pulled.",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_k = st.slider("Top-K documents", 3, 12, 6)
    mmr_lambda = st.slider("MMR diversity (λ)", 0.0, 1.0, 0.5, 0.05)
    show_chunks = st.checkbox("Show retrieved chunks", value=False)

    st.divider()
    st.write("**Tips**")
    st.markdown(
        "• Make sure FAISS index exists: `data/kb/vectorstore/faiss_index/`\n"
        "• Start Ollama first: `ollama serve` (usually auto) and `ollama pull mistral`.\n"
        "• If you update the KB, rebuild the index with `scripts/build_faiss.py`."
    )

# -------------------- LLM & Retriever --------------------
@st.cache_resource(show_spinner=False)
def get_llm(model: str, temperature: float):
    return ChatOllama(
        model=model,
        temperature=temperature,
        num_ctx=4096,
        base_url="http://127.0.0.1:11434",
    )


@st.cache_resource(show_spinner=False)
def get_chain(model: str, temperature: float, k: int):
    retriever = get_retriever(k=k)  # uses FAISS index on disk

    def format_docs(docs: List[Document]) -> str:
        parts = []
        for d in docs:
            src = (d.metadata or {}).get("source", "unknown")
            parts.append(f"{d.page_content}\n[{src}]")
        return "\n".join(parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        (
            "human",
            "Question: {question}\n"
            "Context:\n"
            "{context}\n"
            "Answer concisely with citations."
        ),
    ])

    llm = get_llm(model, temperature)

    chain = (
        {"question": RunnablePassthrough()}
        | {
            "question": lambda q: preprocess_query(q),
            "docs": lambda q: retriever.get_relevant_documents(q),
        }
        | {
            "question": RunnablePassthrough(),
            "docs": RunnablePassthrough(),
            "context": lambda x: format_docs(x["docs"]),
        }
        | prompt
        | llm
    )
    return chain


# -------------------- APP BODY --------------------
query = st.text_input(
    "Ask a question (e.g., 'Is the City Circle Tram free?')", value=""
)

col1, col2 = st.columns([1, 3])
with col1:
    ask = st.button("Ask")
with col2:
    clear = st.button("Clear")

if clear:
    st.session_state.pop("last_docs", None)
    st.experimental_rerun()

if ask and query.strip():
    try:
        chain = get_chain(model_name, temperature, top_k)
    except Exception as e:
        st.error(
            "Couldn't initialize retriever/LLM. Did you build the FAISS index and start Ollama?"
        )
        st.exception(e)
    else:
        with st.spinner("Retrieving & generating…"):
            # run once to also get docs we formatted inside the chain
            from scripts.retriever import get_retriever

            retriever = get_retriever(k=top_k)
            q_pre = preprocess_query(query)
            docs = retriever.get_relevant_documents(q_pre)
            st.session_state["last_docs"] = docs

            # Stream answer
            ph = st.empty()
            answer_chunks = []
            try:
                for chunk in chain.stream(query):
                    text = getattr(chunk, "content", str(chunk))
                    answer_chunks.append(text)
                    ph.markdown("".join(answer_chunks))
            except Exception as e:
                st.error("Streaming failed — trying a single call…")
                try:
                    resp = chain.invoke(query)
                    ph.markdown(getattr(resp, "content", str(resp)))
                except Exception as e2:
                    st.exception(e2)

# Retrieved chunks panel
if show_chunks and st.session_state.get("last_docs"):
    st.subheader("Retrieved context")
    for i, d in enumerate(st.session_state["last_docs"], 1):
        with st.expander(f"#{i} — {(d.metadata or {}).get('source', 'unknown')}"):
            st.write(d.page_content)
            st.code(d.metadata, language="json")

st.divider()
st.caption(
    "TramMate is offline-only. Data sources: your committed PDFs/markdown "
    "and curated GTFS schedule. For live departures or disruptions, integrate PTV v3 "
    "or GTFS-RT later."
)

