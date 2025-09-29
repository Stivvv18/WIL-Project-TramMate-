#!/usr/bin/env python3
import os
from typing import List
from operator import itemgetter

from pathlib import Path
import streamlit as st

# LangChain pieces
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.schema import Document
from langchain_ollama import ChatOllama

# Our helpers
from scripts.retriever import get_retriever, preprocess_query
from scripts.faq_router import maybe_answer_faq

APP_TITLE = "TramMate (offline)"

SYSTEM = (
    "You are TramMate, a Melbourne tram helper focused on the CBD. "
    "Use ONLY the provided context. Do not claim live times or disruptions. "
    "If a question needs live data, say you are offline-only. "
    "Always cite source filenames in [brackets]."
)

# -------------------- PAGE & STYLE --------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚋",
    layout="wide",
    initial_sidebar_state="collapsed",  # hide sidebar
)

st.markdown(
    """
    <style>
      /* Page background + typography */
      .stApp { background: linear-gradient(180deg,#0b1220 0%, #0a0f1a 100%); }
      h1, h2, h3, h4, h5, h6 { letter-spacing:.2px; }
      /* Hero */
      .hero { text-align:center; padding: 0.5rem 0 0.25rem; }
      .hero h1 { font-size: 3rem; margin: 0; }
      .hero p { color:#a3b1c6; margin: .25rem 0 0; }
      /* Input bar */
      .ask-wrap { margin: 1rem auto; max-width: 1100px; }
      .ask-row { display:flex; gap:.6rem; align-items:center; }
      .ask-row .stTextInput>div>div>input { height:3rem; font-size:1.05rem; }
      .ask-row .stButton>button { height:3rem; border-radius:12px; font-weight:600; padding:0 1.2rem; }
      /* Answer card */
      .card { border:1px solid #1d2639; border-radius:16px; padding:1.25rem 1.25rem; background:#0e1628; }
      .chips { display:flex; gap:.4rem; flex-wrap:wrap; margin-top:.5rem; }
      .chip { background:#172036; color:#cbd5e1; padding:.15rem .6rem; border-radius:999px; font-size:.78rem; }
      /* Top bar settings */
      .topbar { display:flex; justify-content:space-between; align-items:center; gap:1rem; }
      .tiny { color:#94a3b8; font-size:.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- TOP BAR --------------------
colA, colB = st.columns([7, 5])
with colA:
    st.markdown(
        '<div class="hero">'
        '<h1>🚋 TramMate — offline RAG</h1>'
        '<p class="tiny">Static knowledge base (FAISS + LangChain + Ollama). No live data.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

with colB:
    with st.expander("⚙️ Settings", expanded=False):
        model_name = st.text_input(
            "Ollama model",
            value=os.environ.get("TRAMMATE_MODEL", "mistral"),
            help="Run `ollama pull mistral` or switch to another local model you've pulled.",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        top_k = st.slider("Top-K documents", 3, 12, 6)
        mmr_lambda = st.slider("MMR diversity (λ)", 0.0, 1.0, 0.5, 0.05)
        show_chunks = st.checkbox("Show retrieved chunks", value=False)
        require_ctx = st.checkbox("Only answer if context found", value=False)
        st.caption("Tip: Rebuild index after KB changes → `scripts/make_chunks.py` then `scripts/build_faiss.py`.")

# -------------------- LLM & Chain --------------------
@st.cache_resource(show_spinner=False)
def get_llm(model: str, temperature: float):
    base_url = os.environ.get("TRAMMATE_OLLAMA_URL", "http://127.0.0.1:11434")
    return ChatOllama(model=model, temperature=temperature, num_ctx=4096, base_url=base_url)

@st.cache_resource(show_spinner=False)
def get_chain(model: str, temperature: float, k: int, lambda_mult: float):
    retriever = get_retriever(k=k, lambda_mult=lambda_mult)

    def format_docs(docs: List[Document]) -> str:
        if not docs:
            return "[no context retrieved]"
        parts = []
        for d in docs:
            src = (d.metadata or {}).get("source", "unknown")
            parts.append(f"{d.page_content}\n[{src}]")
        return "\n\n".join(parts)

    def build_inputs(q: str):
        qn = preprocess_query(q)
        docs = retriever.invoke(qn) if qn else []
        return {"question": qn, "docs": docs}

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer concisely with citations.")
    ])
    llm = get_llm(model, temperature)

    chain = (
        RunnableLambda(build_inputs)
        | {
            "question": itemgetter("question"),
            "context":  RunnableLambda(lambda x: format_docs(x["docs"])),
        }
        | prompt
        | llm
    )
    return chain

# -------------------- ASK FORM --------------------
st.markdown('<div class="ask-wrap">', unsafe_allow_html=True)
with st.form("ask_form", clear_on_submit=False):
    st.markdown('<div class="ask-row">', unsafe_allow_html=True)
    query = st.text_input("", placeholder="Ask about trams in the Melbourne CBD…", label_visibility="collapsed")
    submitted = st.form_submit_button("Ask")
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

clear = st.button("Clear")
if clear:
    st.session_state.pop("last_docs", None)
    st.rerun()

# -------------------- ANSWER FLOW --------------------
if submitted:
    q_pre = preprocess_query(query)
    if not q_pre.strip():
        st.warning("Please type a question 🙂")
        st.stop()

    # FAQ fast-path (deterministic)
    faq_ans = maybe_answer_faq(q_pre, threshold=90)
    if faq_ans:
        st.success("Answer from TramMate FAQ")
        with st.container():
            st.markdown(f'<div class="card">{faq_ans}<div class="chips"><span class="chip">faq.json</span></div></div>', unsafe_allow_html=True)
        st.session_state["last_docs"] = []
        st.stop()

    try:
        chain = get_chain(model_name, temperature, top_k, mmr_lambda)
    except Exception as e:
        st.error("Couldn't initialize retriever/LLM. Did you build the FAISS index and start Ollama?")
        st.exception(e)
        st.stop()

    with st.spinner("Retrieving & generating…"):
        retriever = get_retriever(k=top_k, lambda_mult=mmr_lambda)
        docs = retriever.invoke(q_pre)
        st.session_state["last_docs"] = docs

        if require_ctx and not docs:
            st.warning("I couldn’t find anything relevant in the knowledge base for that question.")
            st.stop()

        # Stream the answer
        ph = st.empty()
        answer_chunks = []
        try:
            for chunk in chain.stream(query):
                text = getattr(chunk, "content", str(chunk))
                answer_chunks.append(text)
                ph.markdown(f'<div class="card">{"".join(answer_chunks)}</div>', unsafe_allow_html=True)
        except Exception:
            st.error("Streaming failed — trying a single call…")
            try:
                resp = chain.invoke(query)
                ph.markdown(f'<div class="card">{getattr(resp, "content", str(resp))}</div>', unsafe_allow_html=True)
            except Exception as e2:
                st.exception(e2)

# -------------------- OPTIONAL: Sources panel --------------------
if 'last_docs' in st.session_state and st.session_state['last_docs'] and show_chunks:
    st.subheader("Retrieved context")
    for i, d in enumerate(st.session_state["last_docs"], 1):
        with st.expander(f"#{i} — {(d.metadata or {}).get('source', 'unknown')}"):
            st.write(d.page_content)
            st.code(d.metadata, language="json")

st.caption("TramMate is offline-only. For live departures/disruptions, integrate PTV v3 or GTFS-RT later.")
