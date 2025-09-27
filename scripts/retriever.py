#!/usr/bin/env python3
from pathlib import Path
import json
from typing import Dict, Any, List, Optional
from functools import lru_cache

from langchain_community.vectorstores import FAISS
try:
    # preferred modern package
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # graceful fallback if not installed
    from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parents[1]
VSDIR = ROOT / "data/kb/vectorstore/faiss_index"
ALIASES = ROOT / "data/curated/aliases.json"

# Single, shared embedder
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# --- load alias map safely (tolerate missing file / BOM) ---
def _load_aliases() -> dict:
    if not ALIASES.exists():
        return {}
    try:
        return json.loads(ALIASES.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}

# --- optional: alias expansion for user slang/nicknames ---
def preprocess_query(q: Optional[str]) -> str:
    """Always return a non-None string. Expand known aliases."""
    base = (q or "").strip()
    if not base:
        return ""
    alias_map = _load_aliases()
    text = base.lower()
    for canon, alist in alias_map.items():
        for a in (alist or []):
            if not a:
                continue
            a_l = a.lower()
            if a_l in text:
                text = text.replace(a_l, str(canon).lower())
    return text

@lru_cache(maxsize=1)
def _cached_vs() -> FAISS:
    if not VSDIR.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {VSDIR}. "
            "Build it first with: scripts/make_chunks.py then scripts/build_faiss.py"
        )
    # allow_dangerous_deserialization is required for FAISS docstore pickle
    return FAISS.load_local(str(VSDIR), emb, allow_dangerous_deserialization=True)

# --- load vector store and build a retriever ---
def get_vectorstore() -> FAISS:
    # allow_dangerous_deserialization required for FAISS docstore pickle
    return _cached_vs()

def get_retriever(k: int = 6, lambda_mult: float = 0.5):
    vs = get_vectorstore()
    # MMR => diverse results (policy + routes + landmarks)
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 35, "lambda_mult": float(lambda_mult)},
    )

# --- metadata filter (manual, since FAISS has no native filters) ---
def filtered_similar_docs(query: str, where: Optional[Dict[str, Any]] = None, k: int = 6):
    """Apply a simple post-filter over metadata keys.
    Example: where={"source": lambda s: s and ("policy" in s or s.endswith("policy_summaries.md"))}
    """
    vs = get_vectorstore()
    q_norm = preprocess_query(query)
    if not q_norm:
        return []
    # grab more then filter down
    docs = vs.similarity_search(q_norm, k=25)
    if where:
        kept: List = []
        for d in docs:
            ok = True
            for key, pred in where.items():
                val = (d.metadata or {}).get(key)
                ok = ok and bool(pred(val))
            if ok:
                kept.append(d)
        docs = kept
    return docs[:k]

if __name__ == "__main__":
    r = get_retriever()
    q = "Is City Circle free and do I need to tap on?"
    qn = preprocess_query(q)
    for i, d in enumerate(r.invoke(qn), 1):  # modern API
        print("#", i, d.metadata)
        print(d.page_content[:400])
