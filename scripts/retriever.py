#!/usr/bin/env python3
from pathlib import Path
import json
from typing import Dict, Any, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

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

# --- load vector store and build a retriever ---
def get_vectorstore() -> FAISS:
    # allow_dangerous_deserialization required for FAISS docstore pickle
    return FAISS.load_local(str(VSDIR), emb, allow_dangerous_deserialization=True)

def get_retriever(k: int = 6):
    vs = get_vectorstore()
    # MMR => diverse results (policy + routes + landmarks)
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 35, "lambda_mult": 0.5},
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
