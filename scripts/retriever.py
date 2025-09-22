#!/usr/bin/env python3
from encodings.aliases import aliases
from pathlib import Path
import json
from typing import Dict, Any, List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parents[1]
VSDIR = ROOT/"data/kb/vectorstore/faiss_index"
ALIASES = ROOT/"data/curated/aliases.json"

emb = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2",
encode_kwargs={"normalize_embeddings": True},
)

# --- optional: alias expansion for user slang/nicknames ---
def preprocess_query(q: str) -> str:
    try:
        aliases = json.loads(ALIASES.read_text(encoding='utf-8')) if ALIASES.exists() else {}
    except Exception:
        aliases = {}
    lower = q.lower()
    for canon, alist in aliases.items():
        for a in alist:
            if a.lower() in lower:
                lower = lower.replace(a.lower(), canon.lower())
                return lower

# --- load vector store and build a retriever ---
def get_vectorstore() -> FAISS:
    return FAISS.load_local(str(VSDIR), emb, allow_dangerous_deserialization=True)

def get_retriever(k: int = 6):
    vs = get_vectorstore()
    # MMR is useful to diversify results for policy + route + landmark blends
    return vs.as_retriever(search_type="mmr", search_kwargs={
        "k": k,
        "fetch_k": 35,
        "lambda_mult": 0.5,
    })

# --- metadata filter (manual, since FAISS has no native filters) ---
def filtered_similar_docs(query: str, where: Dict[str, Any] | None = None, k: int = 6):
    """Apply a simple post-filter over metadata keys.
    Example: where={"source": lambda s: "policy" in s or
    s.endswith("policy_summaries.md")}
    """
    vs = get_vectorstore()
    docs = vs.similarity_search(preprocess_query(query), k=25) # grab more, then filter
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
    for i, d in enumerate(r.get_relevant_documents(preprocess_query(q)), 1):
        print("#", i, d.metadata)
        print(d.page_content[:400])
