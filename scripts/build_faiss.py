#!/usr/bin/env python3
import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

KB_DIR = Path("data/kb")
CHUNKS = KB_DIR / "chunks.jsonl"
OUTDIR = KB_DIR / "vectorstore"  # will contain FAISS files
OUTDIR.mkdir(parents=True, exist_ok=True)

texts, metas = [], []

with CHUNKS.open('r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"])
        metas.append(obj.get("meta", {}))

emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vs = FAISS.from_texts(texts=texts, embedding=emb, metadatas=metas)
vs.save_local(str(OUTDIR / "faiss_index"))

print("Saved FAISS index →", OUTDIR / "faiss_index")

