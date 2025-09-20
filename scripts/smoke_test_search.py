#!/usr/bin/env python3
import sys, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

STORE_DIR = Path("data/kb/vectorstore")

TEXTS = (STORE_DIR / "texts.jsonl").read_text(encoding='utf-8').splitlines()
METAS = json.loads((STORE_DIR / "metas.json").read_text(encoding='utf-8'))
X = np.load(STORE_DIR / "embeddings.npz")["X"]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Is the City Circle Tram free?"
qv = model.encode([q], normalize_embeddings=True)[0].astype(np.float32)

# cosine via dot product (vectors are normalized)
sims = X @ qv
idxs = np.argsort(-sims)[:5]

for rank, i in enumerate(idxs, 1):
    print(f"\n#{rank} score={sims[i]:.3f}")
    print("text:", json.loads(TEXTS[i])["text"][:500])
    print("meta:", METAS[i])
