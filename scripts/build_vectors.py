#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import yaml
from sentence_transformers import SentenceTransformer

CFG = yaml.safe_load(Path("config/settings.yaml").read_text())
KB_DIR = Path("data/kb"); KB_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS = KB_DIR/"chunks.jsonl"
STORE_DIR = Path(CFG['store']['path']); STORE_DIR.mkdir(parents=True, exist_ok=True)

# load chunks
texts, metas = [], []
with CHUNKS.open('r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        t = (obj.get('text') or '').strip()
        if not t:
            continue
        texts.append(t)
        metas.append(obj.get('meta', {}))

print(f"Embedding {len(texts)} chunks …")
model = SentenceTransformer(CFG['embed']['model_name'])
X = model.encode(texts, normalize_embeddings=bool(CFG['embed'].get('normalize', True)), batch_size=64, show_progress_bar=True)
X = X.astype(np.float32)

# save texts + metas
(Path(STORE_DIR/"texts.jsonl")).write_text("\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in texts), encoding='utf-8')
(Path(STORE_DIR/"metas.json")).write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding='utf-8')

kind = CFG['store'].get('kind', 'npz')
if kind == 'faiss':
    try:
        import faiss
    except Exception as e:
        print(f"FAISS not available ({e}); falling back to NPZ")
        kind = 'npz'
        
if kind == 'faiss':
    d = X.shape[1]
    index = faiss.IndexFlatIP(d) # cosine if vectors are normalized
    index.add(X)
    faiss.write_index(index, str(STORE_DIR/"index.faiss"))
    print(f"FAISS index saved → {STORE_DIR/'index.faiss'}")
else:
    np.savez_compressed(STORE_DIR/"embeddings.npz", X=X)
    print(f"NPZ embeddings saved → {STORE_DIR/'embeddings.npz'}")
print("Done.")

