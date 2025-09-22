#!/usr/bin/env python3
import csv
from email.mime import text
from importlib.resources import path
import json, re
from pathlib import Path
from pdfminer.high_level import extract_text
from tqdm import tqdm
import yaml

CFG = yaml.safe_load(Path("config/settings.yaml").read_text())
KB_DIR = Path("data/kb"); KB_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS = KB_DIR/"chunks.jsonl"

# ---------- helpers ----------
def clean_ws(s: str) -> str:
    s = re.sub(r"[\t\x0b\x0c\r]+", " ", s)
    s = re.sub(r"\u00A0", " ", s) # nbsp
    return re.sub(r"\s+", " ", s).strip()

def chunk_text(text: str, source: str, size=900, overlap=150):
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + size)
        piece = text[i:j]
        out.append({"text": piece.strip(), "meta": {"source": source}})
        if j == n: break
        i = max(i + size - overlap, 0)
    return out

def pdf_to_chunks(path: Path):
    try:
        txt = extract_text(str(path))
    except Exception as e:
        print(f"[warn] PDF extract failed for {path}: {e}")
        return []
    txt = clean_ws(txt)
    # split by double newline into paragraphs for nicer boundaries
    paras = [p.strip() for p in re.split(r"\n{2,}", txt) if len(p.strip()) > 60]
    joined = "\n\n".join(paras)
    return chunk_text(joined, str(path), CFG['chunk']['size_chars'], CFG['chunk']['overlap_chars'])

def md_to_chunks(path: Path):
    txt = clean_ws(path.read_text(encoding='utf-8'))
    return chunk_text(txt, str(path), CFG['chunk']['size_chars'], CFG['chunk']['overlap_chars'])

def json_faqs_to_chunks(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    items = data.get('faqs', data)
    out = []
    for i, it in enumerate(items):
        lower = {str(k).lower(): v for k,v in it.items()}
        q = lower.get('question') or lower.get('q') or ''
        a = lower.get('answer') or lower.get('a') or ''
        if q and a:
            text = f"Q: {q}\nA: {a}"
        else:
            text = json.dumps(it, ensure_ascii=False)
        out.append({"text": text.strip(), "meta": {"source": str(path), "type": "faq", "idx": i}})
    return out

def csv_summary_lines(path: Path, max_kv=8):
    import csv
    rows = []
    with path.open('r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for i, r in enumerate(rdr):
            kv = []
            for k, v in r.items():
                if v is None or str(v).strip() == '':
                    continue
                kv.append(f"{k}={v}")
            if kv:
                rows.append({"text": f"Tram data: {' | '.join(kv[:max_kv])}", "meta": {"source": str(path), "row": i}})
    return rows

# ---------- main ----------

all_chunks = []
# PDFs
for p in CFG['sources'].get('include_pdfs', []):
    path = Path(p)
    if path.exists():
        all_chunks += pdf_to_chunks(path)

# Markdown
for p in CFG['sources'].get('include_markdown', []):
    path = Path(p)
    if path.exists():
        all_chunks += md_to_chunks(path)

# JSON FAQs
for p in CFG['sources'].get('include_json_faqs', []):
    path = Path(p)
    if path.suffix.lower() == '.json' and path.exists():
        all_chunks += json_faqs_to_chunks(path)

# Curated CSVs → concise lines (nice to answer place/stop questions)
for name in [
    'data/curated/tram_routes.csv',
    'data/curated/tram_stops.csv',
    'data/curated/route_stops_cbd.csv',
    'data/curated/stops_in_ftz.csv'
]:
    p = Path(name)
    if p.exists():
        all_chunks += csv_summary_lines(p)

# write JSONL
with CHUNKS.open('w', encoding='utf-8') as f:
    for ch in all_chunks:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")
print(f"Wrote {len(all_chunks)} chunks to {CHUNKS}")
