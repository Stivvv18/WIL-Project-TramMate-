#!/usr/bin/env python3
from pathlib import Path
import json
from typing import Optional, List, Tuple

# requires: pip install rapidfuzz
from rapidfuzz import process, fuzz

ROOT = Path(__file__).resolve().parents[1]
FAQ_PATH = ROOT / "data/curated/faq.json"

def _load_faq_items() -> List[Tuple[str, str]]:
    """Return list of (query_variant_lower, canonical_answer) pairs."""
    if not FAQ_PATH.exists():
        return []
    data = json.loads(FAQ_PATH.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "faqs" in data:
        data = data["faqs"]

    items: List[Tuple[str, str]] = []
    for row in data or []:
        q = (row.get("q") or "").strip()
        a = (row.get("a") or "").strip()
        if not a:
            continue
        aliases = [q] + list(row.get("aliases") or [])
        for alias in aliases:
            alias = (alias or "").strip()
            if alias:
                items.append((alias.lower(), a))
    return items

_ITEMS = _load_faq_items()
_QUERIES = [t[0] for t in _ITEMS]
_ANSWERS = [t[1] for t in _ITEMS]

def maybe_answer_faq(query: Optional[str], threshold: int = 90) -> Optional[str]:
    """If query matches an FAQ (exact or fuzzy), return the canonical answer; else None."""
    if not query:
        return None
    q = query.strip().lower()
    if not q or not _ITEMS:
        return None

    # exact match first
    for cand, ans in _ITEMS:
        if q == cand:
            return ans

    # fuzzy match
    match = process.extractOne(q, _QUERIES, scorer=fuzz.token_sort_ratio)
    if match:
        _, score, idx = match
        if score >= threshold:
            return _ANSWERS[idx]
    return None

if __name__ == "__main__":
    print("Loaded FAQ items:", len(_ITEMS))
    print("Sample:", maybe_answer_faq("Is the City Circle Tram free?"))
