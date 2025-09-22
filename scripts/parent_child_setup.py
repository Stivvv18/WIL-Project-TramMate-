#!/usr/bin/env python3
from langchain.schema import Document
from pathlib import Path
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load our JSONL chunks as Documents
root = Path("data/kb/chunks.jsonl")
records = [json.loads(l) for l in root.read_text(encoding='utf-8').splitlines()]
from langchain.schema import Document
children = [Document(page_content=r['text'], metadata=r.get('meta', {})) for r in records]

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})
vectorstore = FAISS.from_documents(children, emb)

# Store full parents in memory using a synthetic parent_id from metadata (e.g.,source+para)
parent_store = InMemoryStore()
for d in children:
    pid= f"{d.metadata.get('source', '')}-{d.metadata.get('para', d.metadata.get('row', d.metadata.get('idx', '0')))}"
    parent_store.mset([(pid, Document(page_content=d.page_content, metadata=d.metadata))])
    
retriever = ParentDocumentRetriever(
    vectorstore = vectorstore,
    docstore = parent_store,
    child_splitter = None,
    search_kwargs = {"k": 6}
)
