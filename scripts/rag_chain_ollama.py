#!/usr/bin/env python3
from typing import List
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain.schema import Document

from retriever import get_retriever, preprocess_query

SYSTEM = (
    "You are TramMate, a Melbourne tram helper for the CBD. Use ONLY the context. "
    "Static info only (no live times). If unsure, say so. Cite source filenames in [brackets]."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    (
        "human",
        "Question: {question}\n"
        "Context:\n"
        "{context}\n"
        "Answer concisely with citations."
    ),
])

# --- choose your local model ---
# Make sure you've run: `ollama pull mistral` or `ollama pull llama2:13b`
llm = ChatOllama(
    model="mistral",  # or "llama2:13b"
    temperature=0.2,
    num_ctx=4096,  # keep modest for 7B models
    base_url="http://127.0.0.1:11434",  # default
)

retriever = get_retriever(k=6)


def format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        parts.append(f"{d.page_content}\n[{src}]")
    return "\n".join(parts)


chain = (
    {"question": RunnablePassthrough()}
    | {
        "question": lambda q: preprocess_query(q),
        "docs": lambda q: retriever.get_relevant_documents(q),
    }
    | {
        "question": RunnablePassthrough(),
        "context": lambda x: format_docs(x["docs"]),
    }
    | prompt
    | llm
)

if __name__ == "__main__":
    print(chain.invoke("Do I need to tap on in the Free Tram Zone?"))

