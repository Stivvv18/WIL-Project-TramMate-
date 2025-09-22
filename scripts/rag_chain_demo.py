#!/usr/bin/env python3
from typing import List
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from retriever import get_retriever, preprocess_query

SYSTEM = (
    "You are TramMate, a Melbourne tram helper. Use ONLY the provided context. "
    "Static info only; do not invent live times. Cite the source filenames in brackets."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Question: {question}\nContext:\n{context}\nAnswer concisely with citations.")
])

# Offline stub LLM: just echoes context. Replace with your ChatOpenAI / Ollama later.
class EchoLLM:
    def invoke(self, inputs):
        q = inputs["question"]
        ctx = inputs["context"]
        return f"[demo mode] Would answer: '{q}' using context below--\n{ctx[:800]}---"

llm = EchoLLM()
retriever = get_retriever(k=6)

def format_docs(docs: List[Document]) -> str:
    out = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        out.append(f"{d.page_content}\n[{src}]")
    return "\n".join(out)

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
    | (lambda x: llm.invoke(x))
)

if __name__ == "__main__":
    print(chain.invoke("Do I need to tap on in the Free Tram Zone?"))

