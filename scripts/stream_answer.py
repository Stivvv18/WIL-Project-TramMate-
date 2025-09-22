#!/usr/bin/env python3
from langchain_ollama import ChatOllama
from retriever import get_retriever, preprocess_query
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYSTEM = "You are TramMate. Use only provided context. Cite sources like [file]."

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Question: {q}\nContext:\n{ctx}")
])

llm = ChatOllama(model="mistral", temperature=0.2)
retriever = get_retriever(k=6)

fmt = lambda docs: "\n".join(
    f"{d.page_content}\n[{d.metadata.get('source', '')}]" for d in docs
)

chain = (
    {
        "q": lambda q: preprocess_query(q),
        "ctx": lambda q: fmt(retriever.get_relevant_documents(q)),
    }
    | PROMPT
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Which routes pass Bourke St Mall?"
    for chunk in chain.stream(q):
        print(chunk, end="", flush=True)
    print()

