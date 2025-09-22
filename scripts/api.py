# scripts/api.py
from retriever import get_retriever, preprocess_query
def retrieve_for_ui(question: str, k: int = 6):


retriever = get_retriever(k)
q = preprocess_query(question)
docs = retriever.get_relevant_documents(q)
return [{"text": d.page_content, "source": d.metadata.get("source", ""),
         "meta": d.metadata} for d in docs]

