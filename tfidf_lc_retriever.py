"""
LangChain-compatible wrapper around the existing TFIDFRetriever.

Converts TF-IDF search results into LangChain Document objects so they can be
used with MergerRetriever and compression pipelines.
"""
from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from tfidf import TFIDFRetriever


class TFIDFLangChainRetriever(BaseRetriever):
    """
    Wraps TFIDFRetriever to return LangChain Documents.
    Adds metadata fields like 'score', 'chunk_id', and 'source' = 'tfidf'.
    """

    chunks_path: str = "chunks.json"
    top_k: int = 5
    tfidf: TFIDFRetriever | None = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, chunks_path: str = "chunks.json", top_k: int = 5, **kwargs):
        super().__init__(chunks_path=chunks_path, top_k=top_k, **kwargs)
        self.tfidf = TFIDFRetriever(chunks_path=chunks_path)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = self.tfidf.search(query, top_k=self.top_k)
        docs: List[Document] = []
        for r in results:
            docs.append(
                Document(
                    page_content=r.get("full_content") or r.get("content") or "",
                    metadata={
                        "score": r.get("score", 0.0),
                        "chunk_id": r.get("chunk_id"),
                        "retriever": "tfidf",
                        **(r.get("metadata", {}) or {})
                    },
                )
            )
        return docs


if __name__ == "__main__":
    retriever = TFIDFLangChainRetriever(chunks_path="chunks.json")
    docs = retriever.invoke("luxury hotels")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] score={d.metadata.get('score')} chunk_id={d.metadata.get('chunk_id')} source={d.metadata.get('source')}")
        print(d.page_content[:120] + "...\n")
