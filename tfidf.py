"""
TF-IDF based retrieval system for chunks.

Uses sparse TF-IDF vectors for keyword-based matching.
"""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:
    """
    TF-IDF based retrieval system.
    
    Uses sparse TF-IDF vectors to find relevant chunks matching a query.
    No vector store needed - works directly with JSON chunks.
    """
    
    def __init__(self, chunks_path: str = "chunks.json"):
        """
        Initialize the TF-IDF retriever.
        
        Args:
            chunks_path: Path to the chunks.json file
        """
        self.chunks_path = chunks_path
        self.chunks = []
        self.chunk_texts = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
        self._load_chunks()
        self._fit_tfidf()
    
    def _load_chunks(self):
        """Load chunks from JSON file."""
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        self.chunk_texts = [chunk.get("content", "") for chunk in self.chunks]
        print(f"✓ Loaded {len(self.chunks)} chunks from {self.chunks_path}")
    
    def _fit_tfidf(self):
        """Fit TF-IDF vectorizer on all chunks."""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            lowercase=True,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)
        print(f"✓ TF-IDF model fitted | Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def search(self, query: str, top_k: int = 5) -> list:
        """
        Search for relevant chunks using TF-IDF.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
        
        Returns:
            List of dicts with 'content', 'score', 'chunk_id', and 'metadata'
        """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if similarities[idx] > 0:
                results.append({
                    "rank": rank,
                    "chunk_id": idx,
                    "score": float(similarities[idx]),
                    "content": self.chunk_texts[idx][:200] + "...",
                    "full_content": self.chunk_texts[idx],
                    "metadata": self.chunks[idx].get("metadata", {})
                })
        
        return results


if __name__ == "__main__":
    # Example usage
    retriever = TFIDFRetriever(chunks_path="chunks.json")
    results = retriever.search("luxury hotels", top_k=3)
    
    for result in results:
        print(f"[{result['rank']}] Score: {result['score']:.4f} | Chunk {result['chunk_id']}")
        print(f"Content: {result['content']}\n")
