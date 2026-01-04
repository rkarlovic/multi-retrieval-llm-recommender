import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:
    """
    TF-IDF based retrieval system for chunks.
    
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
            max_features=5000,          # Limit to top 5000 terms
            stop_words='english',        # Remove common words
            lowercase=True,
            min_df=1,                    # Include terms appearing in at least 1 doc
            max_df=0.95,                 # Exclude terms in >95% of docs
            ngram_range=(1, 2)           # Use unigrams and bigrams
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
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity with all chunks
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices sorted by similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if similarities[idx] > 0:  # Only include if similarity > 0
                results.append({
                    "rank": rank,
                    "chunk_id": idx,
                    "score": float(similarities[idx]),
                    "content": self.chunk_texts[idx][:200] + "...",  # Preview
                    "full_content": self.chunk_texts[idx],
                    "metadata": self.chunks[idx].get("metadata", {})
                })
        
        return results
    
    def search_verbose(self, query: str, top_k: int = 5):
        """
        Search with detailed console output showing results and scores.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
        """
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"{'='*80}")
        
        results = self.search(query, top_k)
        
        if not results:
            print("❌ No relevant results found.")
            return results
        
        for result in results:
            print(f"\n[Rank {result['rank']}] Score: {result['score']:.4f}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Preview: {result['content']}")
            if result['metadata']:
                print(f"Metadata: {result['metadata']}")
        
        print(f"\n{'='*80}\n")
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize retriever
    retriever = TFIDFRetriever(chunks_path="chunks.json")
    
    # Example queries
    queries = [
        "luxury hotels",
        "villa apartments Croatia",
        "family hotel seaside"
    ]
    
    for query in queries:
        results = retriever.search_verbose(query, top_k=3)
    
    # You can also use it programmatically without verbose output
    print("\n" + "="*80)
    print("PROGRAMMATIC USAGE (without verbose output):")
    print("="*80)
    results = retriever.search("best hotels", top_k=2)
    for r in results:
        print(f"Score: {r['score']:.4f} | Chunk {r['chunk_id']}")