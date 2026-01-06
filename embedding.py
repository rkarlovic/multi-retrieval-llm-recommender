"""
Embedding-based retrieval using pre-built FAISS vector stores.

For creating new vector stores, use create_vectorstores.py instead.
"""
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class EmbeddingRetriever:
    """
    Vector embedding-based retrieval system using FAISS.
    
    Loads a pre-built FAISS vector store and performs semantic search.
    """
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 vectorstore_path=None, device='cpu'):
        """
        Initialize the embedding retriever.
        
        Args:
            model_name: HuggingFace model name for embeddings
            vectorstore_path: Path to saved FAISS vector store (if None, auto-detects)
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        
        # Initialize embeddings
        print(f"Loading embedding model: {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Auto-detect vector store path if not provided
        if vectorstore_path is None:
            model_short_name = self._get_model_short_name(model_name)
            vectorstore_path = os.path.join('vector_stores', f'vectorstore_{model_short_name}')
        
        self.vectorstore_path = vectorstore_path
        
        # Load vector store
        print(f"Loading vector store from: {vectorstore_path}...")
        self.vectorstore = FAISS.load_local(
            vectorstore_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✓ Vector store loaded successfully")
    
    def search(self, query: str, top_k: int = 5) -> list:
        """
        Search for relevant chunks using semantic embeddings.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
        
        Returns:
            List of dicts with 'content', 'score', 'rank', and 'metadata'
        """
        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        formatted_results = []
        for rank, (doc, score) in enumerate(results_with_scores, 1):
            formatted_results.append({
                "rank": rank,
                "score": float(score),  # Lower is better for FAISS L2 distance
                "content": doc.page_content[:200] + "...",
                "full_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return formatted_results
    
    @staticmethod
    def _get_model_short_name(model_name):
        """Convert model name to short filename-friendly format."""
        short_name = model_name.split('/')[-1]
        return short_name.replace('-', '_').lower()


if __name__ == "__main__":
    # Example usage
    retriever = EmbeddingRetriever()
    results = retriever.search("luxury hotel with spa", top_k=3)
    
    for result in results:
        print(f"[{result['rank']}] Score: {result['score']:.4f}")
        print(f"Content: {result['content']}\n")
