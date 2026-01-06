"""
Comparison script for TF-IDF vs Embedding-based retrieval.

This script demonstrates how to use both retrieval methods and compare their results.
"""

from tfidf import TFIDFRetriever
from embedding import EmbeddingRetriever


def compare_retrievers(query: str, top_k: int = 5):
    """
    Compare TF-IDF and embedding-based retrieval for a given query.
    
    Args:
        query: Search query string
        top_k: Number of results to retrieve from each method
    """
    print("\n" + "="*100)
    print(f"COMPARING RETRIEVAL METHODS")
    print(f"Query: '{query}' | Top-{top_k} results")
    print("="*100)
    
    # Initialize TF-IDF retriever
    print("\n[1] Initializing TF-IDF Retriever...")
    tfidf = TFIDFRetriever(chunks_path="chunks.json")
    
    # Initialize Embedding retriever (using MiniLM model by default)
    print("\n[2] Initializing Embedding Retriever...")
    embedding = EmbeddingRetriever(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        device='cpu'
    )
    
    # Get results from both methods
    print(f"\n{'─'*100}")
    print("RETRIEVING RESULTS...")
    print(f"{'─'*100}")
    
    tfidf_results = tfidf.search(query, top_k=top_k)
    embedding_results = embedding.search(query, top_k=top_k)
    
    # Display TF-IDF results
    print("\n" + "🔍 TF-IDF RESULTS".center(100, "─"))
    print("Characteristics: Keyword-based, finds exact term matches\n")
    
    if not tfidf_results:
        print("❌ No results found.")
    else:
        for result in tfidf_results:
            print(f"[Rank {result['rank']}] Score: {result['score']:.4f} | Chunk ID: {result['chunk_id']}")
            print(f"  {result['content']}")
            print()
    
    # Display Embedding results
    print("\n" + "🤖 EMBEDDING RESULTS".center(100, "─"))
    print("Characteristics: Semantic-based, understands meaning and context\n")
    
    if not embedding_results:
        print("❌ No results found.")
    else:
        for result in embedding_results:
            print(f"[Rank {result['rank']}] Score: {result['score']:.4f}")
            print(f"  {result['content']}")
            print()
    
    # Analyze differences
    print("\n" + "📊 ANALYSIS".center(100, "─"))
    analyze_differences(tfidf_results, embedding_results)
    
    print("\n" + "="*100 + "\n")
    
    return tfidf_results, embedding_results


def analyze_differences(tfidf_results, embedding_results):
    """
    Analyze and display key differences between the two retrieval methods.
    
    Args:
        tfidf_results: Results from TF-IDF retrieval
        embedding_results: Results from embedding retrieval
    """
    print("\nKey Differences:")
    print("-" * 100)
    
    # Score comparison
    print("\n1. SCORING:")
    print("   • TF-IDF: Higher is better (cosine similarity, 0-1 range)")
    print("   • Embeddings: Lower is better (L2 distance in FAISS)")
    
    if tfidf_results:
        avg_tfidf = sum(r['score'] for r in tfidf_results) / len(tfidf_results)
        print(f"   • TF-IDF avg score: {avg_tfidf:.4f}")
    
    if embedding_results:
        avg_embed = sum(r['score'] for r in embedding_results) / len(embedding_results)
        print(f"   • Embedding avg score: {avg_embed:.4f}")
    
    # Result overlap
    print("\n2. RESULT OVERLAP:")
    if tfidf_results and embedding_results:
        # Compare by content similarity (first 100 chars)
        tfidf_contents = {r['content'][:100] for r in tfidf_results}
        embed_contents = {r['content'][:100] for r in embedding_results}
        overlap = len(tfidf_contents & embed_contents)
        print(f"   • Common results in top-{len(tfidf_results)}: {overlap}")
        print(f"   • Overlap percentage: {(overlap/len(tfidf_results)*100):.1f}%")
    
    # When to use each
    print("\n3. WHEN TO USE EACH METHOD:")
    print("   • TF-IDF:")
    print("     - Best for exact keyword matches")
    print("     - Fast, no model loading required")
    print("     - Good for specific entity/name searches")
    print("     - Works well with domain-specific terminology")
    
    print("\n   • Embeddings:")
    print("     - Best for semantic/conceptual queries")
    print("     - Understands synonyms and paraphrases")
    print("     - Better for natural language questions")
    print("     - Captures context and meaning")
    print("     - Requires pre-built vector store and model loading")


def run_multiple_queries():
    """Run comparison on multiple example queries."""
    
    queries = [
        "luxury hotels with spa",
        "family-friendly accommodation near beach",
        "Hotel Bellevue",  # Specific entity name
        "romantic getaway villa",
        "budget apartments Croatia"
    ]
    
    print("\n" + "="*100)
    print("MULTI-QUERY COMPARISON TEST")
    print("="*100)
    
    # Initialize retrievers once (more efficient)
    print("\nInitializing retrievers...")
    tfidf = TFIDFRetriever(chunks_path="chunks.json")
    embedding = EmbeddingRetriever(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        device='cpu'
    )
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*100}")
        print(f"Query {i}/{len(queries)}: '{query}'")
        print(f"{'='*100}")
        
        tfidf_results = tfidf.search(query, top_k=3)
        embedding_results = embedding.search(query, top_k=3)
        
        # Quick comparison
        print("\n📊 Quick Comparison:")
        print(f"  TF-IDF:     {len(tfidf_results)} results, avg score: {sum(r['score'] for r in tfidf_results)/max(len(tfidf_results),1):.4f}")
        print(f"  Embeddings: {len(embedding_results)} results, avg score: {sum(r['score'] for r in embedding_results)/max(len(embedding_results),1):.4f}")
        
        # Show top result from each
        if tfidf_results:
            print(f"\n  🔍 TF-IDF Top Result:")
            print(f"     {tfidf_results[0]['content']}")
        
        if embedding_results:
            print(f"\n  🤖 Embedding Top Result:")
            print(f"     {embedding_results[0]['content']}")


def main():
    """Main entry point for the comparison script."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   RETRIEVAL METHODS COMPARISON DEMO                          ║
║                   TF-IDF vs Semantic Embeddings                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Single query comparison
    query = "luxury hotels with spa and wellness facilities"
    compare_retrievers(query, top_k=5)
    
    # Optional: Uncomment to run multiple queries
    # run_multiple_queries()


if __name__ == "__main__":
    main()
