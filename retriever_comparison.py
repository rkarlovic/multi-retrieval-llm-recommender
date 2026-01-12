"""
Retriever comparison utilities for analyzing individual retrieval methods.

Provides functions to compare TF-IDF and embedding-based retrievers (MiniLM, BGE-M3).
"""

from embedding import EmbeddingRetriever
from tfidf_lc_retriever import TFIDFLangChainRetriever


def compare_retrievers(query: str, top_k: int = 5):
    """
    Compare TF-IDF and embedding-based retrievers for a given query.
    
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
    tfidf = TFIDFLangChainRetriever(chunks_path="chunks.json")
    
    # Initialize MiniLM embedding retriever
    print("\n[2] Initializing Embedding Retriever (MiniLM)...")
    embedding_minilm = EmbeddingRetriever(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        device='cpu'
    )
    
    # Initialize BGE-M3 embedding retriever
    print("\n[3] Initializing Embedding Retriever (BGE-M3)...")
    embedding_bge = EmbeddingRetriever(
        model_name='BAAI/bge-m3',
        device='cpu'
    )
    
    # Get results from all methods
    print(f"\n{'─'*100}")
    print("RETRIEVING RESULTS...")
    print(f"{'─'*100}")
    
    tfidf_results = tfidf.search(query, top_k=top_k)
    minilm_results = embedding_minilm.invoke(query)
    bge_results = embedding_bge.invoke(query)
    
    # Display TF-IDF results
    print("\n" + "🔍 TF-IDF RESULTS".center(100, "─"))
    print("Characteristics: Keyword-based, exact term matching\n")
    
    if not tfidf_results:
        print("❌ No results found.")
    else:
        for result in tfidf_results:
            print(f"[Rank {result['rank']}] Score: {result['score']:.4f}")
            print(f"  {result['content']}")
            print()
    
    # Display MiniLM results
    print("\n" + "🤖 MINILM RESULTS".center(100, "─"))
    print("Model: sentence-transformers/all-MiniLM-L6-v2\n")
    
    if not minilm_results:
        print("❌ No results found.")
    else:
        for i, doc in enumerate(minilm_results, 1):
            score = doc.metadata.get('score', 'N/A')
            print(f"[Rank {i}] Score: {score}")
            print(f"  {doc.page_content[:200]}...")
            print()
    
    # Display BGE-M3 results
    print("\n" + "🤖 BGE-M3 RESULTS".center(100, "─"))
    print("Model: BAAI/bge-m3\n")
    
    if not bge_results:
        print("❌ No results found.")
    else:
        for i, doc in enumerate(bge_results, 1):
            score = doc.metadata.get('score', 'N/A')
            print(f"[Rank {i}] Score: {score}")
            print(f"  {doc.page_content[:200]}...")
            print()
    
    # Analyze differences
    print("\n" + "📊 ANALYSIS".center(100, "─"))
    analyze_differences(tfidf_results, minilm_results, bge_results)
    
    print("\n" + "="*100 + "\n")
    
    return tfidf_results, minilm_results, bge_results


def analyze_differences(tfidf_results, minilm_results, bge_results):
    """
    Analyze and display key differences between the three retrieval methods.
    
    Args:
        tfidf_results: Results from TF-IDF retrieval
        minilm_results: Results from MiniLM retrieval
        bge_results: Results from BGE-M3 retrieval
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
    
    if minilm_results:
        avg_minilm = sum(doc.metadata.get('score', 0) for doc in minilm_results) / len(minilm_results)
        print(f"   • MiniLM avg score: {avg_minilm:.4f}")
    
    if bge_results:
        avg_bge = sum(doc.metadata.get('score', 0) for doc in bge_results) / len(bge_results)
        print(f"   • BGE-M3 avg score: {avg_bge:.4f}")
    
    # Result overlap
    print("\n2. RESULT OVERLAP:")
    if tfidf_results and minilm_results:
        # Compare by content similarity (first 100 chars)
        tfidf_contents = {r['content'][:100] for r in tfidf_results}
        minilm_contents = {doc.page_content[:100] for doc in minilm_results}
        overlap = len(tfidf_contents & minilm_contents)
        print(f"   • TF-IDF ∩ MiniLM: {overlap}/{len(tfidf_results)} results")
    
    # Method characteristics
    print("\n3. METHOD CHARACTERISTICS:")
    print("   • TF-IDF:")
    print("     - Best for exact keyword matches")
    print("     - Fast, no embeddings needed")
    print("     - Good for specific entity/name searches")
    print("     - Works well with domain-specific terminology")
    
    print("\n   • MiniLM (all-MiniLM-L6-v2):")
    print("     - Lightweight, fast inference")
    print("     - Good for general semantic search")
    print("     - Smaller model size")
    
    print("\n   • BGE-M3 (BAAI/bge-m3):")
    print("     - Multilingual capability")
    print("     - Better performance on dense retrieval")
    print("     - Larger model, more accurate")


def run_multiple_queries(queries: list):
    """Run comparison on multiple example queries."""
    
    print("\n" + "="*100)
    print("MULTI-QUERY COMPARISON TEST")
    print("="*100)
    
    # Initialize retrievers once (more efficient)
    print("\nInitializing retrievers...")
    tfidf = TFIDFLangChainRetriever(chunks_path="chunks.json")
    embedding_minilm = EmbeddingRetriever(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        device='cpu'
    )
    embedding_bge = EmbeddingRetriever(
        model_name='BAAI/bge-m3',
        device='cpu'
    )
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*100}")
        print(f"Query {i}/{len(queries)}: '{query}'")
        print(f"{'='*100}")
        
        tfidf_results = tfidf.search(query, top_k=3)
        minilm_results = embedding_minilm.invoke(query)
        bge_results = embedding_bge.invoke(query)
        
        # Quick comparison
        print("\n📊 Quick Comparison:")
        tfidf_avg = sum(r['score'] for r in tfidf_results) / max(len(tfidf_results), 1)
        minilm_avg = sum(doc.metadata.get('score', 0) for doc in minilm_results) / max(len(minilm_results), 1)
        bge_avg = sum(doc.metadata.get('score', 0) for doc in bge_results) / max(len(bge_results), 1)
        print(f"  TF-IDF: {len(tfidf_results)} results, avg score: {tfidf_avg:.4f}")
        print(f"  MiniLM: {len(minilm_results)} results, avg score: {minilm_avg:.4f}")
        print(f"  BGE-M3: {len(bge_results)} results, avg score: {bge_avg:.4f}")
        
        # Show top result from each
        if tfidf_results:
            print(f"\n  🔍 TF-IDF Top Result:")
            print(f"     {tfidf_results[0]['content'][:150]}...")
        
        if minilm_results:
            print(f"\n  🤖 MiniLM Top Result:")
            print(f"     {minilm_results[0].page_content[:150]}...")
        
        if bge_results:
            print(f"\n  🤖 BGE-M3 Top Result:")
            print(f"     {bge_results[0].page_content[:150]}...")
