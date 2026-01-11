"""
Multi-Retrieval System using LangChain's MergerRetriever.

Combines TF-IDF and two embedding models (MiniLM + BGE-M3) retrieval results
with deduplication and ranking.
"""

from embedding import EmbeddingRetriever
from langchain_classic.retrievers import MergerRetriever, ContextualCompressionRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from tfidf_lc_retriever import TFIDFLangChainRetriever

# ===== Configurable settings =====
TOP_K_TFIDF = 5
TOP_K_MINILM = 5
TOP_K_BGE = 5
FINGERPRINT_LEN = 200
REDUNDANCY_THRESHOLD = 0.95


def merged_retrieval(query: str, top_k: int = 5):
    """
    Retrieve results from TF-IDF and both embedding models using MergerRetriever.
    
    Args:
        query: Search query string
        top_k: Number of results per retriever
        
    Returns:
        List of merged documents
    """
    print("\n" + "="*100)
    print(f"MULTI-RETRIEVAL SYSTEM (LangChain MergerRetriever)")
    print(f"Query: '{query}'")
    print("="*100)
    
    # Initialize all three retrievers
    print("\n[1/3] Initializing TF-IDF Retriever...")
    tfidf = TFIDFLangChainRetriever(chunks_path="chunks.json", top_k=TOP_K_TFIDF)
    
    print("\n[2/3] Initializing Embedding Retriever (MiniLM)...")
    embedding_minilm = EmbeddingRetriever(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        device='cpu',
        top_k=TOP_K_MINILM
    )
    
    print("\n[3/3] Initializing Embedding Retriever (BGE-M3)...")
    embedding_bge = EmbeddingRetriever(
        model_name='BAAI/bge-m3',
        device='cpu',
        top_k=TOP_K_BGE
    )
    
    # Create MergerRetriever to combine all three
    print(f"\n{'─'*100}")
    print("CREATING MERGER RETRIEVER...")
    print(f"{'─'*100}\n")
    
    lotr = MergerRetriever(retrievers=[tfidf, embedding_minilm, embedding_bge])
    
    # Optionally wrap with deduplication filter
    print("Adding redundant filter for deduplication...")
    redundant_filter = EmbeddingsRedundantFilter(
        embeddings=embedding_minilm.embeddings,
        similarity_threshold=REDUNDANCY_THRESHOLD
    )
    pipeline = DocumentCompressorPipeline(transformers=[redundant_filter])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline,
        base_retriever=lotr
    )
    
    # Get merged results
    print(f"\nRetrieving results...")
    # Get merged results BEFORE compression to track duplicates
    merged_pre = lotr.invoke(query)
    # Get results AFTER compression (deduplication)
    merged_docs = compression_retriever.invoke(query)

    # Identify removed duplicates by content fingerprint
    def fp(doc):
        return (doc.page_content or "").strip().lower()[:FINGERPRINT_LEN]

    final_fps = {fp(d) for d in merged_docs}
    removed = [d for d in merged_pre if fp(d) not in final_fps]

    # Overlap analysis across retrievers (TF-IDF, MiniLM, BGE-M3)
    retriever_sets = {}
    for d in merged_pre:
        rname = d.metadata.get('retriever', 'unknown')
        retriever_sets.setdefault(rname, set()).add(fp(d))

    # Compute pairwise overlaps
    def overlap_stats(a_name, b_name):
        a = retriever_sets.get(a_name, set())
        b = retriever_sets.get(b_name, set())
        inter = a & b
        return {
            'a': a_name,
            'b': b_name,
            'count': len(inter),
            'pct_a': (len(inter) / max(len(a), 1)) * 100.0,
            'pct_b': (len(inter) / max(len(b), 1)) * 100.0,
            'size_a': len(a),
            'size_b': len(b)
        }

    retriever_names = list(retriever_sets.keys())
    pairwise = []
    for i in range(len(retriever_names)):
        for j in range(i + 1, len(retriever_names)):
            pairwise.append(overlap_stats(retriever_names[i], retriever_names[j]))

    # Duplicates before deduplication: fingerprints present in >1 retriever
    fp_to_retrievers = {}
    for rname, fps in retriever_sets.items():
        for f in fps:
            fp_to_retrievers.setdefault(f, set()).add(rname)
    duplicates_before = sum(1 for v in fp_to_retrievers.values() if len(v) > 1)
    
    # Display results
    print(f"\n{'='*100}")
    print("🎯 MERGED RESULTS (Deduplicated)".center(100))
    print(f"{'='*100}\n")
    print(f"Total unique results: {len(merged_docs)}\n")

    # Print duplicates summary and pairwise overlaps
    print(f"Duplicates before deduplication (cross-retriever): {duplicates_before}")
    if pairwise:
        print("Pairwise overlap (count | % of A | % of B):")
        for p in pairwise:
            print(f"  {p['a']} ∩ {p['b']}: {p['count']} | {p['pct_a']:.1f}% of {p['a']} | {p['pct_b']:.1f}% of {p['b']}")
    else:
        print("No pairwise overlap (only one retriever produced results).")

    # Print removed duplicates summary
    print("Removed duplicate results:" if removed else "No duplicates removed.")
    for i, doc in enumerate(removed, 1):
        src = doc.metadata.get('source', 'unknown')
        retr = doc.metadata.get('retriever', 'unknown')
        cid = doc.metadata.get('chunk_id', 'n/a')
        print(f"  [-] From retriever={retr} | source={src} | chunk_id={cid}")
        print(f"      {doc.page_content[:200]}...")
        print()
    
    for rank, doc in enumerate(merged_docs, 1):
        score = doc.metadata.get('score', 'N/A')
        src = doc.metadata.get('source', 'unknown')
        retr = doc.metadata.get('retriever', 'unknown')
        print(f"[{rank}] Score: {score} | Retriever: {retr} | Source: {src}")
        print(f"  {doc.page_content[:200]}...")
        print()
    
    return merged_docs


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
    tfidf = TFIDFRetriever(chunks_path="chunks.json")
    
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
    tfidf = TFIDFRetriever(chunks_path="chunks.json")
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


def main():
    """Main entry point for the multi-retrieval system."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MULTI-RETRIEVAL SYSTEM (LangChain)                        ║
║            TF-IDF + MiniLM + BGE-M3 Embeddings (Merged & Deduplicated)       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # ==================== EDIT QUERY HERE ====================
    query = "luxury hotels with spa and wellness facilities"
    # ========================================================
    
    # Merged retrieval with deduplication
    merged_results = merged_retrieval(query, top_k=5)
    
    # Optional: Compare all three methods individually (uncomment to see detailed comparison)
    # compare_retrievers(query, top_k=5)
    
    # Optional: Run multiple queries (uncomment to test)
    # queries = [
    #     "luxury hotels with spa",
    #     "family-friendly accommodation near beach",
    #     "romantic getaway villa",
    #     "budget apartments Croatia"
    # ]
    # run_multiple_queries(queries)


if __name__ == "__main__":
    main()
