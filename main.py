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
from retriever_comparison import compare_retrievers, run_multiple_queries
import cohere

# ===== Configurable settings =====
TOP_K = 30
TOP_K_TFIDF = 10
TOP_K_MINILM = 10
TOP_K_BGE = 10
FINGERPRINT_LEN = 200
REDUNDANCY_THRESHOLD = 0.85
COHERE_RERANK_MODEL = "rerank-english-v3.0"
# COHERE_RERANK_MODEL = "rerank-v4.0-pro"
COHERE_TOP_K = 5

co = cohere.ClientV2("Ebolg4Tx79anZgqfJK2BcrLnpXsuvcGeT1obe91V")


def rerank_with_cohere(query: str, docs, top_n: int = 5):
    """Rerank a list of LangChain Documents using Cohere and return the top_n."""
    if not docs:
        return []

    top_n = min(top_n, len(docs))
    doc_texts = [d.page_content for d in docs]

    response = co.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=doc_texts,
        top_n=top_n
    )

    reranked = []
    for result in response.results:
        doc = docs[result.index]
        doc.metadata = {**doc.metadata, "cohere_score": result.relevance_score}
        reranked.append(doc)

    return reranked


def merged_retrieval(query: str, top_k: int = 5):
    """
    Retrieve results from TF-IDF and both embedding models using MergerRetriever.
    
    Args:
        query: Search query string
        top_k: Number of results per retriever
        
    Returns:
        List of Cohere-reranked documents
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
    
    lotr = MergerRetriever(retrievers=[embedding_minilm, embedding_bge, tfidf])
    
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
    
    # Limit to top_k final results after deduplication
    dedup_docs = merged_docs[:top_k]

    # Identify removed duplicates by content fingerprint
    def fp(doc):
        return (doc.page_content or "").strip().lower()[:FINGERPRINT_LEN]

    final_fps = {fp(d) for d in dedup_docs}
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
    print("🎯 MERGED RESULTS (Deduplicated & Top-K Selected)".center(100))
    print(f"{'='*100}\n")
    print(f"Total retrieved before deduplication: {len(merged_pre)}")
    print(f"After deduplication: {len(dedup_docs)} (top {top_k} selected)")
    print(f"\nRetriever contributions: TF-IDF={TOP_K_TFIDF}, MiniLM={TOP_K_MINILM}, BGE-M3={TOP_K_BGE}\n")

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
    
    print("\nRunning Cohere rerank on deduplicated results...")
    reranked_docs = rerank_with_cohere(query, dedup_docs, top_n=COHERE_TOP_K)

    for rank, doc in enumerate(reranked_docs, 1):
        score = doc.metadata.get('score', 'N/A')
        src = doc.metadata.get('source', 'unknown')
        retr = doc.metadata.get('retriever', 'unknown')
        co_score = doc.metadata.get('cohere_score', 'N/A')
        print(f"[{rank}] Cohere score: {co_score} | Retriever: {retr} | Source: {src} | Orig score: {score}")
        print(f"  {doc.page_content[:200]}...")
        print()
    
    return reranked_docs


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
    merged_results = merged_retrieval(query, top_k=TOP_K)
    
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
