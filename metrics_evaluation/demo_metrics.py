"""
METRICS DEMO - Example evaluation showing metrics on your retriever comparison.

Run this to see how different retrieval methods perform on sample queries.
No need to integrate into existing pipeline - this is standalone for presentation.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import retriever modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from embedding import EmbeddingRetriever
from tfidf_lc_retriever import TFIDFLangChainRetriever
from metrics_calculator import MetricsCalculator, MetricsReport


def load_test_queries(test_file: str) -> list:
    """Load test queries with labeled relevant chunks."""
    with open(test_file, 'r') as f:
        return json.load(f)


def load_chunks(chunks_file: str) -> list:
    """Load chunks for reference."""
    with open(chunks_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_demo():
    """Run metrics demonstration."""
    
    print("\n" + "="*80)
    print("RETRIEVER METRICS EVALUATION DEMO")
    print("="*80)
    
    # Load test queries and chunks
    metrics_dir = Path(__file__).parent
    project_root = metrics_dir.parent
    
    test_queries_file = metrics_dir / "test_queries.json"
    chunks_file = project_root / "chunks.json"
    
    print(f"\nLoading test queries from: {test_queries_file}")
    print(f"Loading chunks from: {chunks_file}")
    
    test_queries = load_test_queries(str(test_queries_file))
    chunks = load_chunks(str(chunks_file))
    
    print(f"✓ Loaded {len(test_queries)} test queries")
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Initialize retrievers
    print("\n" + "-"*80)
    print("INITIALIZING RETRIEVERS")
    print("-"*80)
    
    print("\n[1] Initializing TF-IDF Retriever...")
    tfidf_retriever = TFIDFLangChainRetriever(chunks_path=str(chunks_file))
    
    print("\n[2] Initializing MiniLM Embedding Retriever...")
    minilm_retriever = EmbeddingRetriever(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        vectorstore_path=str(project_root / 'vector_stores' / 'vectorstore_all_minilm_l6_v2'),
        device='cpu'
    )
    
    print("\n[3] Initializing BGE-M3 Embedding Retriever...")
    bge_retriever = EmbeddingRetriever(
        model_name='BAAI/bge-m3',
        vectorstore_path=str(project_root / 'vector_stores' / 'vectorstore_bge_m3'),
        device='cpu'
    )
    
    # Run retrievers on test queries
    print("\n" + "-"*80)
    print("RUNNING RETRIEVALS")
    print("-"*80)
    
    retrievers = {
        "TF-IDF": tfidf_retriever,
        "MiniLM-L6-v2": minilm_retriever,
        "BGE-M3": bge_retriever
    }
    
    retriever_results = {}
    top_k = 10
    
    for name, retriever in retrievers.items():
        print(f"\nRunning {name}...")
        results = []
        
        for query_obj in test_queries:
            query = query_obj["query"]
            print(f"  Query: '{query}'")
            
            try:
                # All retrievers use invoke() which returns List[Document]
                docs = retriever.invoke(query)
                
                # Extract indices by matching content (only first match to avoid duplicates)
                retrieved_indices = []
                for doc in docs[:top_k]:
                    # Find matching chunk index
                    for idx, chunk in enumerate(chunks):
                        if chunk["content"] == doc.page_content and idx not in retrieved_indices:
                            retrieved_indices.append(idx)
                            break
                
                # Debug: show what was retrieved
                relevant = set(query_obj["relevant_chunks_indices"])
                hits = [idx for idx in retrieved_indices if idx in relevant]
                if hits:
                    print(f"    ✓ Found relevant chunks at positions: {hits}")
                else:
                    print(f"    ✗ No relevant chunks found. Retrieved: {retrieved_indices[:5]}")
                
                results.append(retrieved_indices)
                    
            except Exception as e:
                print(f"    Error: {e}")
                results.append([])
        
        retriever_results[name] = results
    
    # Calculate metrics
    print("\n" + "-"*80)
    print("CALCULATING METRICS")
    print("-"*80)
    
    metrics_results = MetricsReport.calculate_retriever_metrics(
        test_queries, 
        retriever_results,
        k_values=[5, 10]
    )
    
    # Display results
    print("\n")
    MetricsReport.print_comparison_table(metrics_results)
    MetricsReport.print_detailed_report(metrics_results, test_queries)
    
    # Summary and interpretation
    print("\n" + "="*80)
    print("METRICS INTERPRETATION GUIDE")
    print("="*80)
    print("""
Precision@k:
  - How many of top-k results are relevant?
  - Range: 0.0 to 1.0 (1.0 = all top-k are relevant)
  - Use for: "Are my top results good?"

Recall@k:
  - Of all relevant items, how many did I retrieve in top-k?
  - Range: 0.0 to 1.0 (1.0 = found all relevant items)
  - Use for: "Did I miss important results?"

MRR (Mean Reciprocal Rank):
  - How quickly did I find the first relevant result?
  - Range: 0.0 to 1.0 (1.0 = first result relevant)
  - Use for: "For single-answer queries, how good is ranking?"
  - Good threshold: > 0.5 (first relevant in top-2)

NDCG@k (Normalized Discounted Cumulative Gain):
  - Overall ranking quality (penalties for relevant items ranked low)
  - Range: 0.0 to 1.0 (1.0 = perfect ranking)
  - Use for: "Which retriever has the best overall ranking?"
  - This is the BEST single metric for comparing retrievers

RECOMMENDATIONS FOR YOUR PROJECT:
1. Primary metric: NDCG@10 (best overall ranking quality)
2. Secondary: Precision@5 (quality of top results shown to users)
3. Tertiary: Recall@10 (aren't we missing relevant items?)
4. Watch: MRR (if you have single best-answer queries)
""")
    
    print("="*80)


if __name__ == "__main__":
    run_demo()
