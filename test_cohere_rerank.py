"""
Test suite for comparing Cohere rerank models.

Compares different Cohere rerank models (e.g., rerank-english-v3.0, rerank-v4.0-pro)
to determine which model selects better chunks for given queries.
"""

import cohere
from typing import List, Dict, Any, Tuple
from embedding import EmbeddingRetriever
from tfidf_lc_retriever import TFIDFLangChainRetriever
from langchain_classic.retrievers import MergerRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_classic.retrievers import ContextualCompressionRetriever
import json
from datetime import datetime
import time
import numpy as np


# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ===== Configuration =====
COHERE_API_KEY = "Ebolg4Tx79anZgqfJK2BcrLnpXsuvcGeT1obe91V"
co = cohere.ClientV2(COHERE_API_KEY)

# Cohere models to test
COHERE_MODELS = [
    "rerank-english-v3.0",
    "rerank-multilingual-v3.0",
    "rerank-v3.5",
    "rerank-v4.0-pro"
]

# Test configuration
TOP_K_RETRIEVAL = 30  # Number of documents to retrieve before reranking
TOP_K_RERANK = 5      # Number of documents to return after reranking
REDUNDANCY_THRESHOLD = 0.85
API_RATE_LIMIT = 10   # Max API calls per minute
API_DELAY = 6.5       # Delay in seconds between API calls (60s / 10 calls = 6s, +0.5s buffer)


class CohereRerankTester:
    """Test suite for comparing different Cohere rerank models."""
    
    def __init__(self, chunks_path: str = "chunks.json"):
        """Initialize retrievers for testing.
        
        Args:
            chunks_path: Path to chunks JSON file
        """
        self.chunks_path = chunks_path
        self.results_history = []
        self.api_calls_count = 0
        # Initialize retrievers once to avoid memory issues
        self.tfidf = None
        self.embedding_minilm = None
        self.embedding_bge = None
        self._initialize_retrievers_once()
    
    def _initialize_retrievers_once(self):
        """Initialize retrievers once at startup."""
        print("\nInitializing retrievers once...")
        self.tfidf = TFIDFLangChainRetriever(chunks_path=self.chunks_path, top_k=10)
        self.embedding_minilm = EmbeddingRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            device='cpu',
            top_k=10
        )
        self.embedding_bge = EmbeddingRetriever(
            model_name='BAAI/bge-m3',
            device='cpu',
            top_k=10
        )
        print("All retrievers initialized successfully")
        
    def _initialize_retrievers(self) -> Tuple:
        """Return cached retrievers."""
        return self.tfidf, self.embedding_minilm, self.embedding_bge
    
    def _get_merged_documents(self, query: str, top_k: int = 30):
        """Retrieve and merge documents from all retrievers.
        
        Args:
            query: Search query
            top_k: Number of top documents to return
            
        Returns:
            List of deduplicated documents
        """
        tfidf, embedding_minilm, embedding_bge = self._initialize_retrievers()
        
        # Create merger retriever
        lotr = MergerRetriever(retrievers=[embedding_minilm, embedding_bge, tfidf])
        
        # Add deduplication filter
        redundant_filter = EmbeddingsRedundantFilter(
            embeddings=embedding_minilm.embeddings,
            similarity_threshold=REDUNDANCY_THRESHOLD
        )
        pipeline = DocumentCompressorPipeline(transformers=[redundant_filter])
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline,
            base_retriever=lotr
        )
        
        # Get deduplicated results
        merged_docs = compression_retriever.invoke(query)
        return merged_docs[:top_k]
    
    def rerank_with_model(self, query: str, docs, model: str, top_n: int = 5):
        """Rerank documents using a specific Cohere model.
        
        Args:
            query: Search query
            docs: List of documents to rerank
            model: Cohere model name
            top_n: Number of top results to return
            
        Returns:
            List of reranked documents with scores
        """
        if not docs:
            return []
        
        top_n = min(top_n, len(docs))
        doc_texts = [d.page_content for d in docs]
        
        try:
            # Rate limiting: delay between API calls
            if self.api_calls_count > 0:
                print(f"⏳ Waiting {API_DELAY}s for rate limiting...")
                time.sleep(API_DELAY)
            
            response = co.rerank(
                model=model,
                query=query,
                documents=doc_texts,
                top_n=top_n
            )
            self.api_calls_count += 1
            
            reranked = []
            for result in response.results:
                doc = docs[result.index]
                doc.metadata = {
                    **doc.metadata,
                    "cohere_score": result.relevance_score,
                    "cohere_model": model,
                    "rerank_position": len(reranked) + 1
                }
                reranked.append(doc)
            
            return reranked
        except Exception as e:
            print(f"ERROR with model {model}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def test_single_query(self, query: str, models: List[str] = None, 
                         top_k_retrieval: int = TOP_K_RETRIEVAL,
                         top_k_rerank: int = TOP_K_RERANK):
        """Test a single query across multiple Cohere models.
        
        Args:
            query: Search query
            models: List of Cohere models to test (default: all models)
            top_k_retrieval: Number of documents to retrieve
            top_k_rerank: Number of documents to return after reranking
            
        Returns:
            Dictionary with results for each model
        """
        if models is None:
            models = COHERE_MODELS
        
        print("\n" + "="*100)
        print(f"TESTING QUERY: '{query}'")
        print("="*100)
        
        # Get merged documents
        print(f"\nRetrieving {top_k_retrieval} documents...")
        try:
            docs = self._get_merged_documents(query, top_k=top_k_retrieval)
            print(f"Retrieved {len(docs)} deduplicated documents")
        except Exception as e:
            print(f"ERROR retrieving documents: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
        
        # Test each model
        results = {}
        for model in models:
            print(f"\nTesting model: {model}")
            try:
                reranked = self.rerank_with_model(query, docs, model, top_n=top_k_rerank)
                
                if reranked:
                    results[model] = {
                        "documents": reranked,
                        "scores": [doc.metadata.get("cohere_score", 0) for doc in reranked],
                        "avg_score": sum(doc.metadata.get("cohere_score", 0) for doc in reranked) / len(reranked),
                        "min_score": min(doc.metadata.get("cohere_score", 0) for doc in reranked),
                        "max_score": max(doc.metadata.get("cohere_score", 0) for doc in reranked)
                    }
                    print(f"Avg score: {results[model]['avg_score']:.4f} | "
                          f"Min: {results[model]['min_score']:.4f} | "
                          f"Max: {results[model]['max_score']:.4f}")
            except Exception as e:
                print(f"ERROR testing model {model}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Display comparison
        try:
            self._display_comparison(query, results)
        except Exception as e:
            print(f"ERROR displaying comparison: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Store results with full chunk content
        try:
            test_result = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "models_tested": models,
                "results": {
                    model: {
                        "avg_score": data["avg_score"],
                        "min_score": data["min_score"],
                        "max_score": data["max_score"],
                        "scores": data["scores"],
                        "top_chunks": [
                            {
                                "rank": i + 1,
                            "content": doc.page_content,
                            "cohere_score": doc.metadata.get("cohere_score", None),
                            "retriever": doc.metadata.get("retriever", "unknown"),
                            "source": doc.metadata.get("source", "unknown"),
                            "chunk_id": doc.metadata.get("chunk_id", None),
                            "metadata": doc.metadata
                        }
                        for i, doc in enumerate(data["documents"])
                    ]
                }
                for model, data in results.items()
            }
        }
            self.results_history.append(test_result)
            print("Saved result to history")
        except Exception as e:
            print(f"ERROR storing result: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def _display_comparison(self, query: str, results: Dict[str, Any]):
        """Display side-by-side comparison of model results.
        
        Args:
            query: Search query
            results: Dictionary with results for each model
        """
        print("\n" + "="*100)
        print("📊 MODEL COMPARISON")
        print("="*100)
        
        # Summary statistics
        print("\n📈 Score Statistics:")
        print(f"{'Model':<35} {'Avg Score':>12} {'Min Score':>12} {'Max Score':>12}")
        print("-" * 75)
        
        sorted_models = sorted(results.items(), key=lambda x: x[1]['avg_score'], reverse=True)
        for model, data in sorted_models:
            print(f"{model:<35} {data['avg_score']:>12.4f} {data['min_score']:>12.4f} {data['max_score']:>12.4f}")
        
        # Best model
        best_model = sorted_models[0][0]
        print(f"\n🏆 Best performing model: {best_model}")
        
        # Top results comparison
        print("\n" + "-"*100)
        print("🔝 TOP RESULTS COMPARISON")
        print("-"*100)
        
        for model, data in results.items():
            print(f"\n{'='*100}")
            print(f"Model: {model}")
            print('='*100)
            
            for i, doc in enumerate(data["documents"], 1):
                score = doc.metadata.get("cohere_score", "N/A")
                retriever = doc.metadata.get("retriever", "unknown")
                source = doc.metadata.get("source", "unknown")
                
                print(f"\n[Rank {i}] Score: {score:.4f} | Retriever: {retriever}")
                print(f"Content: {doc.page_content[:250]}...")
                
        print("\n" + "="*100)
    
    def test_multiple_queries(self, queries: List[str], models: List[str] = None):
        """Test multiple queries across different models.
        
        Args:
            queries: List of search queries
            models: List of Cohere models to test (default: all models)
            
        Returns:
            Dictionary with aggregated results
        """
        if models is None:
            models = COHERE_MODELS
        
        print("\n" + "="*100)
        print("MULTI-QUERY TEST SUITE".center(100))
        print("="*100)
        
        all_results = {}
        
        try:
            for i, query in enumerate(queries, 1):
                print(f"\n\n{'#'*100}")
                print(f"# Query {i}/{len(queries)}")
                print(f"{'#'*100}")
                
                try:
                    query_results = self.test_single_query(query, models)
                    all_results[query] = query_results
                    # Save incrementally after each successful query (always to same file)
                    self.save_results()
                except Exception as e:
                    print(f"ERROR in query {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Auto-save partial results before continuing (always to same file)
                    self.save_results()
            
            # Aggregate statistics
            self._display_aggregate_statistics(all_results, models)
        except Exception as e:
            print(f"CRITICAL ERROR in test_multiple_queries: {str(e)}")
            import traceback
            traceback.print_exc()
            # Save whatever we have so far (to same file)
            if self.results_history:
                self.save_results()
        
        return all_results
    
    def _display_aggregate_statistics(self, all_results: Dict, models: List[str]):
        """Display aggregate statistics across all queries.
        
        Args:
            all_results: Dictionary with results for all queries
            models: List of models tested
        """
        print("\n\n" + "="*100)
        print("📊 AGGREGATE STATISTICS ACROSS ALL QUERIES")
        print("="*100)
        
        model_stats = {model: {"scores": [], "queries": 0} for model in models}
        
        for query, results in all_results.items():
            for model, data in results.items():
                if model in model_stats:
                    model_stats[model]["scores"].extend(data["scores"])
                    model_stats[model]["queries"] += 1
        
        print(f"\n{'Model':<35} {'Avg Score':>12} {'Queries':>10} {'Total Docs':>12}")
        print("-" * 75)
        
        sorted_models = sorted(
            model_stats.items(),
            key=lambda x: sum(x[1]["scores"]) / len(x[1]["scores"]) if x[1]["scores"] else 0,
            reverse=True
        )
        
        for model, stats in sorted_models:
            avg = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
            print(f"{model:<35} {avg:>12.4f} {stats['queries']:>10} {len(stats['scores']):>12}")
        
        print(f"\n🏆 Overall best performing model: {sorted_models[0][0]}")
    
    def save_results(self, filename: str = "cohere_rerank_test_results.json"):
        """Save test results to JSON file.
        
        Args:
            filename: Output filename
        """
        try:
            output_path = f"output/{filename}"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results_history, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            print(f"\nResults saved to {output_path}")
            print(f"Saved {len(self.results_history)} query results")
        except Exception as e:
            print(f"ERROR saving results: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_test_queries(self) -> List[str]:
        """Generate a diverse set of test queries (50+ variations) as questions.
        
        Returns:
            List of test queries covering various categories
        """
        return [
            # ===== LUXURY & PREMIUM (5 queries) =====
            "What luxury hotels offer spa and wellness facilities?",
            "Which five star hotels provide spa treatments?",
            "Where can I find ultra luxury villas with private pools?",
            
            # ===== FAMILY-ORIENTED (6 queries) =====
            "Are there family-friendly hotels near the beach?",
            "Which hotels are suitable for families with children?",
            "Do you have family apartments with kids clubs?",
            
            # ===== ROMANTIC & COUPLES (6 queries) =====
            "What romantic villas are available for couples?",
            "Do you have private villas with sea views?",
            "Where can I book honeymoon suites with Jacuzzis?",
            
            # ===== BUDGET & ECONOMY (5 queries) =====
            "Where can I find affordable apartments in Croatia?",
            "What budget accommodation is available in Kvarner?",
            
            # ===== SPECIFIC AMENITIES (8 queries) =====
            "Which hotels have swimming pools?",
            "What pet-friendly accommodation options are there?",
            "Are there hotels with free WiFi?",
            
            # ===== LOCATION-BASED (7 queries) =====
            "What hotels are located in Lošinj town center?",
            "Is there accommodation near Mali Lošinj?",
            "Which hotels offer sea views?",
            
            # ===== ACTIVITY-BASED (5 queries) =====
            "What hotels cater to diving enthusiasts?",
            "Is there accommodation for hikers and nature lovers?",
            "Where can I find water sports resort accommodation?",
            
            # ===== WELLNESS & HEALTH (4 queries) =====
            "Where are wellness retreat centers located?",
            "Do you have yoga and meditation hotels?",
            "What health spa resort packages are available?",
            
            # ===== SPECIAL OCCASIONS (5 queries) =====
            "What wedding venue accommodation options exist?",
            "Where can I arrange celebration party hotels?",
            "What anniversary dinner package hotels are available?",
            
            # ===== ACCESSIBILITY & SPECIAL NEEDS (3 queries) =====
            "Are there wheelchair accessible hotels?",
            "What accommodation is suitable for elderly guests?",
            "Do any hotels have medical facilities?",
            
            # ===== SEASONAL & WEATHER (3 queries) =====
            "What hotels are best for summer vacations?",
            "Where can I find winter escape accommodation?",
            "Is there all-year-round resort accommodation?",
            
            # ===== GROUP & EVENTS (3 queries) =====
            "Where can I book accommodation for group retreats?",
            "What conference hotels have meeting rooms?",
            "Is there team building accommodation available?",
        ]


def main():
    """Main entry point for Cohere rerank testing."""
    
    print("""
========================================
   COHERE RERANK MODEL COMPARISON SUITE
Test different Cohere rerank models to find the best
========================================
""")
    
    tester = CohereRerankTester()
    
    # ==================== TEST CONFIGURATION ====================
    
    # Option 1: Test a single query across all models
    single_query = "luxury hotels with spa and wellness facilities"
    # tester.test_single_query(single_query)
    
    # Option 2: Test multiple predefined queries
    test_queries = tester.create_test_queries()
    tester.test_multiple_queries(test_queries)
    
    # Option 3: Test specific queries with specific models
    # test_queries = [
    #     "luxury hotels with spa and wellness facilities",
    #     "family-friendly accommodation near beach",
    #     "romantic villa with private pool",
    #     "budget apartments in Croatia"
    # ]
    
    models_to_test = [
        "rerank-english-v3.0",
        "rerank-multilingual-v3.0",  # Uncomment if testing multilingual
        # "rerank-v3.5"  # Uncomment if you have access
    ]
    
    # Run tests
    # tester.test_multiple_queries(test_queries, models=models_to_test)
    
    # Save results
    tester.save_results()
    
    # ============================================================
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()
