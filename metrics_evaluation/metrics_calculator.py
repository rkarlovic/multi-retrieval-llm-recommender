"""
Metrics Evaluation Framework - Standalone metrics calculation for retriever comparison.

Calculates: Precision@k, Recall@k, MRR, and NDCG@k for different retrieval methods.
"""

import json
import numpy as np
from typing import List, Dict, Tuple


class MetricsCalculator:
    """Calculate retrieval quality metrics."""
    
    @staticmethod
    def precision_at_k(relevant_items: set, retrieved_items: List[int], k: int) -> float:
        """
        Precision@k: How many of top-k retrieved items are relevant?
        
        Args:
            relevant_items: Set of indices that are relevant
            retrieved_items: List of retrieved item indices in order
            k: Cutoff position
            
        Returns:
            Precision@k value (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
        
        top_k = retrieved_items[:k]
        hits = len([item for item in top_k if item in relevant_items])
        return hits / k
    
    @staticmethod
    def recall_at_k(relevant_items: set, retrieved_items: List[int], k: int) -> float:
        """
        Recall@k: What proportion of all relevant items were retrieved in top-k?
        
        Args:
            relevant_items: Set of indices that are relevant
            retrieved_items: List of retrieved item indices in order
            k: Cutoff position
            
        Returns:
            Recall@k value (0.0 to 1.0)
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k = retrieved_items[:k]
        hits = len([item for item in top_k if item in relevant_items])
        return hits / len(relevant_items)
    
    @staticmethod
    def mrr(relevant_items: set, retrieved_items: List[int]) -> float:
        """
        Mean Reciprocal Rank: How early is the first relevant item?
        
        Used to measure how quickly you find relevant answers.
        Perfect: 1.0 (first result is relevant)
        Good: > 0.5 (first relevant in top-2)
        
        Args:
            relevant_items: Set of indices that are relevant
            retrieved_items: List of retrieved item indices in order
            
        Returns:
            MRR value (0.0 to 1.0)
        """
        for rank, item in enumerate(retrieved_items, 1):
            if item in relevant_items:
                return 1.0 / rank
        return 0.0
    
    @staticmethod
    def ndcg_at_k(relevant_items: set, retrieved_items: List[int], k: int) -> float:
        """
        NDCG@k: Normalized Discounted Cumulative Gain.
        
        Accounts for ranking quality - penalizes relevant items ranked lower.
        Best metric for comparing ranking quality.
        
        Args:
            relevant_items: Set of indices that are relevant (binary relevance)
            retrieved_items: List of retrieved item indices in order
            k: Cutoff position
            
        Returns:
            NDCG@k value (0.0 to 1.0)
        """
        # DCG calculation
        dcg = 0.0
        for rank, item in enumerate(retrieved_items[:k], 1):
            if item in relevant_items:
                dcg += 1.0 / np.log2(rank + 1)
        
        # Ideal DCG (all relevant items ranked first)
        ideal_dcg = 0.0
        num_relevant = min(len(relevant_items), k)
        for rank in range(1, num_relevant + 1):
            ideal_dcg += 1.0 / np.log2(rank + 1)
        
        if ideal_dcg == 0:
            return 0.0
        
        return dcg / ideal_dcg
    
    @staticmethod
    def calculate_all_metrics(relevant_items: set, retrieved_items: List[int], 
                             k_values: List[int] = [5, 10]) -> Dict:
        """
        Calculate all metrics for a single query.
        
        Args:
            relevant_items: Set of relevant item indices
            retrieved_items: List of retrieved items (in order)
            k_values: Cutoff positions to evaluate
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {
            "mrr": MetricsCalculator.mrr(relevant_items, retrieved_items),
        }
        
        for k in k_values:
            metrics[f"precision@{k}"] = MetricsCalculator.precision_at_k(
                relevant_items, retrieved_items, k
            )
            metrics[f"recall@{k}"] = MetricsCalculator.recall_at_k(
                relevant_items, retrieved_items, k
            )
            metrics[f"ndcg@{k}"] = MetricsCalculator.ndcg_at_k(
                relevant_items, retrieved_items, k
            )
        
        return metrics


class MetricsReport:
    """Generate formatted metrics comparison reports."""
    
    @staticmethod
    def calculate_retriever_metrics(test_queries: List[Dict], 
                                   retriever_results: Dict[str, List[List[int]]],
                                   k_values: List[int] = [5, 10]) -> Dict[str, Dict]:
        """
        Calculate metrics for all retrievers across all test queries.
        
        Args:
            test_queries: List of query dicts with 'relevant_chunks_indices' field
            retriever_results: Dict[retriever_name] -> List[List[results per query]]
            k_values: Cutoff positions
            
        Returns:
            Dict[retriever_name] -> aggregated metrics
        """
        calculator = MetricsCalculator()
        results = {}
        
        for retriever_name, all_results in retriever_results.items():
            query_metrics = []
            
            for query_idx, retrieved_items in enumerate(all_results):
                relevant_items = set(test_queries[query_idx]["relevant_chunks_indices"])
                metrics = calculator.calculate_all_metrics(relevant_items, retrieved_items, k_values)
                query_metrics.append(metrics)
            
            # Calculate averages
            avg_metrics = {}
            if query_metrics:
                for key in query_metrics[0].keys():
                    avg_metrics[key] = np.mean([m[key] for m in query_metrics])
            
            results[retriever_name] = {
                "per_query": query_metrics,
                "averages": avg_metrics,
                "num_queries": len(query_metrics)
            }
        
        return results
    
    @staticmethod
    def print_comparison_table(metrics_results: Dict[str, Dict]):
        """Print formatted comparison table of all retrievers."""
        
        print("\n" + "="*80)
        print("RETRIEVER METRICS COMPARISON")
        print("="*80)
        
        if not metrics_results:
            print("No results to display")
            return
        
        # Get all metric names from first retriever
        first_retriever = next(iter(metrics_results.values()))
        metric_names = sorted(first_retriever["averages"].keys())
        
        # Header
        print(f"{'Retriever':<25}", end="")
        for metric in metric_names:
            print(f"  {metric:<12}", end="")
        print()
        print("-" * 80)
        
        # Data rows
        for retriever_name in sorted(metrics_results.keys()):
            print(f"{retriever_name:<25}", end="")
            avg_metrics = metrics_results[retriever_name]["averages"]
            
            for metric in metric_names:
                value = avg_metrics.get(metric, 0.0)
                print(f"  {value:>10.3f}  ", end="")
            print()
        
        print("="*80)
    
    @staticmethod
    def print_detailed_report(metrics_results: Dict[str, Dict], 
                             test_queries: List[Dict]):
        """Print detailed per-query metrics."""
        
        print("\n" + "="*80)
        print("DETAILED PER-QUERY METRICS")
        print("="*80)
        
        for retriever_name in sorted(metrics_results.keys()):
            print(f"\n{retriever_name}")
            print("-" * 80)
            
            per_query = metrics_results[retriever_name]["per_query"]
            
            for query_idx, metrics in enumerate(per_query):
                query_obj = test_queries[query_idx]
                print(f"\n  Query {query_idx + 1}: '{query_obj['query']}'")
                print(f"  Description: {query_obj['description']}")
                print(f"  Relevant chunks: {sorted(query_obj['relevant_chunks_indices'])}")
                
                for metric_name, value in sorted(metrics.items()):
                    print(f"    {metric_name:<15}: {value:.3f}")
        
        print("\n" + "="*80)
