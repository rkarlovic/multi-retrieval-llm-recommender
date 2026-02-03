"""
Chunk Quality Evaluation - Standalone analysis of chunking quality.

Analyzes: chunk size consistency, semantic coherence, overlap, and content quality.
"""

import json
import re
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path


class ChunkQualityAnalyzer:
    """Analyze quality of chunks."""
    
    @staticmethod
    def word_count(text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    @staticmethod
    def token_estimate(text: str) -> int:
        """Rough token estimate (1 token ≈ 0.75 words)."""
        return int(len(text.split()) * 0.75)
    
    @staticmethod
    def char_count(text: str) -> int:
        """Count characters."""
        return len(text)
    
    @staticmethod
    def analyze_chunk_sizes(chunks: List[Dict]) -> Dict:
        """Analyze chunk size distribution."""
        word_counts = []
        token_estimates = []
        char_counts = []
        
        for chunk in chunks:
            content = chunk.get('content', '')
            word_counts.append(ChunkQualityAnalyzer.word_count(content))
            token_estimates.append(ChunkQualityAnalyzer.token_estimate(content))
            char_counts.append(ChunkQualityAnalyzer.char_count(content))
        
        return {
            "word_count": {
                "min": min(word_counts),
                "max": max(word_counts),
                "mean": np.mean(word_counts),
                "median": np.median(word_counts),
                "std": np.std(word_counts),
                "q25": np.percentile(word_counts, 25),
                "q75": np.percentile(word_counts, 75),
            },
            "token_estimate": {
                "min": min(token_estimates),
                "max": max(token_estimates),
                "mean": np.mean(token_estimates),
                "median": np.median(token_estimates),
                "std": np.std(token_estimates),
            },
            "char_count": {
                "min": min(char_counts),
                "max": max(char_counts),
                "mean": np.mean(char_counts),
                "median": np.median(char_counts),
                "std": np.std(char_counts),
            }
        }
    
    @staticmethod
    def detect_content_quality(chunks: List[Dict]) -> Dict:
        """Analyze content quality (signal vs noise)."""
        html_pattern = re.compile(r'<[^>]+>|\[\[.*?\]\]|href=|onclick=')
        
        quality_scores = []
        
        for chunk in chunks:
            content = chunk.get('content', '')
            if not content:
                quality_scores.append(0.0)
                continue
            
            # Count content words vs HTML/markup
            html_matches = len(html_pattern.findall(content))
            total_words = len(content.split())
            
            # Score: lower HTML ratio = higher quality
            if total_words == 0:
                quality_scores.append(0.0)
            else:
                html_word_estimate = html_matches * 3  # rough estimate
                clean_words = max(0, total_words - html_word_estimate)
                quality = clean_words / total_words
                quality_scores.append(min(1.0, quality))
        
        return {
            "mean_quality": np.mean(quality_scores),
            "median_quality": np.median(quality_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
            "std_quality": np.std(quality_scores),
            "percent_high_quality": np.mean([1 if q > 0.7 else 0 for q in quality_scores]) * 100,
            "percent_low_quality": np.mean([1 if q < 0.4 else 0 for q in quality_scores]) * 100,
        }
    
    @staticmethod
    def detect_overlap(chunks: List[Dict], top_n: int | None = None) -> Dict:
        """Detect overlap between consecutive chunks."""
        overlaps = []
        
        # Check all consecutive pairs unless a limit is provided
        if top_n is None:
            check_count = max(0, len(chunks) - 1)
        else:
            check_count = min(top_n, len(chunks) - 1)
        
        for i in range(check_count):
            chunk1 = chunks[i].get('content', '').lower()
            chunk2 = chunks[i + 1].get('content', '').lower()
            
            if not chunk1 or not chunk2:
                overlaps.append(0.0)
                continue
            
            # Get unique words from each chunk
            words1 = set(re.findall(r'\b\w+\b', chunk1))
            words2 = set(re.findall(r'\b\w+\b', chunk2))
            
            if len(words1) == 0 or len(words2) == 0:
                overlaps.append(0.0)
                continue
            
            # Jaccard similarity
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            overlap = intersection / union if union > 0 else 0.0
            overlaps.append(overlap)
        
        return {
            "mean_overlap": np.mean(overlaps) if overlaps else 0.0,
            "median_overlap": np.median(overlaps) if overlaps else 0.0,
            "min_overlap": min(overlaps) if overlaps else 0.0,
            "max_overlap": max(overlaps) if overlaps else 0.0,
            "std_overlap": np.std(overlaps) if overlaps else 0.0,
            "chunks_analyzed": len(overlaps),
        }
    
    @staticmethod
    def identify_issues(chunks: List[Dict]) -> List[Dict]:
        """Identify specific quality issues."""
        issues = []
        
        for idx, chunk in enumerate(chunks):
            content = chunk.get('content', '')
            
            # Issue 1: Empty or very small chunks
            if len(content) < 50:
                issues.append({
                    "chunk_idx": idx,
                    "type": "TOO_SMALL",
                    "severity": "WARNING",
                    "message": f"Chunk {idx} is very small ({len(content)} chars)",
                    "content_preview": content[:100]
                })
            
            # Issue 2: Very large chunks
            if len(content) > 5000:
                issues.append({
                    "chunk_idx": idx,
                    "type": "TOO_LARGE",
                    "severity": "INFO",
                    "message": f"Chunk {idx} is very large ({len(content)} chars)",
                    "content_preview": content[:100] + "..."
                })
            
            # Issue 3: Mostly HTML/markup
            html_count = len(re.findall(r'<[^>]+>|\[\[.*?\]\]', content))
            text_count = len(content.split())
            if text_count > 0 and html_count > text_count * 0.5:
                issues.append({
                    "chunk_idx": idx,
                    "type": "HIGH_MARKUP",
                    "severity": "WARNING",
                    "message": f"Chunk {idx} is {(html_count/text_count)*100:.0f}% markup",
                    "content_preview": content[:100]
                })
            
            # Issue 4: Possible mid-sentence split (starts with lowercase)
            if content and content.strip() and content.strip()[0].islower():
                issues.append({
                    "chunk_idx": idx,
                    "type": "MID_SENTENCE_START",
                    "severity": "INFO",
                    "message": f"Chunk {idx} starts with lowercase (possible mid-sentence split)",
                    "content_preview": content[:100]
                })
        
        return issues


def run_analysis():
    """Run complete chunk quality analysis."""
    
    print("\n" + "="*80)
    print("CHUNK QUALITY EVALUATION")
    print("="*80)
    
    # Load chunks
    project_root = Path(__file__).parent.parent
    chunks_file = project_root / "chunks.json"
    
    print(f"\nLoading chunks from: {chunks_file}")
    
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"✓ Loaded {len(chunks)} chunks")
    except Exception as e:
        print(f"✗ Error loading chunks: {e}")
        return
    
    analyzer = ChunkQualityAnalyzer()
    
    # 1. Size Analysis
    print("\n" + "-"*80)
    print("1. CHUNK SIZE ANALYSIS")
    print("-"*80)
    
    size_stats = analyzer.analyze_chunk_sizes(chunks)
    
    print("\nWord Count:")
    print(f"  Min:    {size_stats['word_count']['min']:>6} words")
    print(f"  Q25:    {size_stats['word_count']['q25']:>6.0f} words")
    print(f"  Mean:   {size_stats['word_count']['mean']:>6.0f} words  (Target: 200-500)")
    print(f"  Median: {size_stats['word_count']['median']:>6.0f} words")
    print(f"  Q75:    {size_stats['word_count']['q75']:>6.0f} words")
    print(f"  Max:    {size_stats['word_count']['max']:>6} words")
    print(f"  StdDev: {size_stats['word_count']['std']:>6.0f} (Low=consistent, High=variable)")
    
    print("\nToken Estimate (1 token ≈ 0.75 words):")
    print(f"  Min:    {size_stats['token_estimate']['min']:>6} tokens")
    print(f"  Mean:   {size_stats['token_estimate']['mean']:>6.0f} tokens  (Target: 150-375)")
    print(f"  Max:    {size_stats['token_estimate']['max']:>6} tokens")
    
    print("\nCharacter Count:")
    print(f"  Min:    {size_stats['char_count']['min']:>6} chars")
    print(f"  Mean:   {size_stats['char_count']['mean']:>6.0f} chars")
    print(f"  Max:    {size_stats['char_count']['max']:>6} chars")
    
    # Assessment
    avg_tokens = size_stats['token_estimate']['mean']
    if 150 <= avg_tokens <= 375:
        print(f"\n✓ GOOD: Average chunk size is {avg_tokens:.0f} tokens (within target range)")
    elif avg_tokens < 150:
        print(f"\n⚠ WARNING: Average chunk size is {avg_tokens:.0f} tokens (TOO SMALL, target: 150-375)")
    else:
        print(f"\n⚠ WARNING: Average chunk size is {avg_tokens:.0f} tokens (TOO LARGE, target: 150-375)")
    
    std = size_stats['word_count']['std']
    if std < size_stats['word_count']['mean'] * 0.5:
        print(f"✓ GOOD: Chunk sizes are consistent (StdDev: {std:.0f})")
    else:
        print(f"⚠ WARNING: Chunk sizes vary widely (StdDev: {std:.0f})")
    
    # 2. Content Quality
    print("\n" + "-"*80)
    print("2. CONTENT QUALITY ANALYSIS")
    print("-"*80)
    
    quality_stats = analyzer.detect_content_quality(chunks)
    
    print(f"\nContent Cleanliness (signal vs noise):")
    print(f"  Mean:           {quality_stats['mean_quality']:.2%}  (Target: > 70%)")
    print(f"  High quality:   {quality_stats['percent_high_quality']:.1f}% of chunks  (Target: > 80%)")
    print(f"  Low quality:    {quality_stats['percent_low_quality']:.1f}% of chunks  (Should be < 20%)")
    print(f"  Range:          {quality_stats['min_quality']:.2%} - {quality_stats['max_quality']:.2%}")
    
    if quality_stats['mean_quality'] > 0.7:
        print(f"\n✓ GOOD: Content quality is high ({quality_stats['mean_quality']:.2%})")
    else:
        print(f"\n⚠ WARNING: Content quality is low ({quality_stats['mean_quality']:.2%}), too much markup/HTML")
    
    # 3. Overlap Analysis
    print("\n" + "-"*80)
    print("3. OVERLAP ANALYSIS (All consecutive chunk pairs)")
    print("-"*80)
    
    overlap_stats = analyzer.detect_overlap(chunks, top_n=None)
    
    print(f"\nOverlap between consecutive chunks:")
    print(f"  Mean:           {overlap_stats['mean_overlap']:.2%}  (Target: 10-20%)")
    print(f"  Median:         {overlap_stats['median_overlap']:.2%}")
    print(f"  Range:          {overlap_stats['min_overlap']:.2%} - {overlap_stats['max_overlap']:.2%}")
    print(f"  Chunks checked: {overlap_stats['chunks_analyzed']}")
    
    if 0.10 <= overlap_stats['mean_overlap'] <= 0.20:
        print(f"\n✓ GOOD: Overlap is in target range ({overlap_stats['mean_overlap']:.2%})")
    elif overlap_stats['mean_overlap'] < 0.10:
        print(f"\n⚠ WARNING: Overlap is low ({overlap_stats['mean_overlap']:.2%}), may lose context")
    else:
        print(f"\n⚠ WARNING: Overlap is high ({overlap_stats['mean_overlap']:.2%}), may be redundant")
    
    # 4. Issues
    print("\n" + "-"*80)
    print("4. IDENTIFIED ISSUES")
    print("-"*80)
    
    issues = analyzer.identify_issues(chunks)
    
    if not issues:
        print("\n✓ No major issues detected!")
    else:
        # Group by severity
        critical = [i for i in issues if i['severity'] == 'CRITICAL']
        warnings = [i for i in issues if i['severity'] == 'WARNING']
        infos = [i for i in issues if i['severity'] == 'INFO']
        
        if critical:
            print(f"\n🔴 CRITICAL ({len(critical)} issues):")
            for issue in critical[:5]:  # Show first 5
                print(f"  - Chunk {issue['chunk_idx']}: {issue['message']}")
        
        if warnings:
            print(f"\n🟡 WARNINGS ({len(warnings)} issues):")
            for issue in warnings[:5]:  # Show first 5
                print(f"  - Chunk {issue['chunk_idx']}: {issue['message']}")
        
        if infos:
            print(f"\n🔵 INFO ({len(infos)} issues):")
            for issue in infos[:5]:  # Show first 5
                print(f"  - Chunk {issue['chunk_idx']}: {issue['message']}")
        
        if len(issues) > 10:
            print(f"\n... and {len(issues) - 10} more issues (see full report for details)")
    
    # 5. Summary & Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    if avg_tokens < 150:
        recommendations.append("• Increase chunk size (too small) - consider merging adjacent chunks")
    elif avg_tokens > 375:
        recommendations.append("• Decrease chunk size (too large) - consider splitting chunks")
    
    if std > size_stats['word_count']['mean'] * 0.5:
        recommendations.append("• Standardize chunk sizes - aim for consistency")
    
    if quality_stats['mean_quality'] < 0.7:
        recommendations.append("• Clean up chunks - remove excessive HTML/markup")
    
    if overlap_stats['mean_overlap'] < 0.10:
        recommendations.append("• Increase overlap - add context at chunk boundaries")
    elif overlap_stats['mean_overlap'] > 0.20:
        recommendations.append("• Reduce overlap - consider re-chunking to avoid redundancy")
    
    if len(issues) > len(chunks) * 0.1:
        recommendations.append("• Fix identified issues - too many problematic chunks")
    
    if recommendations:
        print()
        for rec in recommendations:
            print(rec)
    else:
        print("\n✓ All metrics look good! Chunks appear to be well-formed.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    run_analysis()
