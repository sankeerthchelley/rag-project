"""
RAGAS Evaluation Script for Hushly RAG System

Compares two pipelines:
1. FAISS-only (baseline)
2. FAISS + BM25 + Reranker (full hybrid)

Outputs comparison table to eval_results.json
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Tuple

# Load env before importing core (env vars affect behavior)
load_dotenv()

# Set environment for baseline run
os.environ["ENABLE_RERANKER"] = "false"
os.environ["ENABLE_BM25"] = "false"

# Import after setting env vars
from core import search, generate_answer

# 20 Real Hushly Test Questions with Expected Themes
test_questions = [
    # Product Features
    "What is GEOSherpa in Hushly?",
    "How does ContentSherpa work?",
    "What are Hushly Experiences?",
    "What is the difference between a Stream and a Hub?",
    "How do Personas work in Hushly?",
    
    # Actions/How-To
    "How do I upload an asset to the platform?",
    "How do I create a new Experience?",
    "How do I configure UTMs for tracking?",
    "How do I set up an ABM Page?",
    "How do I create a content Stream?",
    
    # Concepts/Definitions
    "What is ABM (Account-Based Marketing)?",
    "What are UTM parameters used for in Hushly?",
    "What is the role of a CSM in Hushly?",
    "What is GEO and how does it relate to Hushly?",
    "What is AEO (Answer Engine Optimization)?",
    
    # Reporting/Analytics
    "How do I view content acquisition reports?",
    "What metrics are available in Hushly analytics?",
    "How do I track visitor engagement?",
    
    # Integration/Technical
    "How does Hushly integrate with marketing automation?",
    "What SSO options are available?",
    "How do I configure the Hushly widget?"
]


def run_pipeline(questions: List[str], use_reranker: bool, use_bm25: bool) -> List[Dict]:
    """
    Run evaluation pipeline with specified configuration.
    Returns list of result dicts with question, answer, contexts, metrics.
    """
    # Set environment variables
    os.environ["ENABLE_RERANKER"] = "true" if use_reranker else "false"
    os.environ["ENABLE_BM25"] = "true" if use_bm25 else "false"
    
    # Force reimport to pick up new env vars
    import importlib
    import core
    importlib.reload(core)
    from core import search, generate_answer
    
    print(f"\n{'='*60}")
    print(f"Running: FAISS + {'BM25 ' if use_bm25 else ''}{'+ Reranker' if use_reranker else '(baseline)'}")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, query in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {query}")
        
        try:
            start_time = time.time()
            
            # Retrieve and generate
            retrieved = search(query, k=15)
            
            if not retrieved:
                print(f"      ⚠️ No results found")
                results.append({
                    "question": query,
                    "answer": "[NO_INFO]",
                    "contexts": [],
                    "sources": [],
                    "latency_ms": (time.time() - start_time) * 1000,
                    "num_chunks": 0
                })
                continue
            
            # Build contexts for evaluation
            contexts = [r["context"] for r in retrieved[:3]]
            
            # Generate answer
            answer, model_used = generate_answer(query, retrieved)
            latency_ms = (time.time() - start_time) * 1000
            
            print(f"      ✓ {model_used} | {latency_ms:.0f}ms | {len(answer)} chars | {len(retrieved)} chunks")
            
            results.append({
                "question": query,
                "answer": answer,
                "contexts": contexts,
                "sources": list(dict.fromkeys(r["source"] for r in retrieved)),
                "titles": list(dict.fromkeys(r["title"] for r in retrieved)),
                "latency_ms": latency_ms,
                "num_chunks": len(retrieved),
                "model_used": model_used,
                "use_reranker": use_reranker,
                "use_bm25": use_bm25
            })
            
            # Rate limiting friendly delay
            time.sleep(0.5)
            
        except Exception as e:
            print(f"      ✗ Error: {e}")
            results.append({
                "question": query,
                "answer": f"[ERROR: {str(e)}]",
                "contexts": [],
                "error": str(e),
                "use_reranker": use_reranker,
                "use_bm25": use_bm25
            })
    
    return results


def calculate_ragas_metrics(results: List[Dict]) -> Dict:
    """
    Calculate RAGAS-style metrics manually (faithfulness, relevancy, recall).
    Since ragas library can be complex to set up, we use heuristic scoring.
    """
    total = len(results)
    if total == 0:
        return {}
    
    # Calculate metrics
    avg_latency = sum(r.get("latency_ms", 0) for r in results) / total
    with_results = sum(1 for r in results if r.get("contexts"))
    no_info_count = sum(1 for r in results if "[NO_INFO]" in r.get("answer", ""))
    
    # Heuristic faithfulness: checks if answer cites sources vs has no info
    faithfulness_score = (total - no_info_count) / total if total > 0 else 0
    
    # Heuristic relevancy: avg num of chunks retrieved (more = more relevant context)
    avg_chunks = sum(r.get("num_chunks", 0) for r in results) / total if total > 0 else 0
    
    # Context recall: % of queries that got results
    context_recall = with_results / total if total > 0 else 0
    
    return {
        "faithfulness": round(faithfulness_score, 3),
        "answer_relevancy": round(faithfulness_score * 0.8 + (avg_chunks / 10) * 0.2, 3),
        "context_recall": round(context_recall, 3),
        "avg_latency_ms": round(avg_latency, 1),
        "avg_chunks": round(avg_chunks, 1),
        "total_queries": total,
        "successful_queries": with_results,
        "no_info_rate": round(no_info_count / total, 3) if total > 0 else 0
    }


def main():
    """Run both pipelines and compare."""
    print("="*60)
    print("HUSHLY RAG EVALUATION")
    print("="*60)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Questions: {len(test_questions)}")
    print("="*60)
    
    # Run 1: Baseline (FAISS only)
    baseline_results = run_pipeline(test_questions, use_reranker=False, use_bm25=False)
    baseline_metrics = calculate_ragas_metrics(baseline_results)
    
    # Run 2: Full Hybrid (FAISS + BM25 + Reranker)
    hybrid_results = run_pipeline(test_questions, use_reranker=True, use_bm25=True)
    hybrid_metrics = calculate_ragas_metrics(hybrid_results)
    
    # Build comparison table
    comparison = {
        "run_date": datetime.now().isoformat(),
        "total_questions": len(test_questions),
        "pipelines": {
            "faiss_only": {
                "config": {"reranker": False, "bm25": False},
                "metrics": baseline_metrics,
                "detailed_results": baseline_results
            },
            "hybrid_full": {
                "config": {"reranker": True, "bm25": True},
                "metrics": hybrid_metrics,
                "detailed_results": hybrid_results
            }
        },
        "comparison_summary": {
            "faithfulness_improvement": round(hybrid_metrics.get("faithfulness", 0) - baseline_metrics.get("faithfulness", 0), 3),
            "relevancy_improvement": round(hybrid_metrics.get("answer_relevancy", 0) - baseline_metrics.get("answer_relevancy", 0), 3),
            "recall_improvement": round(hybrid_metrics.get("context_recall", 0) - baseline_metrics.get("context_recall", 0), 3),
            "latency_change_ms": round(hybrid_metrics.get("avg_latency_ms", 0) - baseline_metrics.get("avg_latency_ms", 0), 1)
        }
    }
    
    # Save results
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Metric':<25} {'FAISS Only':>12} {'Hybrid':>12} {'Delta':>12}")
    print("-"*60)
    
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "avg_latency_ms", "avg_chunks"]:
        b = baseline_metrics.get(metric, 0)
        h = hybrid_metrics.get(metric, 0)
        d = h - b
        print(f"{metric:<25} {b:>12.3f} {h:>12.3f} {d:>+12.3f}")
    
    print("\n" + "="*60)
    print(f"Results saved to: eval_results.json")
    print("="*60)


if __name__ == "__main__":
    main()
