"""
Core RAG functionality - search and answer generation.
Refactored from app.py and rag.py for shared use.

SYSTEM CONTEXT:
- WHERE: Imported by app.py (web API) and rag.py (CLI). Central module.
- WHEN: Loaded at application startup. Lazy-loads heavy models (reranker, BM25, ChromaDB) on first use.
- WHAT: Handles all retrieval (FAISS/ChromaDB/BM25), reranking, answer generation,
  caching, health checks, and logging. Contains the shared business logic.
"""

import json
import os
import pickle
import time
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from groq import Groq
from dotenv import load_dotenv
from loguru import logger
from cachetools import TTLCache

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.0"))
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
ENABLE_BM25 = os.getenv("ENABLE_BM25", "false").lower() == "true"
USE_CHROMA = os.getenv("USE_CHROMA", "false").lower() == "true"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
FAISS_INDEX_FILE = "faiss_index_cosine.bin"
FAISS_TEXTS_FILE = "faiss_texts_cosine.pkl"

# Initialize cache (skip queries with pronouns)
response_cache = TTLCache(maxsize=200, ttl=3600)
PRONOUNS = ["my", "our", "your", "his", "her", "their", "its"]

# ─────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────
# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
logger.add("logs/rag.log", rotation="1 day", retention="7 days", level="INFO")

# ─────────────────────────────────────────
# AI CLIENTS
# ─────────────────────────────────────────
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────
# LOAD CHUNKS & METADATA
# ─────────────────────────────────────────
with open("chunks_enterprise.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract metadata
KB_METADATA = data.get("meta", {})
KB_VERSION = KB_METADATA.get("version", "unknown")
KB_LAST_UPDATED = KB_METADATA.get("generated_at", "unknown")
KB_TOTAL_PARENTS = KB_METADATA.get("total_parent_chunks", 0)
KB_TOTAL_CHILDREN = KB_METADATA.get("total_child_chunks", 0)

# Log KB metadata on startup
logger.info("=" * 50)
logger.info("KB METADATA")
logger.info("=" * 50)
logger.info(f"Version: {KB_VERSION}")
logger.info(f"Last Updated: {KB_LAST_UPDATED}")
logger.info(f"Total Parents: {KB_TOTAL_PARENTS}")
logger.info(f"Total Children: {KB_TOTAL_CHILDREN}")
logger.info("=" * 50)

# Also print to console
print(f"\n[KB METADATA] Version: {KB_VERSION} | Last Updated: {KB_LAST_UPDATED}")
print(f"[KB METADATA] Total Parents: {KB_TOTAL_PARENTS} | Total Children: {KB_TOTAL_CHILDREN}\n")

children = [c for c in data.get("children", []) if not c.get("deprecated", False)]
parents = {p["chunk_id"]: p for p in data.get("parents", [])}

child_texts = [c.get("content", "") for c in children]
child_sources = [c.get("source_url", "") for c in children]
child_titles = [c.get("title", "") for c in children]
parent_ids = [c.get("parent_chunk_id", "") for c in children]

print(f"[LOAD] {len(children)} child chunks | {len(parents)} parent chunks")

# ─────────────────────────────────────────
# EMBEDDING MODEL & FAISS INDEX
# ─────────────────────────────────────────
model = SentenceTransformer(EMBED_MODEL_NAME)

# Reranker (cross-encoder) - lazy loaded
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None and ENABLE_RERANKER:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Reranker loaded: cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

# BM25 index - lazy loaded
_bm25 = None
_tokenized_texts = None

# ChromaDB - lazy loaded
_chroma_collection = None

def get_chroma_collection():
    """Initialize ChromaDB collection as FAISS alternative."""
    global _chroma_collection
    if _chroma_collection is None and USE_CHROMA:
        import chromadb
        from chromadb.config import Settings
        
        chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="chroma_db"
        ))
        
        # Create or get collection
        collection_name = "hushly_kb"
        try:
            _chroma_collection = chroma_client.get_collection(name=collection_name)
            logger.info(f"ChromaDB collection loaded: {collection_name}")
        except:
            # Build collection from chunks
            _chroma_collection = chroma_client.create_collection(name=collection_name)
            
            # Add documents in batches
            batch_size = 100
            for i in range(0, len(children), batch_size):
                batch = children[i:i+batch_size]
                ids = [c["chunk_id"] for c in batch]
                texts = [c.get("content", "") for c in batch]
                metadatas = [{
                    "title": c.get("title", ""),
                    "source": c.get("source_url", ""),
                    "parent_id": c.get("parent_chunk_id", "")
                } for c in batch]
                
                # Generate embeddings
                embeddings = model.encode(texts).tolist()
                
                _chroma_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=texts
                )
            
            logger.info(f"ChromaDB collection built: {collection_name} with {len(children)} documents")
    
    return _chroma_collection

def chroma_search(query: str, k: int = 15) -> List[Tuple[int, float]]:
    """
    Search using ChromaDB as FAISS alternative.
    Returns list of (child_index, score) tuples.
    """
    collection = get_chroma_collection()
    if collection is None:
        return []
    
    # Encode query
    query_embedding = model.encode([query]).tolist()
    
    # Search
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    
    # Map ChromaDB results to indices
    chroma_ids = results["ids"][0]
    distances = results["distances"][0]
    
    # Create ID to index mapping
    child_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(children)}
    
    results_list = []
    for doc_id, dist in zip(chroma_ids, distances):
        idx = child_id_to_idx.get(doc_id, -1)
        if idx >= 0:
            # Convert distance to similarity score (ChromaDB uses cosine distance)
            score = 1.0 - float(dist)
            results_list.append((idx, score))
    
    return results_list

def get_bm25():
    global _bm25, _tokenized_texts
    if _bm25 is None and ENABLE_BM25:
        from rank_bm25 import BM25Okapi
        _tokenized_texts = [text.lower().split() for text in child_texts]
        _bm25 = BM25Okapi(_tokenized_texts)
        logger.info("BM25 index built")
    return _bm25

def build_faiss_index():
    """Build FAISS index with cosine similarity (normalized vectors + Inner Product)."""
    print(f"[FAISS] Building embeddings with cosine similarity — this runs ONCE then saves...")
    embeddings = model.encode(child_texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Use Inner Product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_TEXTS_FILE, "wb") as f:
        pickle.dump({"count": len(child_texts)}, f)
    
    print(f"[FAISS] Built and saved — {index.ntotal} vectors (cosine similarity) ✅")
    return index

# Load or build FAISS index
if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_TEXTS_FILE):
    print("[FAISS] Loading saved cosine index from disk — fast restart ✅")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(FAISS_TEXTS_FILE, "rb") as f:
        saved_meta = pickle.load(f)
    if saved_meta.get("count") != len(child_texts):
        print("[FAISS] Chunk count changed — rebuilding index...")
        os.remove(FAISS_INDEX_FILE)
        os.remove(FAISS_TEXTS_FILE)
        index = build_faiss_index()
    else:
        print(f"[FAISS] Index loaded — {index.ntotal} vectors ready (cosine similarity)")
else:
    index = build_faiss_index()

# ─────────────────────────────────────────
# SEARCH FUNCTIONS
# ─────────────────────────────────────────

def faiss_search(query: str, k: int = 15) -> List[Tuple[int, float]]:
    """
    Perform FAISS search with cosine similarity.
    Returns list of (child_index, score) tuples.
    """
    query_vec = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)  # Normalize for cosine similarity
    
    scores, indices = index.search(query_vec, k)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if 0 <= idx < len(children):
            results.append((int(idx), float(score)))
    
    return results

def bm25_search(query: str, k: int = 15) -> List[Tuple[int, float]]:
    """
    Perform BM25 search.
    Returns list of (child_index, score) tuples.
    """
    if not ENABLE_BM25:
        return []
    
    bm25 = get_bm25()
    if bm25 is None:
        return []
    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Get top k indices
    top_indices = np.argsort(scores)[-k:][::-1]
    results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    return results

def merge_results(faiss_results: List[Tuple[int, float]], 
                  bm25_results: List[Tuple[int, float]], 
                  k: int = 15) -> List[int]:
    """
    Merge FAISS and BM25 results, deduplicate.
    Uses Reciprocal Rank Fusion (RRF) for scoring.
    """
    if not ENABLE_BM25 or not bm25_results:
        return [idx for idx, _ in faiss_results[:k]]
    
    # Reciprocal Rank Fusion
    rrf_scores = {}
    k_rrf = 60  # RRF constant
    
    # Add FAISS scores (rank-based)
    for rank, (idx, _) in enumerate(faiss_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k_rrf + rank + 1)
    
    # Add BM25 scores (rank-based)
    for rank, (idx, _) in enumerate(bm25_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k_rrf + rank + 1)
    
    # Sort by RRF score
    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return sorted_indices[:k]

def rerank_results(query: str, results: List[Dict]) -> List[Dict]:
    """
    Rerank results using cross-encoder.
    Returns top 5 after reranking.
    """
    if not ENABLE_RERANKER or len(results) <= 1:
        return results
    
    reranker = get_reranker()
    if reranker is None:
        return results
    
    # Prepare pairs for reranking
    pairs = [(query, r["match"]) for r in results]
    scores = reranker.predict(pairs)
    
    # Add scores to results and sort
    for r, score in zip(results, scores):
        r["rerank_score"] = float(score)
    
    results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    
    # Keep top 5 after reranking
    return results[:5]

def search(query: str, k: int = 15) -> List[Dict]:
    """
    Main search function combining FAISS/ChromaDB, BM25, and reranking.
    Applies similarity threshold filtering.
    Returns list of result dicts with context, match, source, title.
    """
    # Route to ChromaDB or FAISS based on flag
    if USE_CHROMA:
        vector_results = chroma_search(query, k)
        logger.debug(f"ChromaDB search: {len(vector_results)} results")
    else:
        vector_results = faiss_search(query, k)
        logger.debug(f"FAISS search: {len(vector_results)} results")
    
    # Get BM25 results if enabled
    bm25_results = []
    if ENABLE_BM25 and not USE_CHROMA:  # BM25 hybrid only with FAISS
        bm25_results = bm25_search(query, k)
    
    # Merge results (use FAISS-style merging for both)
    merged_indices = merge_results(vector_results, bm25_results, k)
    
    # Build result objects
    results = []
    seen_parent_ids = set()
    
    for idx in merged_indices:
        if idx < 0 or idx >= len(children):
            continue
        
        child = children[idx]
        p_id = child.get("parent_chunk_id", "")
        
        # Get vector search score for this result
        vector_score = next((score for i, score in vector_results if i == idx), 0.0)
        
        # Apply similarity threshold
        if vector_score < SIMILARITY_THRESHOLD:
            continue
        
        # Get parent chunk for full context
        parent = parents.get(p_id)
        context_text = parent["content"] if parent else child["content"]
        source = child.get("source_url", "")
        title = child.get("title", "Hushly Docs")
        
        # Deduplicate by parent
        dedup_key = p_id or child["chunk_id"]
        if dedup_key in seen_parent_ids:
            continue
        seen_parent_ids.add(dedup_key)
        
        results.append({
            "context": context_text,
            "match": child["content"],
            "source": source,
            "title": title,
            "faiss_score": vector_score,
        })
    
    # Rerank if enabled
    if ENABLE_RERANKER and results:
        results = rerank_results(query, results)
    
    return results

# ─────────────────────────────────────────
# ANSWER GENERATION
# ─────────────────────────────────────────

def generate_answer(query: str, results: List[Dict]) -> Tuple[str, str]:
    """
    Generate answer using LLM with retrieved context.
    Returns (answer_text, model_used).
    """
    # Build context from results
    context_blocks = []
    for r in results:
        block = f"[{r['title']}]\n{r['context']}"
        context_blocks.append(block)
    context = "\n\n---\n\n".join(context_blocks)
    
    # Trim context to ~3000 tokens max
    context = context[:12000]
    
    # Read prompt template from file
    with open("prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt = prompt_template.replace("{context}", context).replace("{query}", query)
    
    # Try Gemini first, fall back to Groq
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text, "gemini"
    except Exception as e:
        logger.warning(f"Gemini failed: {e} — falling back to Groq")
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content, "groq"

def check_llm_health() -> Dict[str, str]:
    """
    Check health of LLM providers.
    Returns dict with status for each provider.
    """
    import concurrent.futures
    
    results = {"gemini": "unknown", "groq": "unknown"}
    
    def check_gemini():
        try:
            res = gemini_client.models.generate_content(
                model="gemini-2.0-flash", 
                contents="Say OK",
                config={"max_output_tokens": 10}
            )
            return "ok" if "OK" in res.text or "ok" in res.text else "degraded"
        except Exception as e:
            logger.warning(f"Gemini health check failed: {e}")
            return "down"
    
    def check_groq():
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=10
            )
            msg = completion.choices[0].message.content
            return "ok" if "OK" in msg or "ok" in msg else "degraded"
        except Exception as e:
            logger.warning(f"Groq health check failed: {e}")
            return "down"
    
    # Run health checks with timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        gemini_future = executor.submit(check_gemini)
        groq_future = executor.submit(check_groq)
        
        try:
            results["gemini"] = gemini_future.result(timeout=3)
        except concurrent.futures.TimeoutError:
            results["gemini"] = "timeout"
        except Exception as e:
            results["gemini"] = "down"
            logger.error(f"Gemini health check error: {e}")
        
        try:
            results["groq"] = groq_future.result(timeout=3)
        except concurrent.futures.TimeoutError:
            results["groq"] = "timeout"
        except Exception as e:
            results["groq"] = "down"
            logger.error(f"Groq health check error: {e}")
    
    return results

# ─────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────

def should_use_cache(query: str) -> bool:
    """Check if query should be cached (no pronouns indicating personalization)."""
    query_lower = query.lower()
    return not any(p in query_lower for p in PRONOUNS)

def get_cache_key(query: str) -> str:
    """Generate cache key from normalized query."""
    return query.lower().strip()

def log_request(query: str, rewritten_query: str, results: List[Dict], 
                model_used: str, latency_ms: float, answer_length: int):
    """Log request details."""
    top_3 = [(r.get("title", ""), r.get("faiss_score", 0)) for r in results[:3]]
    logger.info(f"Request: query='{query[:50]}...', rewritten='{rewritten_query[:50]}...', "
                f"top_3={top_3}, model={model_used}, latency={latency_ms:.0f}ms, "
                f"answer_len={answer_length}")
