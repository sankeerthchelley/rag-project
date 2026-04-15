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
from groq import Groq
from openai import OpenAI
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
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

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
try:
    model = SentenceTransformer(EMBED_MODEL_NAME)
except Exception as e:
    print(f"[MODEL] ERROR loading embedding model: {e}")
    model = None

# Reranker (cross-encoder) - initialize at startup if enabled
_reranker = None
if ENABLE_RERANKER:
    from sentence_transformers import CrossEncoder
    _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    logger.info("Reranker loaded: cross-encoder/ms-marco-MiniLM-L-6-v2")

def get_reranker():
    return _reranker

# BM25 index - initialize at startup if enabled
_bm25 = None
_tokenized_texts = None
if ENABLE_BM25:
    from rank_bm25 import BM25Okapi
    _tokenized_texts = [text.lower().split() for text in child_texts]
    _bm25 = BM25Okapi(_tokenized_texts)
    logger.info("BM25 index built")

# ChromaDB - initialize at startup if enabled
_chroma_collection = None
if USE_CHROMA:
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

def get_chroma_collection():
    """Return initialized ChromaDB collection."""
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
            # SCALE: Convert ChromaDB cosine distance [0, 2] to similarity [0, 1]
            # where 1.0 = identical vectors, 0.0 = opposite vectors
            score = 1.0 - float(dist)
            results_list.append((idx, score))
    
    return results_list

def get_bm25():
    """Return initialized BM25 index."""
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
    
    print(f"[FAISS] Built and saved — {index.ntotal} vectors (cosine similarity) [OK]")
    return index

# Load or build FAISS index
try:
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_TEXTS_FILE):
        print("[FAISS] Loading saved cosine index from disk — fast restart [OK]")
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
except Exception as e:
    print(f"[FAISS] ERROR during index load/build: {e}")
    index = None

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
    
    # Similarity Threshold Guard: If the best match is too weak, reject early
    # (scores[0][0] is the similarity of the top match on [0, 1] scale)
    if len(scores[0]) == 0 or scores[0][0] < SIMILARITY_THRESHOLD:
        return []
    
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
    
    NOTE: faiss_results scores are [0, 1] similarity (1.0 = best, 0.0 = worst)
          BM25 scores are arbitrary positive values, so we use rank-based fusion
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

def search(query: str, k: int = 15, use_reranker: bool = None, use_bm25: bool = None) -> List[Dict]:
    """
    Main search function combining FAISS/ChromaDB, BM25, and reranking.
    Applies similarity threshold filtering.
    Returns list of result dicts with context, match, source, title.
    
    Args:
        query: Search query string
        k: Number of results to retrieve
        use_reranker: Override global ENABLE_RERANKER flag (None = use global)
        use_bm25: Override global ENABLE_BM25 flag (None = use global)
    """
    # Determine effective flags (override > global)
    effective_reranker = ENABLE_RERANKER if use_reranker is None else use_reranker
    effective_bm25 = ENABLE_BM25 if use_bm25 is None else use_bm25
    
    # Route to ChromaDB or FAISS based on flag
    if USE_CHROMA:
        vector_results = chroma_search(query, k)
        logger.debug(f"ChromaDB search: {len(vector_results)} results")
    else:
        vector_results = faiss_search(query, k)
        logger.debug(f"FAISS search: {len(vector_results)} results")
    
    # Get BM25 results if enabled
    bm25_results = []
    if effective_bm25 and not USE_CHROMA:  # BM25 hybrid only with FAISS
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
        # SCALE: Both FAISS and ChromaDB return [0, 1] similarity where 1.0 = best match
        vector_score = next((score for i, score in vector_results if i == idx), 0.0)
        
        # Apply similarity threshold (reject if below threshold)
        # Threshold is on [0, 1] scale; higher threshold = stricter matching
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
    
    # Rerank if enabled (use effective flag)
    if effective_reranker and results:
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
    
    # Fallback Chain: OpenRouter -> Groq
    # Try OpenRouter first (Primary)
    try:
        or_model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
        completion = openrouter_client.chat.completions.create(
            model=or_model,
            messages=[{"role": "user", "content": prompt}],
            extra_headers={
                "HTTP-Referer": "https://hushly.com", # Optional, for OpenRouter rankings
                "X-Title": "Hushly RAG Assistant",    # Optional
            }
        )
        return completion.choices[0].message.content, "openrouter"
    except Exception as e:
        logger.warning(f"OpenRouter failed: {e} — falling back to Groq")

    # Try Groq second (Fallback)
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content, "groq"
    except Exception as e:
        logger.error(f"All AI providers failed: {e}")
        return "I'm having trouble connecting to my AI providers. Please try again in a moment.", "error"

# Health check state management
_groq_last_quota_error = 0
_openrouter_last_quota_error = 0
_last_health_check_time = 0
_last_health_results = {"groq": "unknown", "openrouter": "unknown"}
GROQ_COOLDOWN_SEC = 300     # 5 minutes
OPENROUTER_COOLDOWN_SEC = 300 # 5 minutes
HEALTH_CACHE_SEC = 60       # 1 minute

def check_llm_health() -> Dict[str, str]:
    """
    Check health of LLM providers with throttling and cooldowns.
    Returns dict with status for each provider.
    """
    global _last_health_check_time, _last_health_results
    import concurrent.futures
    import time
    
    now = time.time()
    
    # Global throttling: Return cached results if called too frequently
    if now - _last_health_check_time < HEALTH_CACHE_SEC:
        return _last_health_results
    
    _last_health_check_time = now
    results = {"groq": "unknown", "openrouter": "unknown"}
    
    def check_groq():
        global _groq_last_quota_error
        if time.time() - _groq_last_quota_error < GROQ_COOLDOWN_SEC:
            return "quota_exceeded_cooldown"

        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=10
            )
            msg = completion.choices[0].message.content
            return "ok" if "OK" in msg or "ok" in msg else "degraded"
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate_limit_exceeded" in err_str or "quota" in err_str:
                _groq_last_quota_error = time.time()
                logger.warning("Groq quota exhausted (Health Check)")
                return "quota_exceeded"
            logger.warning(f"Groq health check failed: {e}")
            return "down"
    
    def check_openrouter():
        global _openrouter_last_quota_error
        if time.time() - _openrouter_last_quota_error < OPENROUTER_COOLDOWN_SEC:
            return "quota_exceeded_cooldown"
            
        try:
            or_model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
            completion = openrouter_client.chat.completions.create(
                model=or_model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=10
            )
            msg = completion.choices[0].message.content
            return "ok" if "OK" in msg or "ok" in msg else "degraded"
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str:
                _openrouter_last_quota_error = time.time()
                logger.warning("OpenRouter quota exhausted (Health Check)")
                return "quota_exceeded"
            logger.warning(f"OpenRouter health check failed: {e}")
            return "down"
    
    # Run health checks with timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        groq_future = executor.submit(check_groq)
        openrouter_future = executor.submit(check_openrouter)
        
        try:
            results["groq"] = groq_future.result(timeout=4)
        except Exception:
            results["groq"] = "timeout"
            
        try:
            results["openrouter"] = openrouter_future.result(timeout=4)
        except Exception:
            results["openrouter"] = "timeout"
    
    _last_health_results = results
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
