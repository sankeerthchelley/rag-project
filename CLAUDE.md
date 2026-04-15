# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Run development server
```bash
python app.py
# Serves on http://localhost:8000
```

### Run CLI (quick local testing, no web server)
```bash
python rag.py
```

### Run production server
```bash
gunicorn app:app --workers 4 --timeout 120 --bind 0.0.0.0:8000
```

### Run evaluations (compares FAISS-only vs FAISS+BM25+Reranker)
```bash
python eval_rag.py
# Outputs eval_results.json
```

### Rebuild knowledge base (after updating kb.pdf)
```bash
python script.py            # Regenerates chunks_enterprise.json from kb.pdf
# Then delete stale FAISS index so it rebuilds on next start:
rm faiss_index_cosine.bin faiss_texts_cosine.pkl
python app.py               # Triggers FAISS rebuild automatically
```

## Environment Variables

Required in `.env`:
```
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
```

Optional feature flags:
```
OPENROUTER_MODEL=openrouter/free   # Model for answer generation
ENABLE_RERANKER=false              # Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
ENABLE_BM25=false                  # Lexical hybrid search (requires FAISS, incompatible with ChromaDB)
USE_CHROMA=false                   # Use ChromaDB instead of FAISS
SIMILARITY_THRESHOLD=0.0           # Minimum cosine similarity score [0,1] to include a result
EMBED_MODEL_NAME=all-MiniLM-L6-v2  # Sentence-transformers embedding model
```

## Architecture

### Module Roles

- **`core.py`** — Central module. Imported by both `app.py` and `rag.py`. Owns: chunk loading, FAISS/ChromaDB/BM25 index initialization, `search()`, `generate_answer()`, caching, logging, LLM health checks.
- **`app.py`** — Flask API. Owns: HTTP routing, rate limiting (flask-limiter), input guardrails (injection detection, topic guard, length check), query reformulation from history, `/ask` `/enhance` `/generate_steps` `/feedback` endpoints.
- **`rag.py`** — CLI chat loop. Calls `core.search()` + `core.generate_answer()` directly. Used for local testing.
- **`eval_rag.py`** — Benchmark script. Runs 25 test questions against FAISS-only and full-hybrid pipelines, outputs RAGAS-style metrics to `eval_results.json`.
- **`script.py`** — One-time ingestion. Loads `kb.pdf`, fetches content from embedded URLs, chunks with LangChain, writes `chunks_enterprise.json`.

### Knowledge Base Structure (`chunks_enterprise.json`)

Hierarchical chunk design:
- **Children** — Small chunks (~500 tokens) used for FAISS/BM25 search.
- **Parents** — Full-context chunks sent to the LLM as answer context.
- Each child has a `parent_chunk_id` linking it to the parent. At query time, child matches are deduped by parent and replaced with parent content for the LLM prompt.

### Search Pipeline (`core.search()`)

1. **Vector search** — FAISS cosine similarity (default) or ChromaDB on child chunks.
2. **BM25 search** — Optional lexical search, FAISS-mode only.
3. **Merge** — Reciprocal Rank Fusion (RRF, k=60) when BM25 is enabled; otherwise FAISS ranking.
4. **Threshold filter** — Results below `SIMILARITY_THRESHOLD` are dropped.
5. **Parent promotion** — Each child result is replaced with its parent chunk content; deduped by parent ID.
6. **Reranking** — Optional cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores top results, keeps top 5.

### LLM Providers

Answer generation uses a fallback chain:
1. **OpenRouter** (primary) — model set by `OPENROUTER_MODEL` env var.
2. **Groq / Llama-3.1-8b-instant** (fallback) — used if OpenRouter fails.

Query reformulation (rewriting follow-up questions into standalone queries) always uses Groq `llama-3.3-70b-versatile`.

### Response Caching

`TTLCache(maxsize=200, ttl=3600)` in `core.py`. Queries containing pronouns (`my`, `our`, `your`, `his`, `her`, `their`, `its`) are never cached, as they imply personalized context.

### Input Guardrails (`app.py`)

- Max query length: 500 chars.
- Prompt injection blocklist (regex): patterns like "ignore previous instructions", "act as", "jailbreak".
- Topic guard: queries not containing any Hushly-related keywords return `{"no_info": true, "no_info_reason": "off_topic"}` without hitting the LLM.

### Prompt Template (`prompt.txt`)

Uses `{context}` and `{query}` placeholders. The prompt instructs the LLM to answer only from KB context and emit `[NO_INFO]` when information is absent. The `[NO_INFO]` token is detected in `app.py` and replaced with a Freshdesk search URL in the API response.

### FAISS Index

- Built automatically on first run if `faiss_index_cosine.bin` and `faiss_texts_cosine.pkl` are absent.
- Auto-rebuilt if the chunk count in the PKL file does not match the current `chunks_enterprise.json`.
- Uses `IndexFlatIP` on L2-normalized vectors (equivalent to cosine similarity).
