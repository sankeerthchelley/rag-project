# Hushly RAG Knowledge Base System

A production-ready Retrieval-Augmented Generation (RAG) system for the Hushly B2B SaaS platform, featuring hybrid search (FAISS + BM25), cross-encoder reranking, response caching, and comprehensive evaluation tools.

## Project Overview

This system provides an AI-powered knowledge base assistant that:
- Retrieves relevant documentation chunks using semantic (FAISS cosine similarity) and lexical (BM25) search
- Reranks results using a cross-encoder for optimal relevance
- Generates answers using Gemini (primary) or Groq/Llama (fallback)
- Caches responses for frequently asked questions
- Logs all requests for monitoring and evaluation

## File Structure

```
.
├── app.py                  # Flask API server with rate limiting and guardrails
├── core.py                 # Core RAG logic (search, reranking, answer generation)
├── rag.py                  # CLI interface for testing (uses core.py)
├── eval_rag.py             # RAGAS evaluation script
├── chunks_enterprise.json  # Knowledge base with hierarchical chunks
├── prompt.txt              # LLM prompt template
├── requirements.txt        # Python dependencies
├── Procfile               # Heroku deployment config
├── .env.example           # Environment variables template
├── logs/                  # Runtime logs and feedback
│   ├── rag.log           # Application logs (rotated daily)
│   └── feedback.jsonl    # User feedback for evaluation
└── faiss_index_cosine.*   # FAISS index files (cosine similarity)
```

## Environment Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Fill in your API keys:
   ```
   GEMINI_API_KEY=your_gemini_key
   GROQ_API_KEY=your_groq_key
   ```

3. Optional feature flags:
   ```
   ENABLE_RERANKER=false      # Enable cross-encoder reranking
   ENABLE_BM25=false          # Enable BM25 hybrid search
   USE_CHROMA=false           # Use ChromaDB instead of FAISS
   SIMILARITY_THRESHOLD=0.0   # Minimum similarity score for results
   ```

## Building the Index

The FAISS index is built automatically on first run using cosine similarity (normalized L2 + Inner Product):

```bash
# Build via CLI
python rag.py

# Or start the API server (will build if needed)
python app.py
```

Index files created:
- `faiss_index_cosine.bin` - FAISS index with normalized vectors
- `faiss_texts_cosine.pkl` - Metadata for validation

## Running the Application

### Development (localhost)
```bash
python app.py
# Server runs on http://localhost:8000
```

### Production (Heroku)
```bash
# Deploy using Procfile
heroku create your-app-name
git push heroku main
```

### Production (Gunicorn locally)
```bash
gunicorn app:app --workers 4 --timeout 120 --bind 0.0.0.0:8000
```

## API Endpoints

| Endpoint | Method | Rate Limit | Description |
|----------|--------|------------|-------------|
| `/` | GET | - | Serve kb.html UI |
| `/health` | GET | - | Health check with KB metadata & LLM status |
| `/ask` | POST | 30/min | Main RAG query endpoint |
| `/enhance` | POST | 10/min | Query enhancement/rewriting |
| `/generate_steps` | POST | 10/min | Extract steps from answer |
| `/feedback` | POST | 60/min | Submit feedback for evaluation |

### Example: /ask Request
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I upload an asset?", "history": []}'
```

Response:
```json
{
  "answer": "To upload an asset...",
  "model_used": "gemini",
  "no_info": false,
  "sources": ["https://hushly.freshdesk.com/..."],
  "titles": ["Asset Upload Guide"],
  "cached": false
}
```

## Running Evaluations

The evaluation system uses RAGAS metrics (faithfulness, answer relevancy, context recall).

### Quick Evaluation
```bash
python eval_rag.py
```

This will:
1. Run 20 test questions against the RAG pipeline
2. Test both FAISS-only and FAISS+reranker+BM25 configurations
3. Output comparison table to `eval_results.json`

### Evaluation with Different Configurations
```bash
# FAISS only (baseline)
ENABLE_RERANKER=false ENABLE_BM25=false python eval_rag.py

# With reranker
ENABLE_RERANKER=true ENABLE_BM25=false python eval_rag.py

# Full hybrid (FAISS + BM25 + reranker)
ENABLE_RERANKER=true ENABLE_BM25=true python eval_rag.py
```

## Features

### Hybrid Search (Task 8)
Combines FAISS (semantic) and BM25 (lexical) using Reciprocal Rank Fusion:
- Set `ENABLE_BM25=true` in `.env`

### Cross-Encoder Reranker (Task 7)
Reranks top 15 results, keeps top 5 using `ms-marco-MiniLM-L-6-v2`:
- Set `ENABLE_RERANKER=true` in `.env`

### Response Caching (Task 11)
TTLCache (200 entries, 1 hour TTL) with pronoun detection:
- Queries with "my/our/your" skip caching (personalized)
- Other queries cached by normalized lowercase key

### Input Guardrails (Task 9)
- Max 500 character queries
- Prompt injection blocklist: ["ignore previous", "forget instructions", "you are now", "act as"]

### Rate Limiting (Task 10)
- `/ask`: 30 per minute
- `/enhance`, `/generate_steps`: 10 per minute
- `/feedback`: 60 per minute

### Logging (Task 2)
- Application logs: `logs/rag.log` (daily rotation)
- Request logs: query, rewritten query, top 3 chunk scores, LLM used, latency, answer length

## ChromaDB Alternative (Task 16)

To use ChromaDB instead of FAISS:
1. Set `USE_CHROMA=true` in `.env`
2. The system will build/load a ChromaDB collection from `chunks_enterprise.json`
3. Run evaluations to compare FAISS vs ChromaDB performance

## Development Notes

- Old functions in `rag.py` are commented for 48hr grace period before deletion
- KB metadata (version, last_updated, totals) is logged on startup
- LLM health checks run in `/health` endpoint with 3s timeout
- Feedback is stored in `logs/feedback.jsonl` for ground truth evaluation
