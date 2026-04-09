from flask import Flask, request, jsonify, send_from_directory
import json
import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from google import genai
from groq import Groq
from flask_cors import CORS
from dotenv import load_dotenv

# ─────────────────────────────────────────
# LOAD ENV (keys from .env file, not hardcoded)
# ─────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ─────────────────────────────────────────
# KEYS — now loaded from .env file safely
# ─────────────────────────────────────────
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────
# LOAD CHUNKS — children for search,
#               parents dict for full context
# ─────────────────────────────────────────
with open("chunks_enterprise.json", "r", encoding="utf-8") as f:
    data = json.load(f)

children = [c for c in data.get("children", []) if not c.get("deprecated", False)]
parents  = {p["chunk_id"]: p for p in data.get("parents", [])}

child_texts   = [c.get("content", "") for c in children]
child_sources = [c.get("source_url", "") for c in children]
child_titles  = [c.get("title", "") for c in children]
parent_ids    = [c.get("parent_chunk_id", "") for c in children]

print(f"[LOAD] {len(children)} child chunks | {len(parents)} parent chunks")

# ─────────────────────────────────────────
# EMBEDDINGS — load from disk if already
# built, otherwise build and save to disk.
# This means fast restarts after first run.
# ─────────────────────────────────────────
FAISS_INDEX_FILE  = "faiss_index.bin"
FAISS_TEXTS_FILE  = "faiss_texts.pkl"
EMBED_MODEL_NAME  = "all-MiniLM-L6-v2"

model = SentenceTransformer(EMBED_MODEL_NAME)

if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_TEXTS_FILE):
    print("[FAISS] Loading saved index from disk — fast restart ✅")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(FAISS_TEXTS_FILE, "rb") as f:
        saved_meta = pickle.load(f)
    # Verify saved index matches current chunks
    if saved_meta.get("count") != len(child_texts):
        print("[FAISS] Chunk count changed — rebuilding index...")
        os.remove(FAISS_INDEX_FILE)
        os.remove(FAISS_TEXTS_FILE)
        index = None
    else:
        print(f"[FAISS] Index loaded — {index.ntotal} vectors ready")
else:
    index = None

if index is None:
    print("[FAISS] Building embeddings — this runs ONCE then saves to disk...")
    embeddings = model.encode(child_texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_TEXTS_FILE, "wb") as f:
        pickle.dump({"count": len(child_texts)}, f)
    print(f"[FAISS] Built and saved — {index.ntotal} vectors ✅")

# ─────────────────────────────────────────
# SEARCH — finds top child chunks,
# then fetches their PARENT for full context
# ─────────────────────────────────────────
def search(query, k=15):
    query_vec = model.encode([query]).astype("float32")
    _, indices = index.search(query_vec, k)

    results = []
    seen_parent_ids = set()  # avoid duplicate parent context

    for i in indices[0]:
        if i < 0 or i >= len(children):
            continue

        child = children[i]
        p_id  = child.get("parent_chunk_id", "")

        # Get parent chunk for full context
        parent = parents.get(p_id)

        # Use parent content if available, else fall back to child
        context_text = parent["content"] if parent else child["content"]
        source       = child.get("source_url", "")
        title        = child.get("title", "Hushly Docs")

        # Deduplicate — don't add same parent twice
        dedup_key = p_id or child["chunk_id"]
        if dedup_key in seen_parent_ids:
            continue
        seen_parent_ids.add(dedup_key)

        results.append({
            "context": context_text,   # full parent = rich context for AI
            "match":   child["content"],  # exact child match = what triggered it
            "source":  source,
            "title":   title,
        })

    return results

# ─────────────────────────────────────────
# GENERATE ANSWER
# ─────────────────────────────────────────
def generate_answer(query, results):
    # Build context from full parent chunks
    context_blocks = []
    for r in results:
        block = f"[{r['title']}]\n{r['context']}"
        context_blocks.append(block)
    context = "\n\n---\n\n".join(context_blocks)

    # Trim context to ~3000 tokens max (approx 12,000 characters) to save costs/quota
    context = context[:12000]
    
    # Re-read prompt.txt on every request
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
        print(f"[WARN] Gemini failed: {e} — falling back to Groq (Llama 8B)")
        # Switching to llama-3.1-8b-instant which has MUCH higher rate limits than the 70B model
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content, "groq"

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(".", "kb.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "chunks_loaded": len(children),
        "parents_loaded": len(parents),
        "vectors_indexed": index.ntotal,
    }), 200

@app.route("/ask", methods=["POST"])
def ask():
    try:
        body = request.json
        if not body or not body.get("query"):
            return jsonify({"error": "Query is missing or empty"}), 400

        query = body["query"].strip()
        history = body.get("history", [])
        recommendations = [] # Fixed: Always initialize core variables at top

        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        # Query Reformulation: Use only the last 3 turns for context stability
        search_query = query
        if history:
            try:
                # User asked specifically for last 3 turns
                context_history = history[-3:] 
                history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in context_history])
                
                rewrite_prompt = f"""Given the conversation history below, rewrite the user's latest follow-up question into a standalone, fully contextualized question. Do not answer it, just rewrite it so it can be used for a database search keyword lookup. Output ONLY the standalone question.

Conversation History (Latest Turns):
{history_text}

Latest raw question: {query}

Standalone question:"""
                
                rewrite_comp = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": rewrite_prompt}],
                    max_tokens=100,
                    temperature=0.1
                )
                search_query = rewrite_comp.choices[0].message.content.strip().strip('"\'')
                print(f"[MEMORY REWRITE] '{query}' -> '{search_query}'")
            except Exception as e:
                print(f"[WARN] Query rewrite failed: {e}")

        results = search(search_query)
        answer, model_used = generate_answer(search_query, results)

        # Detect [NO_INFO] signal from the AI
        no_info = answer.strip().startswith("[NO_INFO]")

        if no_info:
            # Build a dynamic Freshdesk search URL using the user's query keywords
            import urllib.parse
            search_term = urllib.parse.quote_plus(query)
            return jsonify({
                "answer":     "I don't have that information in the knowledge base. Please check the links below:",
                "model_used": model_used,
                "no_info":    True,
                "search_url": f"https://hushly.freshdesk.com/support/search/solutions?term={search_term}",
                "sources":    [],
                "titles":     [],
            })

        return jsonify({
            "answer":     answer,
            "model_used": model_used,
            "no_info":    False,
            "sources":    list(dict.fromkeys(r["source"] for r in results)),
            "titles":     list(dict.fromkeys(r["title"]  for r in results)),
            "recommendations": recommendations
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/enhance", methods=["POST"])
def enhance():
    try:
        body = request.json
        if not body or not body.get("query"):
            return jsonify({"error": "Query is missing"}), 400

        raw_query = body["query"].strip()
        if not raw_query:
            return jsonify({"error": "Query cannot be empty"}), 400

        enhance_prompt = f"""You are a query enhancer for a Hushly B2B SaaS knowledge base assistant.

Your job is to take a user's raw, short, or vague question and rewrite it into a clear,
detailed, and specific question that will help a RAG search system find the best answer.

HUSHLY CONTEXT (use this to expand abbreviations and add relevant terms):
- GEO = Generative Engine Optimization (handled in Hushly via GEOSherpa)
- AEO = Answer Engine Optimization
- ABM = Account-Based Marketing
- UTM = URL tracking parameters (utm_source, utm_medium, utm_campaign)
- Experiences = personalized content pages in Hushly
- Streams = curated content sequences inside Experiences
- Hubs = grouped content collections
- Pages = landing pages in Hushly
- Personas = AI-generated buyer profile segments
- Assets = content files (PDFs, videos, eBooks) uploaded to Hushly
- CSM = Customer Success Manager

RULES:
1. Expand abbreviations using the context above
2. Add Hushly-specific context where relevant (e.g. "in Hushly", "using the Hushly platform")
3. Make the question complete and specific
4. Keep it as ONE clear question — do not split into multiple questions
5. Output ONLY the enhanced question — no explanation, no preamble, no quotes

Raw user question: {raw_query}
Enhanced question:"""

        # Use Groq for speed (Llama is fast for query rewriting)
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": enhance_prompt}],
            max_tokens=120,
            temperature=0.3,
        )
        enhanced = completion.choices[0].message.content.strip()
        # Strip any surrounding quotes the model might add
        enhanced = enhanced.strip('"\'')

        return jsonify({"enhanced": enhanced, "original": raw_query})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/generate_steps", methods=["POST"])
def generate_steps():
    try:
        body = request.json
        if not body or not body.get("answer"):
            return jsonify({"error": "Answer text is missing"}), 400

        answer_text = body["answer"].strip()
        if not answer_text:
            return jsonify({"error": "Answer cannot be empty"}), 400

        prompt = f"""You are an expert technical writer for the Hushly SaaS platform.
Read the provided knowledge base answer below. If the answer contains actionable, 
step-by-step instructions (e.g., "How to upload an asset", "How to configure SSO"), 
convert it into a strict JSON object. Make each step extremely concise (1 sentence max) 
and conversational so it can be spoken out loud by a voice assistant.

If the answer is purely informational (e.g., "What is ABM?") and does not contain 
clear chronological steps, set "is_actionable_task" to false.

Answer Text: 
{answer_text}

Output Format strictly as JSON:
{{
  "is_actionable_task": true,
  "task_title": "Upload an Asset",
  "steps": [
    "First, click on the left menu and select Content.",
    "Next, click on Assets and hit the Upload button."
  ]
}}
"""
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=400,
            temperature=0.1,
        )
        
        json_resp = json.loads(completion.choices[0].message.content)
        return jsonify(json_resp)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True, use_reloader=False)