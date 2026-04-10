from flask import Flask, request, jsonify, send_from_directory
import json
import os
import time
import re
import urllib.parse
from datetime import datetime
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# Import from refactored core.py
from core import (
    search, 
    generate_answer, 
    check_llm_health,
    response_cache,
    should_use_cache,
    get_cache_key,
    log_request,
    KB_VERSION,
    KB_LAST_UPDATED,
    KB_TOTAL_PARENTS,
    KB_TOTAL_CHILDREN,
    groq_client
)

# ─────────────────────────────────────────
# LOAD ENV
# ─────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ─────────────────────────────────────────
# RATE LIMITING
# ─────────────────────────────────────────
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# ─────────────────────────────────────────
# INPUT GUARDRAILS CONFIG
# ─────────────────────────────────────────
INJECTION_BLOCKLIST = [
    "ignore previous",
    "forget instructions", 
    "you are now",
    "act as"
]
MAX_QUERY_LENGTH = 500

def sanitize_query(query: str) -> str:
    """Strip and basic sanitize query."""
    return query.strip()

def check_injection(query: str) -> tuple[bool, str]:
    """Check for prompt injection attempts."""
    query_lower = query.lower()
    for block in INJECTION_BLOCKLIST:
        if block in query_lower:
            return False, f"Query contains blocked phrase: '{block}'"
    return True, ""

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(".", "kb.html")

@app.route("/health", methods=["GET"])
def health():
    # Check LLM health
    llm_status = check_llm_health()
    
    return jsonify({
        "status": "running",
        "kb_version": KB_VERSION,
        "kb_last_updated": KB_LAST_UPDATED,
        "kb_total_parents": KB_TOTAL_PARENTS,
        "kb_total_children": KB_TOTAL_CHILDREN,
        "llm_status": llm_status,
    }), 200

@app.route("/ask", methods=["POST"])
@limiter.limit("30 per minute")
def ask():
    start_time = time.time()
    try:
        body = request.json
        if not body or not body.get("query"):
            return jsonify({"error": "Query is missing or empty"}), 400

        query = body["query"]
        history = body.get("history", [])
        recommendations = []

        # Check query length
        if len(query) > MAX_QUERY_LENGTH:
            return jsonify({"error": f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters"}), 400

        # Sanitize query
        query = sanitize_query(query)
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        # Check for injection
        is_safe, error_msg = check_injection(query)
        if not is_safe:
            return jsonify({"error": error_msg}), 400

        # Check cache (skip if personalized query with pronouns)
        cache_key = get_cache_key(query)
        if should_use_cache(query) and cache_key in response_cache:
            cached = response_cache[cache_key]
            return jsonify({
                "answer": cached["answer"],
                "model_used": cached.get("model_used", "cached"),
                "no_info": cached.get("no_info", False),
                "sources": cached.get("sources", []),
                "titles": cached.get("titles", []),
                "recommendations": recommendations,
                "cached": True
            })

        # Query Reformulation: Use only the last 3 turns for context stability
        search_query = query
        if history:
            try:
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
                search_query = sanitize_query(rewrite_comp.choices[0].message.content.strip().strip('"\''))
            except Exception as e:
                pass  # Silently fall back to original query

        # Search and generate answer
        results = search(search_query)
        answer, model_used = generate_answer(search_query, results)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Log request details
        log_request(query, search_query, results, model_used, latency_ms, len(answer))

        # Detect [NO_INFO] signal from the AI
        no_info = answer.strip().startswith("[NO_INFO]")

        response_data = {
            "answer":     answer,
            "model_used": model_used,
            "no_info":    no_info,
            "sources":    list(dict.fromkeys(r["source"] for r in results)),
            "titles":     list(dict.fromkeys(r["title"]  for r in results)),
            "recommendations": recommendations,
            "cached": False
        }

        if no_info:
            search_term = urllib.parse.quote_plus(query)
            response_data["answer"] = "I don't have that information in the knowledge base. Please check the links below:"
            response_data["search_url"] = f"https://hushly.freshdesk.com/support/search/solutions?term={search_term}"
            response_data["sources"] = []
            response_data["titles"] = []
        else:
            # Cache successful responses (only if cacheable)
            if should_use_cache(query):
                response_cache[cache_key] = response_data

        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/enhance", methods=["POST"])
@limiter.limit("10 per minute")
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
@limiter.limit("10 per minute")
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

@app.route("/feedback", methods=["POST"])
@limiter.limit("60 per minute")
def feedback():
    """Accept feedback for eval dataset."""
    try:
        body = request.json
        if not body:
            return jsonify({"error": "Request body is missing"}), 400
        
        query = body.get("query", "")
        answer = body.get("answer", "")
        helpful = body.get("helpful")
        
        if helpful is None:
            return jsonify({"error": "helpful field is required (true/false)"}), 400
        
        # Append to feedback log
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "helpful": helpful
        }
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Append to feedback.jsonl
        with open("logs/feedback.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        
        return jsonify({"success": True}), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True, use_reloader=False)