"""
Hushly RAG Web API - Flask application for production serving.

SYSTEM CONTEXT:
- WHERE: Deployed to cloud (Heroku/AWS/etc). Entry point for web requests.
- WHEN: Runs continuously as a service. Started via Gunicorn (see Procfile).
- WHAT: HTTP endpoints for /ask (RAG queries), /health (status), /feedback (evaluation data).
  Handles rate limiting, input guardrails, caching, and orchestrates core.search() + core.generate_answer().
"""

from flask import Flask, request, jsonify, send_from_directory
import json
import os
import re
import time
import uuid
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
INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|above)\s+instructions?",
    r"forget\s+(everything|instructions?|context)",
    r"you\s+are\s+now",
    r"act\s+as",
    r"jailbreak",
    r"pretend\s+(you\s+are|to\s+be)",
]
MAX_QUERY_LENGTH = 500

HUSHLY_TOPICS = [
    "hushly", "experience", "stream", "hub", "asset", "page",
    "persona", "utm", "abm", "geo", "geosherpa", "aeo",
    "segment", "integration", "content", "upload", "account",
    "csm", "template", "campaign", "visitor", "lead", "form"
]

def is_on_topic(query: str) -> bool:
    """Check if the query contains any Hushly-related keywords."""
    query_lower = query.lower()
    return any(topic in query_lower for topic in HUSHLY_TOPICS)

def sanitize_query(query: str) -> str:
    """Strip and basic sanitize query."""
    return query.strip()

def is_injection(query: str) -> bool:
    """Check for prompt injection attempts using regex patterns."""
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in INJECTION_PATTERNS)

# Reference words that signal a follow-up question needing prior context
_FOLLOWUP_PRONOUNS = {"it", "this", "that", "these", "those", "they", "them", "there"}
_FOLLOWUP_PHRASES = ["what about", "how about", "tell me more", "explain more", "what else", "and the"]

def is_followup_question(query: str) -> bool:
    """
    Returns True only when the query is clearly a follow-up to the previous turn.
    Standalone new-topic questions return False — they skip reformulation entirely.
    """
    q = query.lower().strip()
    words = set(q.split())
    if len(words) <= 4:
        return True
    if words & _FOLLOWUP_PRONOUNS:
        return True
    return any(phrase in q for phrase in _FOLLOWUP_PHRASES)

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(".", "kb.html")

@app.route("/health", methods=["GET"])
@limiter.exempt
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

        # Check for injection (do not log the query content itself)
        if is_injection(query):
            return jsonify({"error": "Invalid query"}), 400

        # Topic Guard: Check if query is related to Hushly
        if not is_on_topic(query):
            return jsonify({
                "answer": None,
                "no_info": True,
                "no_info_reason": "off_topic",
                "sources": [],
                "titles": []
            }), 200

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

        # Query Reformulation: only runs when query is a follow-up (contains reference
        # pronouns, is very short, or uses bridging phrases). New-topic questions skip
        # this entirely — no history bias, no extra latency.
        search_query = query
        if history and is_followup_question(query):
            try:
                context_history = history[-3:]
                history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in context_history])

                rewrite_prompt = f"""Given the conversation history below, rewrite the user's latest follow-up question into a standalone, fully contextualized question. Do not answer it, just rewrite it so it can be used for a database search keyword lookup. Output ONLY the standalone question.

Conversation History (Latest Turns):
{history_text}

Latest raw question: {query}

Standalone question:"""

                rewrite_comp = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": rewrite_prompt}],
                    max_tokens=100,
                    temperature=0.1
                )
                search_query = sanitize_query(rewrite_comp.choices[0].message.content.strip().strip('"\''))
            except Exception:
                pass  # Silently fall back to original query

        # Search and generate answer
        results = search(search_query)
        
        if not results:
            return jsonify({
                "answer": None,
                "no_info": True,
                "no_info_reason": "low_relevance",
                "sources": [],
                "titles": []
            }), 200

        answer, model_used = generate_answer(search_query, results)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Log request details
        log_request(query, search_query, results, model_used, latency_ms, len(answer))

        # Detect [NO_INFO] signal from the AI (case-insensitive and position-independent)
        no_info = "[NO_INFO]" in answer.strip().upper()

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

        enhance_prompt = f"""You are a search query optimizer for the Hushly knowledge base.

Your ONLY job is to take a vague or short input and rewrite it as ONE clear, specific question.

RULES:
- Maximum 20 words in output
- Add "in Hushly" if not already implied
- Expand abbreviations only if directly relevant: UTM = URL tracking parameters, GEO = Generative Engine Optimization, ABM = Account-Based Marketing
- Do NOT add features, tools, or concepts the user did not mention
- Do NOT stuff keywords
- Output ONLY the question, nothing else

Examples:
Input: configure → Output: How do I configure settings in Hushly?
Input: UTM → Output: How do I set up UTM tracking parameters in Hushly?
Input: settings → Output: What settings can I manage in Hushly?
Input: upload asset → Output: How do I upload an asset in Hushly?

Input: {raw_query}
Output:"""

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
convert it into a strict JSON object.

Each step must have:
- "text": Concise instruction (1 sentence, conversational, suitable to be spoken aloud).
- "element_hint": The exact label/name of the UI element the user should click on that
  step (e.g., "Assets", "Upload", "Content", "Save"). Use an empty string if the step
  does not involve clicking a specific labeled element.

If the answer is purely informational (e.g., "What is ABM?") and does not contain
clear chronological steps, set "is_actionable_task" to false.

Answer Text:
{answer_text}

Output Format strictly as JSON:
{{
  "is_actionable_task": true,
  "task_title": "Upload an Asset",
  "steps": [
    {{
      "text": "First, open the left sidebar menu and click on Content.",
      "element_hint": "Content"
    }},
    {{
      "text": "Next, click on Assets.",
      "element_hint": "Assets"
    }},
    {{
      "text": "Finally, click the Upload button and select your file.",
      "element_hint": "Upload"
    }}
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

@app.route("/analyze_page", methods=["POST"])
@limiter.limit("10 per minute")
def analyze_page():
    """Analyze a live page and provide context about it using the knowledge base."""
    try:
        body = request.json
        if not body:
            return jsonify({"error": "Request body is missing"}), 400
        
        url = body.get("url", "")
        title = body.get("title", "")
        page_elements = body.get("page_elements", [])
        
        if not url:
            return jsonify({"error": "URL is required"}), 400
        
        # Extract page path and key identifiers
        parsed_url = urllib.parse.urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        # Build element summary for context
        element_summary = []
        for el in page_elements[:30]:  # Top 30 elements
            text = el.get("text", "")
            tag = el.get("tag", "")
            if text and len(text) < 50:
                element_summary.append(f"[{tag}] {text}")
        
        # Build context-rich query for structured output
        elements_text = ', '.join(element_summary[:20])

        structured_prompt = f"""You are analyzing a Hushly platform page. Based on the page information below, provide a structured summary.

Page URL: {url}
Page Title: {title}
Path: {parsed_url.path}
Visible UI Elements: {elements_text}

Return your analysis in this exact JSON structure:
{{
  "page_context": "2-5 sentences explaining what this page is for and its primary purpose",
  "key_features": [
    "Feature name - Brief description of what it does",
    "Feature name - Brief description of what it does"
  ],
  "navigation_summary": [
    {{
      "section": "Left Nav Menu / Top Bar / Main Content / etc",
      "purpose": "What this section contains or provides access to (e.g., Access to Assets, Experiences, Hubs, and Analytics)"
    }},
    {{
      "section": "Main Table / Content Area / etc",
      "purpose": "What data or content is displayed here"
    }}
  ]
}}

Rules:
- page_context: Maximum 5 lines, conversational tone
- key_features: 3-5 bullet points describing major interactive elements
- navigation_summary: Break down the page layout into logical sections with their purposes
- Use specific element names found in the UI elements list above
- If this is a forms/listing page, describe what items are typically shown"""

        # Search for relevant knowledge base content
        results = search(structured_prompt, k=10)

        if not results:
            return jsonify({
                "page_context": "This appears to be a page on the Hushly platform, but I don't have specific documentation about it.",
                "key_features": [],
                "navigation_summary": [],
                "no_info": True,
                "no_info_reason": "page_not_recognized",
                "sources": [],
                "titles": []
            }), 200

        # Generate contextual answer with structured format
        context_blocks = []
        for r in results:
            block = f"[{r['title']}]\n{r['context']}"
            context_blocks.append(block)
        context = "\n\n---\n\n".join(context_blocks)
        context = context[:12000]

        full_prompt = f"""Use the following knowledge base context to answer the question.

{context}

{structured_prompt}

Output strictly as JSON:"""

        # Try Groq first with JSON format
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": full_prompt}],
                response_format={"type": "json_object"},
                max_tokens=800,
                temperature=0.2
            )
            answer_text = completion.choices[0].message.content
            model_used = "groq"
        except Exception as e:
            logger.warning(f"Groq structured analysis failed: {e}")
            # Fallback: generate plain answer
            answer, model_used = generate_answer(structured_prompt, results)
            answer_text = json.dumps({
                "page_context": answer,
                "key_features": [],
                "navigation_summary": []
            })

        # Parse the JSON response
        try:
            structured_data = json.loads(answer_text)
        except json.JSONDecodeError:
            # If not valid JSON, wrap it
            structured_data = {
                "page_context": answer_text,
                "key_features": [],
                "navigation_summary": []
            }

        # Ensure all required fields exist
        response_data = {
            "page_context": structured_data.get("page_context", ""),
            "key_features": structured_data.get("key_features", []),
            "navigation_summary": structured_data.get("navigation_summary", []),
            "no_info": "[NO_INFO]" in str(answer_text).upper() or len(structured_data.get("key_features", [])) == 0,
            "sources": list(dict.fromkeys(r["source"] for r in results)),
            "titles": list(dict.fromkeys(r["title"] for r in results)),
            "page_path": parsed_url.path,
            "model_used": model_used
        }
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"analyze_page error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────
# PATH MEMORY STORE — helpers
# ─────────────────────────────────────────
DATA_DIR = "data"
PATHS_FILE  = os.path.join(DATA_DIR, "paths.json")
GUIDE_FEEDBACK_FILE = os.path.join(DATA_DIR, "guide_feedback.json")

def _load_json(path, default):
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path, data):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def _normalize(text):
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()

def _match_path(query, paths):
    """Return (best_path, score) from stored paths. Threshold 0.30."""
    q_words = set(_normalize(query))
    best, best_score = None, 0.0
    for p in paths.values():
        if p.get("status") == "ignored":
            continue
        keywords = p.get("keywords", [])
        title_words = _normalize(p.get("task_title", ""))
        p_words = set(title_words + [w for kw in keywords for w in _normalize(kw)])
        if not p_words:
            continue
        overlap = len(q_words & p_words)
        score = overlap / max(len(q_words), len(p_words), 1)
        if score > best_score:
            best_score = score
            best = p
    if best_score >= 0.30:
        return best, round(best_score, 2)
    return None, 0.0

# ─────────────────────────────────────────
# SMART GUIDE — check cache or call LLM
# ─────────────────────────────────────────
@app.route("/smart_guide", methods=["POST"])
@limiter.limit("20 per minute")
def smart_guide():
    try:
        body = request.json or {}
        query        = body.get("query", "").strip()
        answer       = body.get("answer", "").strip()
        page_elements = body.get("page_elements", [])

        if not query:
            return jsonify({"error": "query required"}), 400

        paths = _load_json(PATHS_FILE, {})

        # ── 1. Check stored paths ──────────────────────
        matched, score = _match_path(query, paths)
        if matched:
            return jsonify({**matched, "from_cache": True, "match_score": score})

        # ── 2. No stored path — need page scan ─────────
        if not page_elements:
            return jsonify({"needs_scan": True})

        # ── 3. LLM maps answer steps → page elements ───
        elements_str = "\n".join(
            f"[{e['idx']}] {e['tag'].upper()} | \"{e['text'][:50]}\" "
            f"| aria:\"{e.get('aria_label','')}\" | sel:\"{e.get('selector','')}\""
            for e in page_elements[:80]
        )

        prompt = f"""You are a UI guide assistant for the Hushly SaaS platform.

User asked: "{query}"

Knowledge base answer:
{answer[:800]}

Visible page elements (index | tag | text | aria-label | CSS selector):
{elements_str}

CRITICAL RULES:
1. element_text and element_hint MUST be copied VERBATIM from the "text" or "aria:" column above.
   Do NOT paraphrase. If the page says "Add Asset", write "Add Asset" — never "Create Asset".
2. Match each KB step to the closest page element. Use element_idx -1 only if no element exists.
3. The step "text" should say exactly which label to click (use the verbatim label in quotes).

Return strictly as JSON:
{{
  "task_title": "short title",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "steps": [
    {{
      "text": "Click \\'Content\\' in the left sidebar",
      "element_idx": 3,
      "element_text": "Content",
      "element_hint": "Content",
      "selector": "<copy the sel: value for that element>",
      "selector_fallbacks": []
    }}
  ]
}}"""

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=700,
            temperature=0.1,
        )
        result = json.loads(completion.choices[0].message.content)
        result["from_cache"] = False
        result["status"] = "auto"
        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# SAVE / UPDATE A GUIDE PATH
# ─────────────────────────────────────────
@app.route("/guide/save_path", methods=["POST"])
@limiter.limit("20 per minute")
def save_guide_path():
    try:
        body = request.json or {}
        task_title = body.get("task_title", "").strip()
        keywords   = body.get("keywords", [])
        steps      = body.get("steps", [])

        if not task_title or not steps:
            return jsonify({"error": "task_title and steps required"}), 400

        path_id = re.sub(r"[^a-z0-9]+", "-", task_title.lower()).strip("-")
        paths   = _load_json(PATHS_FILE, {})
        now     = datetime.now().isoformat()

        if path_id in paths:
            paths[path_id].update({"steps": steps, "keywords": keywords, "updated_at": now})
            paths[path_id]["use_count"] = paths[path_id].get("use_count", 0) + 1
        else:
            paths[path_id] = {
                "id": path_id, "task_title": task_title, "keywords": keywords,
                "steps": steps, "status": "auto",
                "created_at": now, "updated_at": now,
                "use_count": 1, "success_count": 0
            }

        _save_json(PATHS_FILE, paths)
        return jsonify({"saved": path_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# GUIDE STEP FEEDBACK
# ─────────────────────────────────────────
@app.route("/guide/feedback", methods=["POST"])
@limiter.limit("30 per minute")
def guide_step_feedback():
    try:
        body = request.json or {}
        path_id  = body.get("path_id", "")
        step_idx = body.get("step_idx", -1)
        issue    = body.get("issue", "wrong_element")
        comment  = body.get("comment", "")
        page_url = body.get("page_url", "")

        fb_list = _load_json(GUIDE_FEEDBACK_FILE, [])
        entry = {
            "id":        str(uuid.uuid4())[:8],
            "path_id":   path_id,
            "step_idx":  step_idx,
            "issue":     issue,
            "comment":   comment,
            "page_url":  page_url,
            "timestamp": datetime.now().isoformat(),
            "status":    "pending",
            "admin_action": None
        }
        fb_list.append(entry)
        _save_json(GUIDE_FEEDBACK_FILE, fb_list)

        # Flag the path so admin knows to review it
        if path_id:
            paths = _load_json(PATHS_FILE, {})
            if path_id in paths and paths[path_id].get("status") != "validated":
                paths[path_id]["status"] = "flagged"
                _save_json(PATHS_FILE, paths)

        return jsonify({"logged": entry["id"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# ADMIN PANEL
# ─────────────────────────────────────────
@app.route("/admin")
def admin_panel():
    return send_from_directory(".", "admin.html")

@app.route("/admin/api/paths")
def admin_api_paths():
    return jsonify(_load_json(PATHS_FILE, {}))

@app.route("/admin/api/feedback")
def admin_api_feedback():
    return jsonify(_load_json(GUIDE_FEEDBACK_FILE, []))

@app.route("/admin/api/paths/<path_id>", methods=["POST"])
def admin_update_path(path_id):
    try:
        body  = request.json or {}
        paths = _load_json(PATHS_FILE, {})
        if path_id not in paths:
            return jsonify({"error": "not found"}), 404
        for field in ("steps", "status", "keywords", "task_title"):
            if field in body:
                paths[path_id][field] = body[field]
        paths[path_id]["updated_at"] = datetime.now().isoformat()
        _save_json(PATHS_FILE, paths)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/api/feedback/<fid>/action", methods=["POST"])
def admin_feedback_action(fid):
    try:
        body    = request.json or {}
        action  = body.get("action")          # confirm | correct | ignore
        new_steps = body.get("steps")

        fb_list = _load_json(GUIDE_FEEDBACK_FILE, [])
        entry   = next((f for f in fb_list if f["id"] == fid), None)
        if not entry:
            return jsonify({"error": "not found"}), 404

        entry["status"]       = "resolved"
        entry["admin_action"] = action
        _save_json(GUIDE_FEEDBACK_FILE, fb_list)

        if entry.get("path_id"):
            paths = _load_json(PATHS_FILE, {})
            path  = paths.get(entry["path_id"])
            if path:
                if action == "confirm":
                    path["status"] = "validated"
                elif action == "ignore":
                    path["status"] = "auto"   # unflag, keep path
                elif action == "correct" and new_steps:
                    path["steps"]   = new_steps
                    path["status"]  = "corrected"
                    path["updated_at"] = datetime.now().isoformat()
                _save_json(PATHS_FILE, paths)

        return jsonify({"ok": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode, use_reloader=False)