import json
import faiss
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from google import genai
from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────────────────────
# LOAD ENV
# ─────────────────────────────────────────
load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Testing AI connections...\n")
try:
    res = gemini_client.models.generate_content(model="gemini-2.0-flash", contents="Say OK")
    print("✅ Gemini connected:", res.text.strip())
except Exception as e:
    print("❌ Gemini failed:", str(e))

try:
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say OK"}]
    )
    print("✅ Groq connected:", completion.choices[0].message.content.strip())
except Exception as e:
    print("❌ Groq failed:", str(e))

print("\n--- Loading RAG System ---\n")

# ─────────────────────────────────────────
# LOAD CHUNKS
# ─────────────────────────────────────────
with open("chunks_enterprise.json", "r", encoding="utf-8") as f:
    data = json.load(f)

children = [c for c in data.get("children", []) if not c.get("deprecated", False)]
parents  = {p["chunk_id"]: p for p in data.get("parents", [])}

child_texts = [c.get("content", "") for c in children]
parent_ids  = [c.get("parent_chunk_id", "") for c in children]

print(f"[LOAD] {len(children)} child chunks | {len(parents)} parent chunks")

# ─────────────────────────────────────────
# EMBEDDINGS — saved to disk after first run
# ─────────────────────────────────────────
FAISS_INDEX_FILE = "faiss_index.bin"
FAISS_TEXTS_FILE = "faiss_texts.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_TEXTS_FILE):
    print("[FAISS] Loading saved index — fast startup ✅")
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(FAISS_TEXTS_FILE, "rb") as f:
        saved_meta = pickle.load(f)
    if saved_meta.get("count") != len(child_texts):
        print("[FAISS] Chunks changed — rebuilding...")
        os.remove(FAISS_INDEX_FILE)
        os.remove(FAISS_TEXTS_FILE)
        index = None
    else:
        print(f"[FAISS] {index.ntotal} vectors ready")
else:
    index = None

if index is None:
    print("[FAISS] Building embeddings — runs ONCE then saves...")
    embeddings = model.encode(child_texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_TEXTS_FILE, "wb") as f:
        pickle.dump({"count": len(child_texts)}, f)
    print(f"[FAISS] Saved to disk ✅")

print("\n✅ RAG system ready!\n")

# ─────────────────────────────────────────
# SEARCH WITH PARENT-CHILD RETRIEVAL
# ─────────────────────────────────────────
def search(query, k=15):
    query_vec = model.encode([query]).astype("float32")
    _, indices = index.search(query_vec, k)

    results = []
    seen = set()

    for i in indices[0]:
        if i < 0 or i >= len(children):
            continue

        child  = children[i]
        p_id   = child.get("parent_chunk_id", "")
        parent = parents.get(p_id)

        context_text = parent["content"] if parent else child["content"]
        dedup_key    = p_id or child["chunk_id"]

        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        results.append({
            "context": context_text,
            "match":   child["content"],
            "source":  child.get("source_url", ""),
            "title":   child.get("title", ""),
        })

    return results

# ─────────────────────────────────────────
# GENERATE ANSWER
# ─────────────────────────────────────────
def generate_answer(query, results):
    context = "\n\n---\n\n".join(
        f"[{r['title']}]\n{r['context']}" for r in results
    )

    prompt = f"""
You are an AI assistant embedded inside a SaaS product called Hushly.

Your goal is to provide accurate, context-aware responses using the knowledge base below.
Never hallucinate. If the answer is not in the KB, say so clearly.

DOMAIN LANGUAGE:
- "Experiences", "Streams", "Pages", "Hub", "UTMs" = Hushly product features

FORMAT RULES:
- HOW-TO → Short explanation + numbered steps
- DIRECT ANSWER → 1–3 concise lines
- If not in KB → "I don't have that information in the knowledge base."

KNOWLEDGE BASE:
{context}

QUESTION: {query}
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text, "gemini"
    except Exception as e:
        print(f"⚠️ Gemini failed ({e}), using Groq...")
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content, "groq"

# ─────────────────────────────────────────
# CHAT LOOP
# ─────────────────────────────────────────
while True:
    query = input("\nAsk something (or type 'exit'): ").strip()

    if query.lower() == "exit":
        print("Goodbye!")
        break

    if not query:
        continue

    results = search(query)
    answer, model_used = generate_answer(query, results)

    print(f"\n[{model_used.upper()}] Answer:\n")
    print(answer)
    print("\nSources:")
    for r in results:
        print(f"  - {r['source']}")