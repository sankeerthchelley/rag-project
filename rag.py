"""
RAG CLI Interface - Command-line chat for local testing.

SYSTEM CONTEXT:
- WHERE: Run locally by developers/admins. No web server needed.
- WHEN: Used for quick testing, debugging, or when you just want a terminal chat.
- WHAT: Interactive loop that takes user input, calls core.search() + core.generate_answer(),
  prints answers with sources. Tests AI connections at startup.
"""

from core import search, generate_answer, groq_client

# Test AI connections
print("Testing AI connections...\n")
try:
    from google import genai
    from dotenv import load_dotenv
    import os
    load_dotenv()
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
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

print("\n✅ RAG system ready!\n")

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