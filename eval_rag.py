import os
import json
import time
from dotenv import load_dotenv
from google import genai
from app import search, generate_answer

load_dotenv()

# Initialize Gemini Client (already used in app.py)
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 1. Define Test Set
test_questions = [
    "What is GEOSherpa in Hushly?",
    "How do I upload an asset to the platform?",
    "What are Hushly Experiences?",
    "What is the difference between a Stream and a Hub?",
    "How does Hushly handle ABM (Account-Based Marketing)?",
    "What are UTM parameters used for in Hushly?",
    "What is the role of a CSM in Hushly?"
]

EVAL_RUBRIC = """
You are an expert AI grader. Evaluate the "Answer" based on the provided "Context" and "Question".
Return a JSON object with scores (1-5) and reasoning for the following criteria:

1. Faithfulness: Is the answer derived ONLY from the context? (1 = hallucinating, 5 = perfectly grounded)
2. Relevance: Does the answer address the user question accurately? (1 = irrelevant, 5 = perfect)

JSON Format:
{
  "faithfulness": { "score": 5, "reasoning": "..." },
  "relevance": { "score": 5, "reasoning": "..." }
}
"""

def evaluate_answer(question, context, answer):
    prompt = f"""{EVAL_RUBRIC}

Context:
{context}

Question: {question}
Answer: {answer}
"""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"  [ERROR] Evaluation failed for this question: {e}")
        return None

def run_evaluation():
    print(f"--- Starting Lightweight RAG Evaluation Over {len(test_questions)} Questions ---")
    print("--- (Python 3.14 Compatible - Dependency Free) ---")
    
    results_list = []

    for query in test_questions:
        print(f"\n[QUERY] {query}")
        
        # 1. Search & Generate (Production code)
        retrieved = search(query)
        context_text = "\n---\n".join([r['context'] for r in retrieved])
        answer, model_used = generate_answer(query, retrieved)
        
        print(f"  [BOT] {answer[:100]}...")

        # 2. Evaluate with Gemini Judge
        print("  [JUDGE] Evaluating...")
        eval_result = evaluate_answer(query, context_text, answer)
        
        if eval_result:
            print(f"  [SCORE] Faithfulness: {eval_result['faithfulness']['score']}/5 | Relevance: {eval_result['relevance']['score']}/5")
            eval_result['question'] = query
            eval_result['answer'] = answer
            eval_result['model_used'] = model_used
            results_list.append(eval_result)
        
        time.sleep(1) # Rate limit safety

    # 3. Summary
    if results_list:
        avg_faith = sum(r['faithfulness']['score'] for r in results_list) / len(results_list)
        avg_rel = sum(r['relevance']['score'] for r in results_list) / len(results_list)
        
        print("\n" + "="*40)
        print("AVERAGE PERFORMANCE SCORES")
        print(f"Faithfulness: {avg_faith:.2f} / 5.0")
        print(f"Relevance:    {avg_rel:.2f} / 5.0")
        print("="*40)
        
        with open("rag_eval_results.json", "w") as f:
            json.dump(results_list, f, indent=2)
        print("✅ Detailed results saved to 'rag_eval_results.json'")

if __name__ == "__main__":
    run_evaluation()
