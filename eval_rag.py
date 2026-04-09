import os
import json
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# Import the RAG logic from app.py
from app import search, generate_answer

# Ragas evaluation imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 1. Define Test Set (Questions that should be in the KB)
test_questions = [
    "What is GEOSherpa in Hushly?",
    "How do I upload an asset to the platform?",
    "What are Hushly Experiences?",
    "What is the difference between a Stream and a Hub?",
    "How does Hushly handle ABM (Account-Based Marketing)?",
    "What are UTM parameters used for in Hushly?",
    "Can you explain what GEOSherpa means?",
    "How do I configure SSO in Hushly?",
    "What is the role of a CSM in Hushly?",
    "Define what an Asset is in the platform."
]

def run_evaluation():
    print(f"--- Starting RAG Evaluation Over {len(test_questions)} Questions ---")
    
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for query in test_questions:
        print(f"[QUERY] {query}")
        
        # Run the actual RAG pipeline
        results = search(query)
        answer, _ = generate_answer(query, results)
        
        # Format contexts (list of strings)
        contexts = [r['context'] for r in results]
        
        data["question"].append(query)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append("") # Optional: can be filled if we have gold answers

    # 2. Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(data)

    # 3. Setup Gemini as the Evaluator LLM (via LangChain)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # 4. Run Ragas Evaluation
    print("--- Running Ragas Metrics (Faithfulness, Relevance, Context Quality) ---")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm
    )

    # 5. Output Results
    df = result.to_pandas()
    print("\n--- EVALUATION RESULTS ---")
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])
    
    avg_scores = df.mean(numeric_only=True)
    print("\n--- AVERAGE SCORES ---")
    print(avg_scores)
    
    # Save to CSV for the user
    df.to_csv("rag_eval_results.csv", index=False)
    print("\n✅ Results saved to 'rag_eval_results.csv'")

if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
