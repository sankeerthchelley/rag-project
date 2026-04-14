import os
from core import check_llm_health, generate_answer

def test_health():
    print("Running health checks...")
    results = check_llm_health()
    print(f"Health check results: {results}")

def test_fallback():
    print("\nTesting fallback chain...")
    # This will likely fail with missing keys, but we want to see the error handling
    try:
        results = [{"context": "Test context", "title": "Test", "match": "Test", "source": "test.com"}]
        answer, model = generate_answer("What is Hushly?", results)
        print(f"Answer generated using: {model}")
        print(f"Answer: {answer[:100]}...")
    except Exception as e:
        print(f"Caught expected error during execution: {e}")

if __name__ == "__main__":
    test_health()
    test_fallback()
