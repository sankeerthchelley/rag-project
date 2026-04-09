import os
import json
import subprocess
import sys
from pathlib import Path

# Try to import graphify components
try:
    from graphify.detect import detect
    from graphify.extract import extract
    from graphify.build import build_from_json
    from graphify.cluster import cluster, score_all
    from graphify.analyze import god_nodes, surprising_connections, suggest_questions
    from graphify.report import generate
    from graphify.export import to_json, to_html
except ImportError:
    print("❌ Error: graphifyy not found. Please run 'pip install graphifyy'")
    sys.exit(1)

def main():
    print("--- Graphifyy Knowledge Graph Generator ---")
    
    # 1. Detect Files
    print("Detecting files...")
    corpus = detect(Path('.'))
    with open('.graphify_detect.json', 'w', encoding='utf-8') as f:
        json.dump(corpus, f)
    
    # 2. Extract Structural (AST) Data
    print("Extracting code structure (AST)...")
    code_files = [Path(f) for f in corpus.get('files', {}).get('code', [])]
    if code_files:
        extraction = extract(code_files)
    else:
        extraction = {"nodes": [], "edges": []}
    
    # 3. Build & Analyze
    print("Building knowledge graph...")
    Path('graphify-out').mkdir(exist_ok=True)
    G = build_from_json(extraction)
    
    communities = cluster(G)
    cohesion = score_all(G, communities)
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    labels = {cid: f"Community {cid}" for cid in communities}
    questions = suggest_questions(G, communities, labels)
    
    # 4. Generate Report
    print("Generating reports...")
    tokens = {'input': 0, 'output': 0}
    report = generate(G, communities, cohesion, labels, gods, surprises, corpus, tokens, '.', suggested_questions=questions)
    
    with open('graphify-out/GRAPH_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 5. Export Visualization
    to_json(G, communities, 'graphify-out/graph.json')
    to_html(G, communities, 'graphify-out/graph.html', community_labels=labels)
    
    print(f"\n✅ Graph complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print("Open 'graphify-out/graph.html' in your browser to explore!")
    print("Read 'graphify-out/GRAPH_REPORT.md' for an architectural audit.")

if __name__ == "__main__":
    main()
