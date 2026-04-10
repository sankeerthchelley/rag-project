# Graph Report - .  (2026-04-09)

## Corpus Check
- Corpus is ~8,060 words - fits in a single context window. You may not need a graph.

## Summary
- 18 nodes · 17 edges · 5 communities detected
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `ask()` - 3 edges
2. `search()` - 2 edges
3. `generate_answer()` - 2 edges
4. `evaluate_answer()` - 2 edges
5. `run_evaluation()` - 2 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Communities

### Community 0 - "Community 0"
Cohesion: 0.32
Nodes (3): ask(), generate_answer(), search()

### Community 1 - "Community 1"
Cohesion: 1.0
Nodes (2): evaluate_answer(), run_evaluation()

### Community 2 - "Community 2"
Cohesion: 0.67
Nodes (0): 

### Community 3 - "Community 3"
Cohesion: 1.0
Nodes (0): 

### Community 4 - "Community 4"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **Thin community `Community 3`** (2 nodes): `run_graphify.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 4`** (2 nodes): `script.py`, `get_text()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Not enough signal to generate questions. This usually means the corpus has no AMBIGUOUS edges, no bridge nodes, no INFERRED relationships, and all communities are tightly cohesive. Add more files or run with --mode deep to extract richer edges._