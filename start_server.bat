@echo off
echo Starting RAG Server...
start "" http://localhost:8000
python app.py
pause
