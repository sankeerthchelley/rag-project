@echo off
echo Running initial startup checks...
pip install -r requirements.txt --quiet
echo.
echo Starting RAG Server...
start "" http://localhost:8000
python app.py
pause
