@echo off
REM start.bat — Launch RAG API with UTF-8 encoding (Windows-safe)

REM 1. Set console to UTF-8
chcp 65001 >nul

REM 2. Force Python to use UTF-8 for stdout/stderr
set PYTHONIOENCODING=utf-8

REM 3. Activate virtual environment
call .venv\Scripts\activate.bat

REM 4. Set Ollama host (IPv4 to avoid Windows IPv6 issues)
set OLLAMA_HOST=http://127.0.0.1:11434

REM 5. Start Uvicorn
echo [start] Starting uvicorn with UTF-8 console...
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
