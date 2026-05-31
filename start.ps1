# start.ps1 — Launch RAG API with UTF-8 encoding (Windows-safe)

# 1. Set PowerShell console to UTF-8
chcp 65001 | Out-Null
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 2. Force Python to use UTF-8 for stdout/stderr
$env:PYTHONIOENCODING = "utf-8"

# 3. Check virtual environment exists
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[start] ERROR: Virtual environment not found. Run: python -m venv .venv" -ForegroundColor Red
    exit 1
}

# 4. Activate virtual environment
& .venv\Scripts\Activate.ps1

# 5. Set Ollama host (IPv4 to avoid Windows IPv6 issues)
$env:OLLAMA_HOST = "http://127.0.0.1:11434"

# 6. Start Uvicorn
Write-Host "[start] Starting uvicorn with UTF-8 console..." -ForegroundColor Green
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
