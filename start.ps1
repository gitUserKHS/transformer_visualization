$ErrorActionPreference = "Stop"

if (-not (Test-Path "frontend/dist/index.html")) {
    Push-Location frontend
    try {
        cmd /c npm run build
    } finally {
        Pop-Location
    }
}

$python = "python"
$gpuPython = Join-Path $PSScriptRoot ".venv-gpu\Scripts\python.exe"
if (Test-Path $gpuPython) {
    $python = $gpuPython
}

& $python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
