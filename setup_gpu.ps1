$ErrorActionPreference = "Stop"

$python = (Get-Command python).Source
if (-not (Test-Path ".venv-gpu\Scripts\python.exe")) {
    & $python -m venv .venv-gpu
}

$venvPython = Resolve-Path ".venv-gpu\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu130 `
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
& $venvPython -m pip install -r requirements.txt

& $venvPython -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.__version__, torch.cuda.get_device_name(0))"
Write-Host "GPU environment is ready: .\.venv-gpu\Scripts\python.exe"
