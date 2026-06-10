$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    Write-Host ""
    Write-Host "==> $Description" -ForegroundColor Cyan
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed (exit code $LASTEXITCODE)."
    }
}

$nvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    throw "nvidia-smi was not found. Install an NVIDIA driver before GPU setup."
}

$py = Get-Command "py.exe" -ErrorAction SilentlyContinue
$python = Get-Command "python.exe" -ErrorAction SilentlyContinue
if ($py) {
    $basePython = $py.Source
    $pythonPrefix = @("-3")
} elseif ($python) {
    $basePython = $python.Source
    $pythonPrefix = @()
} else {
    throw "Python 3 was not found. Install Python 3.11+ before GPU setup."
}

$venvDir = Join-Path $PSScriptRoot ".venv-gpu"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Invoke-Checked `
        -FilePath $basePython `
        -ArgumentList ($pythonPrefix + @("-m", "venv", $venvDir)) `
        -Description "Creating .venv-gpu"
}

Invoke-Checked `
    -FilePath $venvPython `
    -ArgumentList @("-m", "pip", "install", "--upgrade", "pip") `
    -Description "Updating pip"
Invoke-Checked `
    -FilePath $venvPython `
    -ArgumentList @(
        "-m", "pip", "install", "--upgrade",
        "--index-url", "https://download.pytorch.org/whl/cu130",
        "torch==2.11.0", "torchvision==0.26.0", "torchaudio==2.11.0"
    ) `
    -Description "Installing PyTorch 2.11 CUDA 13.0"
Invoke-Checked `
    -FilePath $venvPython `
    -ArgumentList @("-m", "pip", "install", "-r", (Join-Path $PSScriptRoot "requirements.txt")) `
    -Description "Installing project dependencies"
Invoke-Checked `
    -FilePath $venvPython `
    -ArgumentList @(
        "-c",
        "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.__version__, torch.cuda.get_device_name(0))"
    ) `
    -Description "Verifying CUDA"

Write-Host ""
Write-Host "GPU environment is ready: .\.venv-gpu\Scripts\python.exe" -ForegroundColor Green
