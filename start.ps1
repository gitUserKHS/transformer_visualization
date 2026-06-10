$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$logPath = Join-Path $PSScriptRoot "startup.log"
$transcriptStarted = $false

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

function Get-BasePython {
    $py = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($py) {
        return @{
            File = $py.Source
            Prefix = @("-3")
        }
    }

    $python = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($python) {
        return @{
            File = $python.Source
            Prefix = @()
        }
    }

    throw "Python 3 was not found. Install Python 3.11+ and run start.bat again."
}

function Initialize-CpuEnvironment {
    $cpuVenv = Join-Path $PSScriptRoot ".venv"
    $cpuPython = Join-Path $cpuVenv "Scripts\python.exe"
    if (-not (Test-Path $cpuPython)) {
        $basePython = Get-BasePython
        Invoke-Checked `
            -FilePath $basePython.File `
            -ArgumentList ($basePython.Prefix + @("-m", "venv", $cpuVenv)) `
            -Description "Creating the local Python environment"
    }

    Invoke-Checked `
        -FilePath $cpuPython `
        -ArgumentList @("-m", "pip", "install", "--upgrade", "pip") `
        -Description "Updating pip"
    Invoke-Checked `
        -FilePath $cpuPython `
        -ArgumentList @("-m", "pip", "install", "-r", (Join-Path $PSScriptRoot "requirements.txt")) `
        -Description "Installing Python dependencies"
    return $cpuPython
}

try {
    try {
        Start-Transcript -Path $logPath -Append | Out-Null
        $transcriptStarted = $true
    } catch {
        Write-Warning "Could not open startup.log. Startup will continue."
    }

    Write-Host "Transformer Learning Studio" -ForegroundColor Magenta
    Write-Host "Project: $PSScriptRoot"

    $frontend = Join-Path $PSScriptRoot "frontend"
    $distIndex = Join-Path $frontend "dist\index.html"
    $nodeModules = Join-Path $frontend "node_modules"
    $needsBuild = -not (Test-Path $distIndex)

    if (-not $needsBuild) {
        $distTime = (Get-Item $distIndex).LastWriteTimeUtc
        $newerSource = Get-ChildItem `
            -Path (Join-Path $frontend "src"), (Join-Path $frontend "package.json") `
            -Recurse `
            -File |
            Where-Object { $_.LastWriteTimeUtc -gt $distTime } |
            Select-Object -First 1
        $needsBuild = $null -ne $newerSource
    }

    if ($needsBuild) {
        $npm = Get-Command "npm.cmd" -ErrorAction SilentlyContinue
        if (-not $npm) {
            throw "Node.js/npm was not found. Install Node.js 20+ and run start.bat again."
        }

        if (-not (Test-Path $nodeModules)) {
            Invoke-Checked `
                -FilePath $npm.Source `
                -ArgumentList @("ci", "--prefix", $frontend) `
                -Description "Installing frontend dependencies"
        }

        Invoke-Checked `
            -FilePath $npm.Source `
            -ArgumentList @("run", "build", "--prefix", $frontend) `
            -Description "Building the frontend"
    }

    $gpuPython = Join-Path $PSScriptRoot ".venv-gpu\Scripts\python.exe"
    $cpuPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    $python = $null

    if (Test-Path $gpuPython) {
        $python = $gpuPython
        Write-Host "Using GPU environment: .venv-gpu" -ForegroundColor Green
    } elseif (Test-Path $cpuPython) {
        $python = $cpuPython
        Write-Host "Using CPU environment: .venv" -ForegroundColor Yellow
    } else {
        $nvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
        if ($nvidiaSmi) {
            Write-Host ""
            Write-Host "First launch: preparing the CUDA environment." -ForegroundColor Yellow
            Write-Host "This downloads PyTorch and can take several minutes."
            try {
                & (Join-Path $PSScriptRoot "setup_gpu.ps1")
                if ($LASTEXITCODE -ne 0 -or -not (Test-Path $gpuPython)) {
                    throw "GPU setup did not create .venv-gpu."
                }
                $python = $gpuPython
            } catch {
                Write-Warning "GPU setup failed: $($_.Exception.Message)"
                Write-Host "Falling back to a CPU environment for the learning microscope."
                $python = Initialize-CpuEnvironment
            }
        } else {
            Write-Host ""
            Write-Host "No NVIDIA GPU was detected. Preparing a CPU environment." -ForegroundColor Yellow
            $python = Initialize-CpuEnvironment
        }
    }

    & $python -c "import fastapi, torch, tokenizers, safetensors, uvicorn"
    if ($LASTEXITCODE -ne 0) {
        Invoke-Checked `
            -FilePath $python `
            -ArgumentList @("-m", "pip", "install", "-r", (Join-Path $PSScriptRoot "requirements.txt")) `
            -Description "Repairing Python dependencies"
    }

    Write-Host ""
    Write-Host "Opening http://127.0.0.1:8000" -ForegroundColor Green
    Write-Host "Keep this window open while using the app. Press Ctrl+C to stop."
    & $python (Join-Path $PSScriptRoot "backend\scripts\launch_app.py")
    if ($LASTEXITCODE -ne 0) {
        throw "The application server stopped with exit code $LASTEXITCODE."
    }
} catch {
    Write-Host ""
    Write-Host "Startup failed" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Details: $logPath" -ForegroundColor Yellow
    exit 1
} finally {
    if ($transcriptStarted) {
        try {
            Stop-Transcript | Out-Null
        } catch {
            # The host may already have stopped the transcript after Ctrl+C.
        }
    }
}
