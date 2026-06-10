import platform
import subprocess

import torch


GPU_SETUP_COMMAND = (
    ".\\setup_gpu.ps1; .\\.venv-gpu\\Scripts\\python.exe "
    "-m uvicorn backend.app.main:app --reload"
)


def system_diagnostics() -> dict:
    cuda = torch.cuda.is_available()
    gpu = None
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        gpu = {
            "name": properties.name,
            "total_vram_bytes": total,
            "free_vram_bytes": free,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    else:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            values = [value.strip() for value in result.stdout.strip().split(",")]
            if len(values) == 4:
                gpu = {
                    "name": values[0],
                    "driver_version": values[1],
                    "total_vram_mb": int(values[2]),
                    "free_vram_mb": int(values[3]),
                    "visible_to_driver": True,
                }
        except (OSError, ValueError, subprocess.SubprocessError):
            gpu = None
    precision = "bf16" if cuda and torch.cuda.is_bf16_supported() else "fp16"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
        "device": "cuda" if cuda else "cpu",
        "precision": precision if cuda else "fp32",
        "gpu": gpu,
        "production_ready": cuda,
        "microscope_available": True,
        "setup_command": GPU_SETUP_COMMAND,
        "message": (
            "CUDA 가속이 준비되었습니다."
            if cuda
            else "NVIDIA GPU는 감지됐지만 현재 PyTorch가 CPU 빌드입니다."
        ),
    }

