from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"
HEALTH_URL = f"{URL}/api/health"


def _port_is_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.25)
        return connection.connect_ex((HOST, PORT)) == 0


def _health_is_ready() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=0.5) as response:
            return response.status == 200
    except OSError:
        return False


def _open_browser_when_ready() -> None:
    if os.environ.get("TRANSFORMER_NO_BROWSER") == "1":
        return

    for _ in range(120):
        if _health_is_ready():
            webbrowser.open(URL)
            return
        time.sleep(0.25)


def main() -> int:
    if _port_is_open():
        if _health_is_ready():
            print(f"Transformer Learning Studio is already running at {URL}")
            if os.environ.get("TRANSFORMER_NO_BROWSER") != "1":
                webbrowser.open(URL)
            return 0
        print(
            f"Port {PORT} is already used by another program. "
            "Close that program or change the port.",
            file=sys.stderr,
        )
        return 1

    browser_thread = threading.Thread(
        target=_open_browser_when_ready,
        name="open-transformer-studio",
        daemon=True,
    )
    browser_thread.start()

    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
