"""
Facial Mapping Server
=====================
Entry point for the facial mapping application.
Starts a local HTTP server and opens the browser automatically.

Requirements (Python 3.13):
    - flask==3.1.0
    - flask-cors==5.0.0
    - opencv-python==4.10.0.84
    - mediapipe==0.10.18
    - numpy==2.2.4
    - Pillow==11.1.0

Install via:
    pip install flask flask-cors opencv-python mediapipe numpy Pillow
"""

import webbrowser
import threading
import os
from app import create_app


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 5050


def open_browser():
    """Open the default browser after a short delay to let Flask start."""
    webbrowser.open(f"http://{HOST}:{PORT}")


def main():
    app = create_app()

    print("=" * 55)
    print("  Facial Mapping Application")
    print("=" * 55)
    print(f"  Server : http://{HOST}:{PORT}")
    print("  Press  : Ctrl+C to stop")
    print("=" * 55)

    # Open the browser slightly after the server starts
    timer = threading.Timer(1.2, open_browser)
    timer.daemon = True
    timer.start()

    app.run(host=HOST, port=PORT, debug=False)


if __name__ == "__main__":
    main()