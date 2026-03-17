"""Script to start the Streamlit UI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    """Start the Streamlit UI."""
    ui_path = Path(__file__).parent.parent / "src" / "ui" / "app.py"

    try:
        from src.core.config import load_settings

        settings = load_settings()
        port = settings.ui.port
    except Exception:
        port = 8502

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ui_path),
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ]
    )


if __name__ == "__main__":
    main()
