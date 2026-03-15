"""Script to start the Streamlit UI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    """Start the Streamlit UI."""
    ui_path = Path(__file__).parent.parent / "src" / "ui" / "app.py"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ui_path),
            "--server.port=8501",
            "--browser.gatherUsageStats=false",
        ]
    )


if __name__ == "__main__":
    main()
