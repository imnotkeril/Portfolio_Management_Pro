"""Application entry point for running Streamlit app."""

import subprocess
import sys

if __name__ == "__main__":
    # Run Streamlit app
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app/app.py",
            "--server.headless",
            "true",
        ]
    )

