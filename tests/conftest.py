"""
pytest configuration and fixtures.

Loads environment variables from .env before tests run.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"

if env_file.exists():
    load_dotenv(env_file)
