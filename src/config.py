"""
Centralized configuration for the Clinical Trial Eligibility Agent.

This module contains shared constants and utilities used across multiple modules.
By centralizing here, we avoid duplication and make maintenance easier.
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Model fallback chain for Gemini API
# Used in orchestrator and reasoning modules
MODELS_FALLBACK = [
    "gemini-3.7-flash",  # Primary (highest RPM)
    "gemini-3.6-flash",  # Fallback (when 3.7 is unavailable)
]

# API timeouts (seconds) for different services
# ClinicalTrials.gov API is fast and responsive
CLINICALTRIALS_TIMEOUT = 10.0
# Gemini API is slower (includes model inference)
GEMINI_TIMEOUT = 30.0

# Maximum conversation history elements before compression
MAX_CONVERSATION_HISTORY = 20


def load_api_keys() -> List[str]:
    """
    Load all available API keys from environment.

    Strategy:
    1. Try numbered keys: GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.
    2. Fallback to single key: GOOGLE_API_KEY

    This unified function ensures all modules (orchestrator, reasoning, eligibility_parser)
    use the same key loading logic for consistency.
    """
    keys = []
    i = 1
    while True:
        key = os.environ.get(f"GOOGLE_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1

    # Fallback to single GOOGLE_API_KEY if no numbered keys found
    if not keys:
        key = os.environ.get("GOOGLE_API_KEY")
        if key:
            keys.append(key)

    return keys
