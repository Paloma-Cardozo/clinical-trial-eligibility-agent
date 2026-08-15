"""
Eligibility criteria parser with hybrid regex+LLM approach.

Strategy:
1. First attempt: use regex to find standard "Inclusion Criteria:" / "Exclusion Criteria:"
   headers and split text into structured lists
2. Fallback: if regex doesn't find headers, use Google Gemini API REST to structure the text

This ensures fast deterministic parsing for well-formatted criteria while
handling edge cases that don't follow the standard format.
"""

import re
import os
import json
import logging
from typing import Tuple, List, Optional
import httpx
from src.config import load_api_keys, GEMINI_TIMEOUT

logger = logging.getLogger(__name__)


def parse_eligibility_criteria(text: Optional[str]) -> Tuple[List[str], List[str]]:
    """
    Parse eligibility criteria text into inclusion and exclusion lists.

    Args:
        text: Free-text eligibility criteria from TrialSummary.eligibility_criteria

    Returns:
        Tuple of (inclusion_criteria: List[str], exclusion_criteria: List[str])
        Each list contains individual criteria as separate strings (bullet points).
        Returns ([], []) if text is None or empty.
    """
    if not text or not text.strip():
        return [], []

    # First attempt: regex-based parsing for standard format
    inclusion, exclusion = _parse_with_regex(text)
    if inclusion or exclusion:
        return inclusion, exclusion

    # Fallback: use LLM to structure non-standard text
    return _parse_with_llm(text)


def _parse_with_regex(text: str) -> Tuple[List[str], List[str]]:
    """
    Parse eligibility criteria using regex for standard format.

    Looks for headers like "Inclusion Criteria:", "Exclusion Criteria:" (tolerant
    of whitespace/capitalization) and splits text into lists of criteria.

    Returns:
        Tuple of (inclusion, exclusion). Returns ([], []) if headers not found.
    """
    # Normalize text: lowercase for search, but keep original for extraction
    text_lower = text.lower()

    inclusion = []
    exclusion = []

    # Find "Inclusion Criteria:" section
    inclusion_match = re.search(
        r"inclusion\s+criteria\s*:(.+?)(?=exclusion\s+criteria|$)",
        text_lower,
        re.IGNORECASE | re.DOTALL,
    )
    if inclusion_match:
        inclusion_text = inclusion_match.group(1)
        inclusion = _extract_criteria_list(inclusion_text)

    # Find "Exclusion Criteria:" section
    exclusion_match = re.search(
        r"exclusion\s+criteria\s*:(.+?)$",
        text_lower,
        re.IGNORECASE | re.DOTALL,
    )
    if exclusion_match:
        exclusion_text = exclusion_match.group(1)
        exclusion = _extract_criteria_list(exclusion_text)

    return inclusion, exclusion


def _extract_criteria_list(section_text: str) -> List[str]:
    """
    Extract individual criteria from a section (inclusion or exclusion).

    Splits on bullet points (*), dashes (-), or numbered lists (1., 2., etc).
    Cleans and filters empty lines.
    """
    # Split on common bullet markers
    criteria = re.split(r"[*\-•]\s*", section_text)

    # Clean each criterion: strip whitespace, remove empty lines
    cleaned = []
    for criterion in criteria:
        clean = criterion.strip()
        if clean and len(clean) > 2:  # Skip very short lines (likely noise)
            cleaned.append(clean)

    return cleaned


def _parse_with_llm(text: str) -> Tuple[List[str], List[str]]:
    """
    Parse eligibility criteria using Google Gemini API REST (fallback for non-standard format).

    Makes a single, simple LLM call to structure the criteria text.
    Standalone utility, not part of the agent loop.
    Uses unified load_api_keys() to be consistent with orchestrator and reasoning modules.
    """
    api_keys = load_api_keys()
    if not api_keys:
        # If no API key, return empty
        return [], []

    api_key = api_keys[0]  # Use first available key

    prompt = f"""Extract inclusion and exclusion criteria from this trial eligibility text.
Return a simple JSON object with "inclusion" and "exclusion" keys, each containing a list of criteria.

Text:
{text}

Example output:
{{"inclusion": ["Age 18-75", "Diagnosed with cancer"], "exclusion": ["Pregnant", "Prior chemotherapy"]}}

Return ONLY valid JSON, no markdown or extra text."""

    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": api_key,
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        response = httpx.post(url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        # Extract text from Google Gemini API response
        response_text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Parse JSON response
        criteria_data = json.loads(response_text)
        inclusion = criteria_data.get("inclusion", [])
        exclusion = criteria_data.get("exclusion", [])

        # Ensure lists
        if not isinstance(inclusion, list):
            inclusion = []
        if not isinstance(exclusion, list):
            exclusion = []

        return inclusion, exclusion

    except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.debug(f"LLM fallback failed, returning empty lists: {type(e).__name__}")
        return [], []
