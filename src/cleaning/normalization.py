"""
Normalization functions for clinical trial data.

Reusable, testable functions that clean and parse structured but non-normalized
fields from TrialSummary/TrialDetail objects.
"""

import re
from typing import Optional
from datetime import date


def parse_age(age_str: Optional[str]) -> Optional[int]:
    """
    Parse age string into integer years.

    Handles common formats from ClinicalTrials.gov API:
    - "18 Years"
    - "65 Years"
    - "N/A"
    - None
    - Empty string

    Args:
        age_str: Age string from TrialSummary.minimum_age or maximum_age

    Returns:
        Integer age in years, or None if cannot parse or input is None/"N/A"/empty
    """
    if not age_str or age_str.upper() == "N/A":
        return None

    # Try to extract leading integer from string like "18 Years"
    match = re.search(r"(\d+)", age_str)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, AttributeError):
            return None

    return None


def parse_date(date_str: Optional[str]) -> Optional[date]:
    """
    Parse date string into Python date object.

    Handles common formats from ClinicalTrials.gov API:
    - "2024-03-15" (ISO, YYYY-MM-DD)
    - "2024-03" (year-month, YYYY-MM)
    - "N/A"
    - None
    - Empty string

    Args:
        date_str: Date string from TrialSummary.start_date or completion_date

    Returns:
        Python date object, or None if cannot parse or input is None/"N/A"/empty
    """
    if not date_str or date_str.upper() == "N/A":
        return None

    # Try YYYY-MM-DD format
    try:
        return date.fromisoformat(date_str)
    except (ValueError, TypeError):
        pass

    # Try YYYY-MM format (append -01 for day)
    match = re.match(r"(\d{4})-(\d{2})$", date_str)
    if match:
        try:
            year = int(match.group(1))
            month = int(match.group(2))
            return date(year, month, 1)
        except (ValueError, TypeError):
            return None

    # Unrecognized format
    return None
