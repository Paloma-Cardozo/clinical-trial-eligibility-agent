"""
Cleaned trial data model.

CleanedTrial extends TrialDetail with normalized (parsed and cleaned) fields.
"""

from datetime import date
from typing import List, Optional
from pydantic import Field
from src.clinicaltrials.client import TrialDetail


class CleanedTrial(TrialDetail):
    """
    Trial summary with normalized (parsed and structured) fields.

    Extends TrialSummary by adding cleaned versions of ambiguous or
    free-text fields from the original API response.

    Fields added:
    - minimum_age_years: parsed integer age (from minimum_age string like "18 Years")
    - maximum_age_years: parsed integer age (from maximum_age string like "65 Years")
    - inclusion_criteria: structured list of inclusion criteria (parsed from free text)
    - exclusion_criteria: structured list of exclusion criteria (parsed from free text)
    - start_date_parsed: normalized date object (from start_date string like "2024-03")
    - completion_date_parsed: normalized date object (from completion_date string)

    Original fields (minimum_age, maximum_age, eligibility_criteria, start_date,
    completion_date) are inherited unchanged for reference/debugging.
    """

    minimum_age_years: Optional[int] = None
    maximum_age_years: Optional[int] = None
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    start_date_parsed: Optional[date] = None
    completion_date_parsed: Optional[date] = None
