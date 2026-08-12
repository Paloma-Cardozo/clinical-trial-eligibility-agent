"""
EligibilityFilter: Deterministic filtering based on hard constraints.

Responsibility:
- Apply hard constraints that can be checked with structured fields (no LLM needed)
- Will be integrated into the agent loop in Phase 4

Hard constraints include:
  - Age: patient age vs trial's minimum_age_years/maximum_age_years
  - Sex: patient sex vs trial's sex field
  - healthyVolunteers: whether trial accepts healthy volunteers

This is a critical optimization step: filter out obvious non-matches BEFORE
calling the LLM on soft constraints, to save API calls.
"""

from typing import List
from src.cleaning.models import CleanedTrial


class EligibilityFilter:
    """
    Applies hard eligibility constraints (deterministic checks).

    Methods (to be implemented):
    - filter_by_hard_constraints(patient_age: int, patient_sex: str, candidate_trials: List[CleanedTrial]) -> List[str]
      Returns list of trial NCT IDs that pass hard constraint checks.
    """

    def filter_by_hard_constraints(
        self, patient_age: int, patient_sex: str, candidate_trials: List[CleanedTrial]
    ) -> List[str]:
        """
        Filter candidate trials based on age and sex constraints.

        Args:
            patient_age: Patient's age in years (integer)
            patient_sex: Patient's sex ("M", "F", or "All")
            candidate_trials: List of CleanedTrial objects (already normalized by clean_trial())

        Returns:
            List of NCT IDs that pass age and sex checks

        Implementation details:
        1. For each trial, read minimum_age_years and maximum_age_years (already parsed integers)
        2. Check if patient_age falls within [minimum_age_years, maximum_age_years]
           (handling None values as "no restriction")
        3. Read sex field; check if patient_sex matches trial's gender restriction
        4. Optionally check healthyVolunteers flag if patient disease status affects eligibility
        5. Append nct_id to result list if all checks pass
        6. Return list of passing NCT IDs

        Note: All fields are already normalized (ages as integers, dates as date objects, etc.)
        by the clean_trial() function in src/cleaning/__init__.py. No parsing needed here.
        """
        raise NotImplementedError("filter_by_hard_constraints not yet implemented")
