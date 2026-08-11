"""
EligibilityFilter: Deterministic filtering based on hard constraints.

Responsibility (to be implemented in Phase 2):
- Apply hard constraints that can be checked with structured fields (no LLM needed)
- Hard constraints include:
  - Age: patient age vs trial's minimumAge/maximumAge
  - Sex: patient sex vs trial's gender restriction
  - healthyVolunteers: whether trial accepts healthy volunteers (patient disease status)

This is a critical optimization step: filter out obvious non-matches BEFORE
calling the LLM on soft constraints, to save API calls.
"""

from typing import Dict, Any, List


class EligibilityFilter:
    """
    Applies hard eligibility constraints (deterministic checks).

    Methods (to be implemented):
    - filter_by_hard_constraints(patient_age: int, patient_sex: str, candidate_trials: List[Dict]) -> List[str]
      Returns list of trial NCT IDs that pass hard constraint checks.
    """

    def filter_by_hard_constraints(
        self, patient_age: int, patient_sex: str, candidate_trials: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Filter candidate trials based on age and sex constraints.

        Args:
            patient_age: Patient's age in years
            patient_sex: Patient's sex ("M", "F", or "All")
            candidate_trials: List of trial records (with eligibilityModule fields)

        Returns:
            List of NCT IDs that pass age and sex checks

        Implementation details (Phase 2):
        1. For each trial, extract minimumAge, maximumAge from eligibilityModule.eligibility_criteria
        2. Check if patient_age falls within [min, max]
        3. Check if patient_sex matches trial's gender (if restricted)
        4. Append NCT ID to result list if both checks pass
        5. Return list of passing NCT IDs

        Note: Age parsing may need unit normalization (e.g., "18 Years" -> 18).
        """
        raise NotImplementedError("Hard constraint filtering implemented in Phase 2")
