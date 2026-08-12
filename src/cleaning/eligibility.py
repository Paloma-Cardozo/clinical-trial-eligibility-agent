"""
EligibilityFilter: Deterministic filtering based on hard constraints.

Responsibility (implemented in Phase 2, integrated into agent loop in Phase 4):
- Apply hard constraints that can be checked with structured fields (no LLM needed)
- Hard constraints include:
  - Age: patient age vs trial's minimum_age/maximum_age (already structured in TrialSummary)
  - Sex: patient sex vs trial's sex field (already structured in TrialSummary)
  - healthyVolunteers: whether trial accepts healthy volunteers (already structured in TrialSummary)

This is a critical optimization step: filter out obvious non-matches BEFORE
calling the LLM on soft constraints, to save API calls.

Design: Phase 2 implements the filter logic; Phase 4 integrates it into the
agent's orchestrator loop so hard constraints are checked after search but
before LLM-based soft constraint reasoning.
"""

from typing import List
from src.clinicaltrials.client import TrialSummary


class EligibilityFilter:
    """
    Applies hard eligibility constraints (deterministic checks).

    Methods (to be implemented):
    - filter_by_hard_constraints(patient_age: int, patient_sex: str, candidate_trials: List[TrialSummary]) -> List[str]
      Returns list of trial NCT IDs that pass hard constraint checks.
    """

    def filter_by_hard_constraints(
        self, patient_age: int, patient_sex: str, candidate_trials: List[TrialSummary]
    ) -> List[str]:
        """
        Filter candidate trials based on age and sex constraints.

        Args:
            patient_age: Patient's age in years (integer)
            patient_sex: Patient's sex ("M", "F", or "All")
            candidate_trials: List of TrialSummary objects from search results

        Returns:
            List of NCT IDs that pass age and sex checks

        Implementation details (Phase 2):
        1. For each trial, read minimum_age and maximum_age (structured but NOT normalized;
           format is strings like "18 Years", "65 Years", "N/A", etc.)
        2. Parse minimum_age/maximum_age strings to extract integer years (e.g., "18 Years" -> 18)
        3. Check if patient_age falls within [parsed_min, parsed_max]
        4. Read sex field; check if patient_sex matches trial's gender restriction
        5. Optionally check healthyVolunteers flag if patient disease status affects eligibility
        6. Append nct_id to result list if all checks pass
        7. Return list of passing NCT IDs

        Note: Fields are already structured (not free-text), but require parsing
        (e.g., age normalization) before comparison.
        """
        raise NotImplementedError("Hard constraint filtering implemented in Phase 2")
