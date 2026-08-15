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

from typing import List, Union
from src.clinicaltrials.client import TrialSummary
from src.cleaning.models import CleanedTrial
from src.cleaning.normalization import parse_age


class EligibilityFilter:
    """
    Applies hard eligibility constraints (deterministic checks).

    Accepts both TrialSummary (from search) and CleanedTrial (from detail fetch).
    """

    def filter_by_hard_constraints(
        self, patient_age: int, patient_sex: str, candidate_trials: List[Union[TrialSummary, CleanedTrial]]
    ) -> List[str]:
        """
        Filter candidate trials based on age and sex constraints.

        Args:
            patient_age: Patient's age in years (integer)
            patient_sex: Patient's sex ("M", "F", or "All")
            candidate_trials: List of TrialSummary or CleanedTrial objects

        Returns:
            List of NCT IDs that pass age and sex checks
        """
        passing_nct_ids = []

        for trial in candidate_trials:
            # Check age constraints
            age_passes = True

            # Handle both TrialSummary (string ages like "18 Years") and CleanedTrial (integer ages)
            # Use 'is not None' to handle edge case where minimum_age_years=0 (peds trials from birth)
            min_age = getattr(trial, 'minimum_age_years', None) if getattr(trial, 'minimum_age_years', None) is not None else parse_age(getattr(trial, 'minimum_age', None))
            max_age = getattr(trial, 'maximum_age_years', None) if getattr(trial, 'maximum_age_years', None) is not None else parse_age(getattr(trial, 'maximum_age', None))

            if min_age is not None:
                age_passes = age_passes and (patient_age >= min_age)
            if max_age is not None:
                age_passes = age_passes and (patient_age <= max_age)

            if not age_passes:
                continue

            # Check sex constraints
            sex_passes = True
            if trial.sex and trial.sex.upper() != "ALL":
                # trial.sex is e.g. "M", "F", or "ALL"
                patient_sex_upper = patient_sex.upper() if patient_sex else "ALL"
                sex_passes = (patient_sex_upper == trial.sex.upper() or trial.sex.upper() == "ALL")

            if not sex_passes:
                continue

            # Both age and sex pass
            passing_nct_ids.append(trial.nct_id)

        return passing_nct_ids

