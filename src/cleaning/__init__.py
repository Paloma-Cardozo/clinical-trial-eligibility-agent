from typing import Union
from .eligibility import EligibilityFilter
from .reasoning import EligibilityReasoner
from .models import CleanedTrial
from .normalization import parse_age, parse_date
from .eligibility_parser import parse_eligibility_criteria
from src.clinicaltrials.client import TrialSummary, TrialDetail


def clean_trial(trial: Union[TrialSummary, TrialDetail]) -> CleanedTrial:
    """
    Orchestrate cleaning of a trial by applying all normalization functions.

    Converts a TrialSummary or TrialDetail from the API into a CleanedTrial with
    normalized age, date, and eligibility criteria fields. Works with both search
    results (TrialSummary, which lack start/completion dates) and detail fetches
    (TrialDetail, which include them).

    Args:
        trial: TrialSummary or TrialDetail object from API

    Returns:
        CleanedTrial with parsed and structured fields
    """
    # Parse eligibility criteria into inclusion/exclusion lists
    inclusion_criteria, exclusion_criteria = parse_eligibility_criteria(
        trial.eligibility_criteria
    )

    # Parse age strings to integers
    minimum_age_years = parse_age(trial.minimum_age)
    maximum_age_years = parse_age(trial.maximum_age)

    # Parse date strings to date objects (use getattr to handle TrialSummary which lacks these fields)
    start_date_parsed = parse_date(getattr(trial, "start_date", None))
    completion_date_parsed = parse_date(getattr(trial, "completion_date", None))

    # Create CleanedTrial with all normalized fields
    return CleanedTrial(
        # Inherited from TrialSummary
        nct_id=trial.nct_id,
        brief_title=trial.brief_title,
        overall_status=trial.overall_status,
        phase=trial.phase,
        condition=trial.condition,
        locations=trial.locations,
        eligibility_criteria=trial.eligibility_criteria,
        minimum_age=trial.minimum_age,
        maximum_age=trial.maximum_age,
        sex=trial.sex,
        healthy_volunteers=trial.healthy_volunteers,
        # Cleaned fields (added by CleanedTrial)
        minimum_age_years=minimum_age_years,
        maximum_age_years=maximum_age_years,
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        start_date_parsed=start_date_parsed,
        completion_date_parsed=completion_date_parsed,
    )


__all__ = [
    "EligibilityFilter",
    "EligibilityReasoner",
    "CleanedTrial",
    "clean_trial",
    "parse_age",
    "parse_date",
    "parse_eligibility_criteria",
]
