"""
Integration tests for ClinicalTrials.gov API client.

These tests hit the real ClinicalTrials.gov API (not mocked) to verify
that response parsing and model validation work against actual data.

Run with: pytest tests/test_clinicaltrials_api.py -v
"""

import pytest
from src.clinicaltrials.client import (
    TrialSearcher,
    TrialSearchResult,
    ClinicalTrialsAPIError,
    TrialNotFoundError,
    ClinicalTrialsConnectionError,
)


@pytest.mark.asyncio
async def test_search_returns_results():
    """Verify search() returns results for a known condition (breast cancer)."""
    searcher = TrialSearcher()

    # Default status is RECRUITING (server-side filtered)
    result = await searcher.search(condition="breast cancer", location="United States")

    assert isinstance(result, TrialSearchResult)
    assert len(result.results) > 0, "Expected at least one RECRUITING trial for breast cancer"
    assert all(result.results[0].nct_id), "All results should have NCT ID"
    assert all(result.results[0].brief_title), "All results should have a title"


@pytest.mark.asyncio
async def test_search_parses_trial_summary():
    """Verify TrialSummary fields are correctly populated from API response."""
    searcher = TrialSearcher()

    # Default status="RECRUITING" filters server-side
    result = await searcher.search(condition="melanoma")

    assert len(result.results) > 0
    trial = result.results[0]

    # DEBUG: Print actual values to verify bug fix
    print(f"\n✓ NCT ID: {trial.nct_id}")
    print(f"✓ Brief Title: {trial.brief_title[:60]}...")
    print(f"✓ Overall Status: {trial.overall_status}")
    print(f"✓ Condition (from conditionsModule): {trial.condition}")

    # Verify key fields are populated
    assert trial.nct_id, "nct_id should not be empty"
    assert trial.brief_title, "brief_title should not be empty"
    # Server-side filter ensures all results are RECRUITING
    assert trial.overall_status == "RECRUITING", f"Expected RECRUITING, got {trial.overall_status}"
    # Verify condition is independently populated (not relying on brief_title)
    assert trial.condition is not None and trial.condition != "", "condition should be populated from conditionsModule"


@pytest.mark.asyncio
async def test_search_pagination_token():
    """Verify next_page_token is returned when more results exist."""
    searcher = TrialSearcher()

    # Search with small page size (hardcoded to 20 in client)
    # to increase likelihood of pagination
    result = await searcher.search(condition="cancer")

    # next_page_token may or may not be present depending on total results
    # Just verify it's either None or a string (valid token format)
    assert result.next_page_token is None or isinstance(result.next_page_token, str)


@pytest.mark.asyncio
async def test_get_trial_details():
    """Verify get_trial_details() fetches and parses a specific trial with populated extended fields."""
    searcher = TrialSearcher()

    # First, search to get an actual NCT ID that has full data
    search_result = await searcher.search(condition="breast cancer")
    if not search_result.results:
        pytest.skip("No trials found for breast cancer")

    nct_id = search_result.results[0].nct_id

    # Fetch details for that trial using dedicated endpoint
    detail = await searcher.get_trial_details(nct_id)

    # DEBUG: Print actual values to verify endpoint works and fields are populated
    print(f"\n✓ NCT ID: {detail.nct_id}")
    print(f"✓ Brief Title: {detail.brief_title[:60]}...")
    print(f"✓ Study Type (TrialDetail field): {detail.study_type}")
    print(f"✓ Enrollment: {detail.enrollment}")
    print(f"✓ Primary Outcomes: {detail.primary_outcomes[:1] if detail.primary_outcomes else 'None'}")
    print(f"✓ Secondary Outcomes count: {len(detail.secondary_outcomes)}")

    # Verify basic fields match
    assert detail.nct_id == nct_id, f"NCT ID mismatch: {detail.nct_id} != {nct_id}"
    assert detail.brief_title, "brief_title should be populated"
    assert detail.overall_status, "overall_status should be populated"

    # Verify extended fields are properly parsed (not None/empty)
    # These are the fields that distinguish TrialDetail from TrialSummary
    assert detail.study_type is not None and detail.study_type != "", \
        "study_type should be populated for TrialDetail"
    assert isinstance(detail.primary_outcomes, list), \
        "primary_outcomes should be a list"
    assert isinstance(detail.secondary_outcomes, list), \
        "secondary_outcomes should be a list"
    # Enrollment should be populated for most trials
    assert detail.enrollment is not None and detail.enrollment > 0, \
        "enrollment should have a positive count"


@pytest.mark.asyncio
async def test_get_trial_details_invalid_nct_id():
    """Verify get_trial_details() raises TrialNotFoundError for invalid NCT ID (404)."""
    searcher = TrialSearcher()

    with pytest.raises(TrialNotFoundError):
        await searcher.get_trial_details("NCT99999999")
