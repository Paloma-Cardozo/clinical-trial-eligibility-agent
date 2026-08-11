"""
TrialSearcher: Wrapper around the ClinicalTrials.gov public API.

Responsibility (to be implemented in Phase 1):
- Make GET requests to https://clinicaltrials.gov/api/v2/studies
- Filter for recruiting trials only (recruitment_status = "RECRUITING")
- Optimize queries using the 'fields' parameter to pull only needed data
- Handle pagination if needed
- Return raw API responses in a structured format ready for cleaning

This is NOT where data is cleaned; it's where data is fetched.
"""

from typing import List, Dict, Any


class TrialSearcher:
    """
    Client for querying the ClinicalTrials.gov public API.

    Methods (to be implemented):
    - search_by_condition_and_location(condition: str, location: str, max_results: int) -> List[Dict]
      Query trials matching a disease and geography, recruiting only.
    - fetch_trial_details(nct_id: str) -> Dict
      Get full details for a specific trial (NCT number).
    """

    def search_by_condition_and_location(
        self, condition: str, location: str, max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search ClinicalTrials.gov for trials matching condition and location.

        Args:
            condition: Disease/condition (e.g., "breast cancer", "melanoma")
            location: Geographic region (e.g., "Copenhagen", "Denmark", "United States")
            max_results: Maximum trials to return (default 20)

        Returns:
            List of trial records (raw API response, not yet cleaned)

        Implementation details (Phase 1):
        1. Build query parameters: condition, location, recruitment_status="RECRUITING"
        2. Use 'fields' parameter to request only: id, title, overallStatus, locations, recruitment
        3. Make GET request to BASE_URL
        4. Handle pagination if results > 100
        5. Return studies list
        """
        raise NotImplementedError("Search implemented in Phase 1")

    def fetch_trial_details(self, nct_id: str) -> Dict[str, Any]:
        """
        Fetch full details for a single trial by NCT ID.

        Args:
            nct_id: NCT number (e.g., "NCT00000123")

        Returns:
            Full trial record including eligibility criteria

        Implementation details (Phase 1):
        1. Query BASE_URL/{nct_id} with fields for eligibility, phase, results, etc.
        2. Return full record
        """
        raise NotImplementedError("Fetch implemented in Phase 1")
