"""
TrialSearcher: Async client for ClinicalTrials.gov API v2.

This module wraps the public ClinicalTrials.gov /studies endpoint.
It is intentionally async to support parallel detail-fetching in Phase 3.

Design decisions:
- Manual retry logic with exponential backoff (no external retry libraries)
- Pydantic models for response validation against real API data
- Pagination is exposed, not hidden: caller controls whether to fetch next page
- Fields requested from the API are hardcoded (not parameterizable) to prevent
  the LLM (Phase 3) from requesting arbitrary fields
- Search uses simple API parameters (query.cond, query.locn), filters by status in code
"""

import httpx
import asyncio
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Exceptions
# ============================================================================

class ClinicalTrialsAPIError(Exception):
    """Raised when ClinicalTrials.gov API calls fail after all retries."""
    pass


# ============================================================================
# API Configuration
# ============================================================================

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# Fields to request from API v2. Using "Piece Names" (simple field names, not nested paths).
# These are hardcoded to prevent the LLM (Phase 3) from requesting arbitrary fields.
# Reference: ClinicalTrials.gov API v2 Study Data Structure documentation.

# Fields for search() — minimal set for listing and quick filtering
SEARCH_FIELDS = [
    "NCTId",
    "BriefTitle",
    "OverallStatus",
    "Phase",
    "Condition",
    "EligibilityCriteria",
    "HealthyVolunteers",
    "Sex",
    "MinimumAge",
    "MaximumAge",
    "LocationFacility",
    "LocationCity",
    "LocationState",
    "LocationCountry",
    "LocationStatus",
    "StartDate",
    "PrimaryCompletionDate",
]

# Fields for get_trial_details() — extends SEARCH_FIELDS with detailed study info
DETAIL_FIELDS = SEARCH_FIELDS + [
    "StudyType",
    "EnrollmentCount",
    "PrimaryOutcomeMeasure",
    "SecondaryOutcomeMeasure",
]

# Retry configuration
RETRY_ATTEMPTS = 3
RETRY_DELAYS = [1, 2, 4]  # seconds: 1s, 2s, 4s
REQUEST_TIMEOUT = 10  # seconds


# ============================================================================
# Response Models
# ============================================================================

class Location(BaseModel):
    """A trial's recruiting location."""
    country: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    facility: Optional[str] = None
    status: Optional[str] = None  # Individual location recruitment status


class TrialSummary(BaseModel):
    """Summary of a trial (for listing and quick filtering)."""
    nct_id: str
    brief_title: str
    overall_status: str
    phase: Optional[str] = None
    condition: Optional[str] = None
    locations: List[Location] = Field(default_factory=list)
    eligibility_criteria: Optional[str] = None
    minimum_age: Optional[str] = None
    maximum_age: Optional[str] = None
    sex: Optional[str] = None
    healthy_volunteers: Optional[bool] = None

    model_config = ConfigDict(populate_by_name=True)


class TrialDetail(TrialSummary):
    """Extended trial details (for deep inspection of a candidate trial).

    Extends TrialSummary with additional fields useful for eligibility reasoning.
    """
    enrollment: Optional[int] = None
    study_type: Optional[str] = None
    primary_outcomes: List[str] = Field(default_factory=list)
    secondary_outcomes: List[str] = Field(default_factory=list)
    start_date: Optional[str] = None
    completion_date: Optional[str] = None


class TrialSearchResult(BaseModel):
    """Paginated search results from ClinicalTrials.gov."""
    results: List[TrialSummary] = Field(default_factory=list)
    next_page_token: Optional[str] = None


# ============================================================================
# TrialSearcher Client
# ============================================================================

class TrialSearcher:
    """
    Async client for ClinicalTrials.gov API v2.

    Supports searching trials by condition/location and fetching detailed
    information for specific trials. All methods are async to enable
    parallelization in Phase 3.
    """

    def __init__(self):
        """Initialize the client (no state to maintain across calls)."""
        pass

    async def search(
        self,
        condition: str,
        location: Optional[str] = None,
        status: str = "RECRUITING",
        page_token: Optional[str] = None,
    ) -> TrialSearchResult:
        """
        Search for clinical trials matching condition, location, and recruitment status.

        Filters are applied server-side to minimize response payload and network usage.
        Uses API v2 parameters: query.cond, query.locn, filter.overallStatus.
        Requests only essential fields to further reduce bandwidth.

        Args:
            condition: Disease or condition to search for (e.g., "breast cancer")
            location: Optional geographic region/country to filter by
            status: Trial recruitment status (default: "RECRUITING"). Server-side filtered.
            page_token: Token for pagination (if fetching a subsequent page)

        Returns:
            TrialSearchResult with list of TrialSummary and next_page_token if more results exist

        Raises:
            ClinicalTrialsAPIError: If all retry attempts fail
        """
        # Build API parameters using simple search areas
        # query.cond — ConditionSearch area
        # query.locn — LocationSearch area (optional)
        # filter.overallStatus — recruitment status filter (server-side)
        params = {
            "query.cond": condition,
            "filter.overallStatus": status,
            "pageSize": 20,
            "fields": ",".join(SEARCH_FIELDS),
        }

        # Add location filter if provided
        if location:
            params["query.locn"] = location

        # Add pagination token if provided
        if page_token:
            params["pageToken"] = page_token

        # Retry logic
        for attempt in range(RETRY_ATTEMPTS):
            try:
                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                    response = await client.get(BASE_URL, params=params)
                    response.raise_for_status()
                    data = response.json()

                # Parse response
                results = []
                for study in data.get("studies", []):
                    # Extract nested fields following API v2 structure
                    protocol = study.get("protocolSection", {})
                    ident = protocol.get("identificationModule", {})
                    status_mod = protocol.get("statusModule", {})
                    eligibility = protocol.get("eligibilityModule", {})
                    design = protocol.get("designModule", {})
                    locations_mod = protocol.get("contactsLocationsModule", {})
                    conditions_mod = protocol.get("conditionsModule", {})

                    # Build trial summary
                    trial = TrialSummary(
                        nct_id=ident.get("nctId", ""),  # Fixed: nctId is in identificationModule
                        brief_title=ident.get("briefTitle", ""),
                        overall_status=status_mod.get("overallStatus", "UNKNOWN"),
                        phase=",".join(design.get("phases", [])) if design.get("phases") else None,
                        condition="; ".join(conditions_mod.get("conditions", [])) if conditions_mod.get("conditions") else None,
                        locations=[
                            Location(
                                country=loc.get("country"),
                                city=loc.get("city"),
                                state=loc.get("state"),
                                facility=loc.get("facility"),
                                status=loc.get("status"),
                            )
                            for loc in locations_mod.get("locations", [])
                        ],
                        eligibility_criteria=eligibility.get("eligibilityCriteria"),
                        minimum_age=eligibility.get("minimumAge"),
                        maximum_age=eligibility.get("maximumAge"),
                        sex=eligibility.get("sex"),
                        healthy_volunteers=eligibility.get("healthyVolunteers"),
                    )
                    results.append(trial)

                next_token = data.get("nextPageToken")
                return TrialSearchResult(results=results, next_page_token=next_token)

            except httpx.HTTPError as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    wait_time = RETRY_DELAYS[attempt]
                    await asyncio.sleep(wait_time)
                else:
                    raise ClinicalTrialsAPIError(
                        f"Failed to search trials after {RETRY_ATTEMPTS} attempts: {e}"
                    ) from e

    async def get_trial_details(self, nct_id: str) -> TrialDetail:
        """
        Fetch full details for a specific trial by NCT ID.

        Uses the dedicated /studies/{nctId} endpoint for direct lookup instead of
        searching, which is more efficient and semantically correct.

        Args:
            nct_id: NCT number (e.g., "NCT00000123")

        Returns:
            TrialDetail with extended information

        Raises:
            ClinicalTrialsAPIError: If all retry attempts fail or trial not found
        """
        # Use dedicated endpoint for direct trial lookup
        url = f"{BASE_URL}/{nct_id}"
        params = {
            "fields": ",".join(DETAIL_FIELDS),
        }

        for attempt in range(RETRY_ATTEMPTS):
            try:
                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()

                # Endpoint returns the study directly, not wrapped in a "studies" array
                study = data

                # Extract nested fields following API v2 structure
                protocol = study.get("protocolSection", {})
                ident = protocol.get("identificationModule", {})
                status_mod = protocol.get("statusModule", {})  # Fixed: statusModule is in protocolSection
                design = protocol.get("designModule", {})
                eligibility = protocol.get("eligibilityModule", {})
                locations_mod = protocol.get("contactsLocationsModule", {})
                conditions_mod = protocol.get("conditionsModule", {})
                outcomes_mod = protocol.get("outcomesModule", {})

                trial = TrialDetail(
                    nct_id=ident.get("nctId", ""),  # Fixed: nctId is in identificationModule
                    brief_title=ident.get("briefTitle", ""),
                    overall_status=status_mod.get("overallStatus", ""),
                    phase=",".join(design.get("phases", [])) if design.get("phases") else None,
                    condition="; ".join(conditions_mod.get("conditions", [])) if conditions_mod.get("conditions") else None,
                    locations=[
                        Location(
                            country=loc.get("country"),
                            city=loc.get("city"),
                            state=loc.get("state"),
                            facility=loc.get("facility"),
                            status=loc.get("status"),
                        )
                        for loc in locations_mod.get("locations", [])
                    ],
                    eligibility_criteria=eligibility.get("eligibilityCriteria"),
                    minimum_age=eligibility.get("minimumAge"),
                    maximum_age=eligibility.get("maximumAge"),
                    sex=eligibility.get("sex"),
                    healthy_volunteers=eligibility.get("healthyVolunteers"),
                    enrollment=design.get("enrollmentInfo", {}).get("count"),
                    study_type=design.get("studyType"),
                    primary_outcomes=[
                        outcome.get("measure", "")
                        for outcome in outcomes_mod.get("primaryOutcomes", [])
                    ],
                    secondary_outcomes=[
                        outcome.get("measure", "")
                        for outcome in outcomes_mod.get("secondaryOutcomes", [])
                    ],
                    start_date=status_mod.get("startDateStruct", {}).get("date"),
                    completion_date=status_mod.get("primaryCompletionDateStruct", {}).get("date"),
                )
                return trial

            except httpx.HTTPError as e:
                if attempt < RETRY_ATTEMPTS - 1:
                    wait_time = RETRY_DELAYS[attempt]
                    await asyncio.sleep(wait_time)
                else:
                    raise ClinicalTrialsAPIError(
                        f"Failed to fetch trial {nct_id} after {RETRY_ATTEMPTS} attempts: {e}"
                    ) from e
