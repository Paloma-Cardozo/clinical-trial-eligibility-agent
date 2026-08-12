"""
Tests for data cleaning and normalization functions.

Covers normalization.py and eligibility_parser.py with pure unit tests
(no external API calls). LLM tests marked with @pytest.mark.llm and excluded by default.

Run with: pytest tests/test_cleaning.py
Run LLM tests only: pytest tests/test_cleaning.py -m llm
"""

import pytest
from datetime import date
from src.cleaning.normalization import parse_age, parse_date
from src.cleaning.eligibility_parser import parse_eligibility_criteria
from src.cleaning import clean_trial
from src.clinicaltrials.client import TrialDetail, Location


# ============================================================================
# Tests for parse_age()
# ============================================================================


class TestParseAge:
    """Unit tests for age normalization."""

    def test_parse_age_standard_format(self):
        """Standard format 'NN Years' should parse to integer."""
        assert parse_age("18 Years") == 18
        assert parse_age("65 Years") == 65
        assert parse_age("21 Years") == 21

    def test_parse_age_na(self):
        """'N/A' should return None."""
        assert parse_age("N/A") is None
        assert parse_age("n/a") is None

    def test_parse_age_none(self):
        """None input should return None."""
        assert parse_age(None) is None

    def test_parse_age_empty_string(self):
        """Empty string should return None."""
        assert parse_age("") is None
        assert parse_age("   ") is None

    def test_parse_age_malformed(self):
        """Unrecognized formats should return None gracefully; flexible on number position."""
        assert parse_age("unknown") is None
        # parse_age is permissive: extracts any number found, even if order is odd
        # This is intentional—handles API variations without crashing
        assert parse_age("Years 18") == 18  # Flexible: finds the number
        assert parse_age("abc xyz") is None  # No number to find


# ============================================================================
# Tests for parse_date()
# ============================================================================


class TestParseDate:
    """Unit tests for date normalization."""

    def test_parse_date_iso_format(self):
        """ISO format YYYY-MM-DD should parse to date object."""
        assert parse_date("2024-03-15") == date(2024, 3, 15)
        assert parse_date("2020-01-01") == date(2020, 1, 1)

    def test_parse_date_year_month_format(self):
        """Year-month format YYYY-MM should parse to first day of month."""
        assert parse_date("2024-03") == date(2024, 3, 1)
        assert parse_date("2020-12") == date(2020, 12, 1)

    def test_parse_date_na(self):
        """'N/A' should return None."""
        assert parse_date("N/A") is None
        assert parse_date("n/a") is None

    def test_parse_date_none(self):
        """None input should return None."""
        assert parse_date(None) is None

    def test_parse_date_empty_string(self):
        """Empty string should return None."""
        assert parse_date("") is None
        assert parse_date("   ") is None

    def test_parse_date_malformed(self):
        """Unrecognized formats should return None gracefully."""
        assert parse_date("unknown") is None
        assert parse_date("03/15/2024") is None  # US format, not recognized
        assert parse_date("invalid-date") is None


# ============================================================================
# Tests for parse_eligibility_criteria()
# ============================================================================


class TestParseEligibilityCriteria:
    """Unit tests for eligibility criteria parsing (regex + LLM fallback)."""

    def test_parse_criteria_standard_format(self):
        """Standard format with 'Inclusion Criteria:' / 'Exclusion Criteria:' headers."""
        text = """
        Inclusion Criteria:
        * Age 18 or older
        * Diagnosed with stage II-IV breast cancer
        * ECOG performance status 0-1

        Exclusion Criteria:
        * Pregnant or nursing
        * Prior chemotherapy within 12 months
        * Uncontrolled comorbidities
        """
        inclusion, exclusion = parse_eligibility_criteria(text)

        # Should find at least the main criteria via regex
        assert len(inclusion) >= 3, "Expected at least 3 inclusion criteria"
        assert len(exclusion) >= 3, "Expected at least 3 exclusion criteria"

    def test_parse_criteria_real_world_breast_cancer(self):
        """Real-world example from breast cancer trial (NCT05800275)."""
        # Actual criteria text from NCT05800275 trial
        text = """Inclusion Criteria:

* Patient ≥ 18 years of age at the time of study enrollment.
* Has histologically confirmed locally advanced or metastatic solid tumor malignancy

Exclusion Criteria:

* Has brain metastases (confirmed by imaging within 28 days of the first dose)
* Has a condition requiring systemic treatment with corticosteroids"""

        inclusion, exclusion = parse_eligibility_criteria(text)

        # Regex should handle this standard format
        assert len(inclusion) > 0, "Should parse inclusion criteria"
        assert len(exclusion) > 0, "Should parse exclusion criteria"

    def test_parse_criteria_none_input(self):
        """None input should return empty lists."""
        inclusion, exclusion = parse_eligibility_criteria(None)
        assert inclusion == []
        assert exclusion == []

    def test_parse_criteria_empty_string(self):
        """Empty string should return empty lists."""
        inclusion, exclusion = parse_eligibility_criteria("")
        assert inclusion == []
        assert exclusion == []

    def test_parse_criteria_no_standard_headers(self):
        """Text without standard headers falls back to LLM (or returns empty)."""
        text = """
        Must be 18 years old. Can't be pregnant.
        No prior cancer treatment allowed.
        """
        # This will attempt LLM fallback if no headers found
        # Should not crash, even if API key is missing
        inclusion, exclusion = parse_eligibility_criteria(text)

        # May return empty if LLM fails (no API key), or structured lists if LLM succeeds
        # Just verify it returns tuple of lists
        assert isinstance(inclusion, list), "Should return list for inclusion"
        assert isinstance(exclusion, list), "Should return list for exclusion"

    def test_parse_criteria_case_insensitive_headers(self):
        """Should find headers regardless of capitalization."""
        text = """
        INCLUSION CRITERIA:
        - Age 18+
        - Confirmed diagnosis

        EXCLUSION CRITERIA:
        - Pregnancy
        """
        inclusion, exclusion = parse_eligibility_criteria(text)

        assert len(inclusion) > 0, "Should find inclusion with uppercase header"
        assert len(exclusion) > 0, "Should find exclusion with uppercase header"

    @pytest.mark.llm
    def test_parse_criteria_non_standard_with_llm(self):
        """
        Test LLM fallback for non-standard format (requires GOOGLE_API_KEY).

        This test exercises the full LLM path when regex headers are not found.
        Marked @pytest.mark.llm to exclude from default test runs.
        Run with: pytest -m llm
        """
        text = """
        Patient Requirements:
        - Must be at least 18 years old
        - Confirmed diagnosis of advanced cancer
        - ECOG performance 0-1

        Cannot Participate If:
        - Pregnant or nursing
        - Prior chemotherapy within 12 months
        """

        inclusion, exclusion = parse_eligibility_criteria(text)

        # If LLM is available, should structure the text
        # If GOOGLE_API_KEY not set, returns ([], [])
        if inclusion or exclusion:
            # LLM succeeded: verify structure
            assert isinstance(inclusion, list), "inclusion should be a list"
            assert isinstance(exclusion, list), "exclusion should be a list"
            # Should have extracted at least some criteria
            assert len(inclusion) > 0, "Should extract some inclusion criteria"
            assert len(exclusion) > 0, "Should extract some exclusion criteria"
        else:
            # LLM not available (no API key); that's OK for CI environment
            pytest.skip("GOOGLE_API_KEY not configured; LLM fallback not tested")


# ============================================================================
# Tests for clean_trial() orchestrator
# ============================================================================


class TestCleanTrial:
    """Unit tests for clean_trial() orchestrator function."""

    def test_clean_trial_with_real_data(self):
        """
        Test clean_trial() with realistic TrialDetail data.

        Uses a sample breast cancer trial structure (NCT05800275 pattern).
        """
        # Create a TrialDetail with realistic data
        trial = TrialDetail(
            nct_id="NCT05800275",
            brief_title="Capecitabine, Tucatinib, and Intrathecal Trastuzumab",
            overall_status="RECRUITING",
            phase="PHASE2",
            condition="Breast Cancer; CNS Metastases",
            locations=[
                Location(
                    country="United States",
                    city="Boston",
                    state="Massachusetts",
                    facility="Massachusetts General Hospital",
                    status="RECRUITING",
                )
            ],
            eligibility_criteria="""Inclusion Criteria:

* Age 18 or older
* Histologically confirmed HER2-positive metastatic breast cancer
* Brain metastases confirmed by MRI

Exclusion Criteria:

* Pregnant or nursing
* Prior trastuzumab within 12 months
* Uncontrolled cardiac disease""",
            minimum_age="18 Years",
            maximum_age="75 Years",
            sex="ALL",
            healthy_volunteers=False,
            start_date="2023-03",
            completion_date="2025-12",
        )

        # Apply cleaning
        cleaned = clean_trial(trial)

        # Verify all fields are present and coherently populated
        assert cleaned.nct_id == "NCT05800275"
        assert cleaned.brief_title == trial.brief_title
        assert cleaned.overall_status == "RECRUITING"

        # Verify normalized age fields
        assert cleaned.minimum_age_years == 18, "minimum_age_years should be parsed integer"
        assert cleaned.maximum_age_years == 75, "maximum_age_years should be parsed integer"

        # Verify structured eligibility criteria
        assert len(cleaned.inclusion_criteria) > 0, "Should extract inclusion criteria"
        assert len(cleaned.exclusion_criteria) > 0, "Should extract exclusion criteria"
        # Verify key criteria are present
        assert any(
            "18" in crit.lower() for crit in cleaned.inclusion_criteria
        ), "Should extract age criterion from inclusion"

        # Verify normalized dates
        assert cleaned.start_date_parsed is not None, "start_date_parsed should be normalized"
        assert cleaned.completion_date_parsed is not None, "completion_date_parsed should be normalized"

        # Verify original fields are preserved for reference
        assert cleaned.minimum_age == "18 Years", "Original minimum_age preserved"
        assert cleaned.maximum_age == "75 Years", "Original maximum_age preserved"

    def test_clean_trial_with_none_fields(self):
        """Test clean_trial() handles None and missing fields gracefully."""
        trial = TrialDetail(
            nct_id="NCT00000123",
            brief_title="Test Trial",
            overall_status="RECRUITING",
            minimum_age=None,
            maximum_age=None,
            eligibility_criteria=None,
            start_date=None,
            completion_date=None,
        )

        cleaned = clean_trial(trial)

        # Should handle None gracefully
        assert cleaned.minimum_age_years is None
        assert cleaned.maximum_age_years is None
        assert cleaned.inclusion_criteria == []
        assert cleaned.exclusion_criteria == []
        assert cleaned.start_date_parsed is None
        assert cleaned.completion_date_parsed is None
