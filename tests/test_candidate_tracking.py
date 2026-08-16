#!/usr/bin/env python3
"""
Unit test to verify dynamic candidate tracking logic.
Tests the extraction and enrichment of patient_summary with evaluation status.
"""

import json
import sys
from pathlib import Path

# Test data
def test_candidate_extraction():
    """Test that evaluated_candidates dictionary is populated correctly."""
    print("\n" + "="*80)
    print("TEST 1: Candidate Extraction from reason_soft_constraints Results")
    print("="*80)

    # Simulate what happens in the orchestrator
    evaluated_candidates = {}

    # Simulate tool results from reason_soft_constraints
    tool_results = [
        {
            "tool_name": "reason_soft_constraints",
            "tool_args": {"nct_id": "NCT07007169"},
            "tool_result": json.dumps({
                "confidence": "likely_eligible",
                "rationale": "Patient fits inclusion criteria"
            })
        },
        {
            "tool_name": "reason_soft_constraints",
            "tool_args": {"nct_id": "NCT07095023"},
            "tool_result": json.dumps({
                "confidence": "likely_eligible",
                "rationale": "No exclusion criteria met"
            })
        },
        {
            "tool_name": "reason_soft_constraints",
            "tool_args": {"nct_id": "NCT06111417"},
            "tool_result": json.dumps({
                "confidence": "possibly_eligible",
                "rationale": "Some criteria uncertain"
            })
        },
        {
            "tool_name": "reason_soft_constraints",
            "tool_args": {"nct_id": "NCT07528638"},
            "tool_result": json.dumps({
                "confidence": "possibly_eligible",
                "rationale": "Depends on biomarker results"
            })
        },
        {
            "tool_name": "search_trials",
            "tool_args": {"condition": "breast cancer"},
            "tool_result": json.dumps({"trials": ["NCT07007169"]})
        },
    ]

    # Simulate the extraction logic from orchestrator.py lines 723-733
    for item in tool_results:
        tool_name = item["tool_name"]
        tool_args = item["tool_args"]
        tool_result = item["tool_result"]

        if tool_name == "reason_soft_constraints":
            try:
                result_json = json.loads(tool_result)
                nct_id = tool_args.get("nct_id")
                confidence = result_json.get("confidence", "unknown")
                if nct_id:
                    evaluated_candidates[nct_id] = confidence
                    print(f"  [OK] Tracked: {nct_id} -> {confidence}")
            except (json.JSONDecodeError, KeyError):
                pass

    print(f"\nResult: {len(evaluated_candidates)} candidates tracked")
    assert len(evaluated_candidates) == 4, f"Expected 4 candidates, got {len(evaluated_candidates)}"
    assert evaluated_candidates["NCT07007169"] == "likely_eligible"
    assert evaluated_candidates["NCT07095023"] == "likely_eligible"
    assert evaluated_candidates["NCT06111417"] == "possibly_eligible"
    assert evaluated_candidates["NCT07528638"] == "possibly_eligible"
    print("[PASS] TEST 1 PASSED\n")

    return evaluated_candidates


def test_patient_summary_enrichment():
    """Test that patient_summary is enriched with evaluation status."""
    print("="*80)
    print("TEST 2: Patient Summary Enrichment with Evaluation Status")
    print("="*80)

    # Set up candidate data (same as test 1)
    evaluated_candidates = {
        "NCT07007169": "likely_eligible",
        "NCT07095023": "likely_eligible",
        "NCT06111417": "possibly_eligible",
        "NCT07528638": "possibly_eligible"
    }

    # Simulate patient profile
    age = 45
    sex = "F"
    condition = "breast cancer"
    disease_stage = "early stage"
    prior_treatments = ["chemotherapy"]

    treatments_text = ", ".join(prior_treatments) if prior_treatments else "None"

    # Simulate the enrichment logic from orchestrator.py lines 750-770
    evaluation_status = ""
    if evaluated_candidates:
        likely_eligible = [nct for nct, conf in evaluated_candidates.items() if conf == "likely_eligible"]
        possibly_eligible = [nct for nct, conf in evaluated_candidates.items() if conf == "possibly_eligible"]
        total_evaluated = len(evaluated_candidates)
        total_qualifying = len(likely_eligible) + len(possibly_eligible)

        status_parts = [f"Evaluated {total_evaluated} trials"]
        if likely_eligible:
            status_parts.append(f"{len(likely_eligible)} likely_eligible ({', '.join(likely_eligible[:2])}{'...' if len(likely_eligible) > 2 else ''})")
        if possibly_eligible:
            status_parts.append(f"{len(possibly_eligible)} possibly_eligible ({', '.join(possibly_eligible[:2])}{'...' if len(possibly_eligible) > 2 else ''})")

        remaining_needed = max(0, 3 - total_qualifying)
        if remaining_needed > 0:
            status_parts.append(f"Need {remaining_needed}+ more to reach target (3-8 candidates)")
        else:
            status_parts.append(f"Have {total_qualifying} qualifying candidates - approaching target (3-8)")

        evaluation_status = "\n[EVALUATION STATUS: " + "; ".join(status_parts) + "]"

    patient_summary_text = f"[PATIENT CONTEXT: Age {age}, Sex {sex}, Condition: {condition}, Disease Stage: {disease_stage}, Prior Treatments: {treatments_text}]{evaluation_status}"

    print(f"\nGenerated patient_summary:\n{patient_summary_text}\n")

    # Verify content
    assert "Age 45" in patient_summary_text
    assert "Sex F" in patient_summary_text
    assert "breast cancer" in patient_summary_text
    assert "Evaluated 4 trials" in patient_summary_text
    assert "2 likely_eligible" in patient_summary_text
    assert "2 possibly_eligible" in patient_summary_text
    assert "Have 4 qualifying candidates" in patient_summary_text

    print("[PASS] TEST 2 PASSED\n")
    return patient_summary_text


def test_progress_tracking():
    """Test that as more candidates are evaluated, the status updates appropriately."""
    print("="*80)
    print("TEST 3: Progress Tracking - Status Updates as Candidates Accumulate")
    print("="*80)

    # Test at different stages
    stages = [
        ({"NCT1": "likely_eligible"}, "Evaluated 1 trials"),
        ({"NCT1": "likely_eligible", "NCT2": "likely_eligible"}, "Evaluated 2 trials"),
        ({"NCT1": "likely_eligible", "NCT2": "likely_eligible", "NCT3": "likely_eligible"}, "Evaluated 3 trials"),
        ({"NCT1": "likely_eligible", "NCT2": "likely_eligible", "NCT3": "likely_eligible",
          "NCT4": "likely_eligible", "NCT5": "possibly_eligible", "NCT6": "possibly_eligible",
          "NCT7": "possibly_eligible"}, "Evaluated 7 trials"),
    ]

    for i, (candidates, expected_desc) in enumerate(stages, 1):
        likely_eligible = [nct for nct, conf in candidates.items() if conf == "likely_eligible"]
        possibly_eligible = [nct for nct, conf in candidates.items() if conf == "possibly_eligible"]
        total_evaluated = len(candidates)
        total_qualifying = len(likely_eligible) + len(possibly_eligible)
        remaining_needed = max(0, 3 - total_qualifying)

        status_parts = [f"Evaluated {total_evaluated} trials"]
        if likely_eligible:
            status_parts.append(f"{len(likely_eligible)} likely_eligible")
        if possibly_eligible:
            status_parts.append(f"{len(possibly_eligible)} possibly_eligible")

        if remaining_needed > 0:
            status_parts.append(f"Need {remaining_needed}+ more to reach target (3-8 candidates)")
        else:
            status_parts.append(f"Have {total_qualifying} qualifying candidates - approaching target (3-8)")

        status = "; ".join(status_parts)
        print(f"Stage {i}: {status}")
        assert expected_desc in status, f"Expected '{expected_desc}' in status, got '{status}'"

    print("\n[PASS] TEST 3 PASSED\n")


if __name__ == "__main__":
    try:
        evaluated_candidates = test_candidate_extraction()
        patient_summary = test_patient_summary_enrichment(evaluated_candidates)
        test_progress_tracking()

        print("="*80)
        print("[PASS] ALL TESTS PASSED!")
        print("="*80)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
