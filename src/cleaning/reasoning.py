"""
EligibilityReasoner: LLM-based reasoning over soft constraints.

Responsibility:
- Use Gemini to reason over free-text eligibility criteria (soft constraints)
- Runs after hard constraints have filtered obvious non-matches
- Soft constraints include:
  - Disease stage (e.g., "stage 2 breast cancer")
  - Prior treatments (e.g., "chemo and tamoxifen")
  - Biomarkers or lab values
  - Functional status
  - Other criteria that require natural language understanding

This is where the LLM adds real value: parsing messy, domain-specific text
that would be hard to extract with simple code.
"""

from typing import Dict, Any, List


class EligibilityReasoner:
    """
    Uses Gemini to reason over soft eligibility constraints (free-text criteria).

    Methods (to be implemented):
    - reason_soft_constraints(patient_profile: Dict, trial: Dict) -> Dict
      Return eligibility assessment: {confidence: "likely" | "possibly" | "unlikely", rationale: str}
    """

    def reason_soft_constraints(
        self, patient_profile: Dict[str, Any], trial: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reason whether patient matches trial's soft constraints.

        Args:
            patient_profile: Extracted patient info (age, condition, treatments, etc.)
            trial: Trial record with eligibilityModule.eligibility_criteria (free text)

        Returns:
            {
                "confidence": "likely_eligible" | "possibly_eligible" | "likely_not_eligible",
                "rationale": "one-line plain-language explanation"
            }

        Implementation details:
        1. Extract free-text eligibility criteria from trial
        2. Build a prompt for Gemini:
           - Patient profile (disease, stage, prior treatments)
           - Trial's eligibility criteria
           - Question: "Is this patient likely to qualify? Be honest about uncertainty."
        3. Parse Gemini's response to extract confidence level and rationale
        4. Return structured result
        """
        raise NotImplementedError("reason_soft_constraints not yet implemented")
