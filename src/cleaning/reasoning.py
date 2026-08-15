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

import httpx
import os
import asyncio
import re
import json as json_module
import logging
from typing import Dict, Any, List
from src.config import MODELS_FALLBACK, load_api_keys, GEMINI_TIMEOUT

logger = logging.getLogger(__name__)


class EligibilityReasoner:
    """
    Uses Gemini to reason over soft eligibility constraints (free-text criteria).

    Methods:
    - reason_soft_constraints(patient_profile: Dict, trial: Dict) -> Dict
      Return eligibility assessment: {confidence: "likely_eligible" | "possibly_eligible" | "likely_not_eligible", rationale: str}
    """

    def __init__(self):
        """
        Initialize with Gemini API configuration.

        IMPORTANT: This class does NOT manage API key rotation or retries.
        Those are handled centrally by Agent in orchestrator.py.
        This keeps reasoning logic simple and allows orchestrator to control
        all quota/retry decisions across all tools uniformly.
        """
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.model = MODELS_FALLBACK[0]  # Use first model; orchestrator handles fallback
        self.base_url_template = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def _get_base_url(self) -> str:
        """Get base URL for current model."""
        return self.base_url_template.format(model=self.model)

    async def reason_soft_constraints(
        self, patient_profile: Dict[str, Any], trial: Dict[str, Any]
    ) -> Dict[str, Any]:  # async to match orchestrator's await pattern
        """
        Reason whether patient matches trial's soft constraints.

        Args:
            patient_profile: Extracted patient info (age, condition, treatments, etc.)
            trial: Trial record with eligibility criteria

        Returns:
            {
                "confidence": "likely_eligible" | "possibly_eligible" | "likely_not_eligible",
                "rationale": "one-line plain-language explanation"
            }

        Implementation details:
        1. Extract free-text eligibility criteria from trial
        2. Build a prompt for Gemini
        3. Parse response to extract confidence and rationale
        4. Return structured result
        """
        if not self.api_key:
            return {
                "confidence": "possibly_eligible",
                "rationale": "API key not configured; unable to evaluate soft constraints.",
            }

        # Extract patient info
        patient_condition = patient_profile.get("condition", "")
        patient_stage = patient_profile.get("disease_stage", "")
        patient_treatments = patient_profile.get("prior_treatments", [])
        patient_notes = patient_profile.get("other_notes", "")

        # Extract trial eligibility criteria
        trial_eligibility = trial.get("eligibility_criteria", "")
        trial_inclusion = trial.get("inclusion_criteria", [])
        trial_exclusion = trial.get("exclusion_criteria", [])
        trial_title = trial.get("brief_title", "")

        # Build prompt
        prompt = f"""You are a medical eligibility expert. Assess whether a patient is likely to qualify for a clinical trial based on soft constraints (those requiring judgment, not structured data like age/sex).

TRIAL: {trial_title}

TRIAL ELIGIBILITY CRITERIA (raw text):
{trial_eligibility}

TRIAL INCLUSION CRITERIA (structured):
{'; '.join(trial_inclusion) if trial_inclusion else 'None provided'}

TRIAL EXCLUSION CRITERIA (structured):
{'; '.join(trial_exclusion) if trial_exclusion else 'None provided'}

PATIENT PROFILE:
- Condition: {patient_condition}
- Disease Stage: {patient_stage}
- Prior Treatments: {', '.join(patient_treatments) if patient_treatments else 'None reported'}
- Other Notes: {patient_notes}

Based on the trial's free-text eligibility criteria and the patient's profile, assess the likelihood of eligibility. Focus on soft constraints like disease stage, prior treatments, biomarkers, functional status, etc.

Respond in exactly this JSON format:
{{
  "confidence": "likely_eligible" or "possibly_eligible" or "likely_not_eligible",
  "rationale": "A concise one-line explanation of why (max 100 characters)"
}}

Be honest about uncertainty. If criteria are vague or patient info is incomplete, say "possibly_eligible" rather than overclaiming."""

        # Call Gemini API (single attempt)
        # IMPORTANT: No retry/rotation logic here. That's handled by Agent in orchestrator.py.
        # If this fails, orchestrator will decide whether to retry, rotate keys, or return graceful degradation.
        # This keeps reasoning logic simple and ensures consistent quota/retry strategy across all tools.
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._get_base_url(),
                    json={
                        "contents": [
                            {
                                "parts": [
                                    {"text": prompt}
                                ]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.5,
                            "topK": 20,
                            "topP": 0.9,
                        },
                    },
                    params={"key": self.api_key},
                    timeout=GEMINI_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()

                # Extract text from response
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if len(parts) > 0 and "text" in parts[0]:
                            text = parts[0]["text"]
                            # Parse JSON response (robust extraction from markdown or plain JSON)
                            try:
                                # Try parsing as plain JSON first (most common case)
                                result = json_module.loads(text.strip())
                            except json_module.JSONDecodeError:
                                # If that fails, extract from markdown code blocks using regex
                                # Handles: ```json ... ```, ``` ... ```, or backticks in JSON
                                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
                                if json_match:
                                    result = json_module.loads(json_match.group(1).strip())
                                else:
                                    raise  # Re-raise JSONDecodeError if no markdown found

                            return {
                                "confidence": result.get("confidence", "possibly_eligible"),
                                "rationale": result.get("rationale", "Unable to determine."),
                            }

                return {
                    "confidence": "possibly_eligible",
                    "rationale": "Unable to parse Gemini response.",
                }

        except Exception as e:
            # Fail fast: let orchestrator handle this error
            logger.exception("Gemini API call failed in soft constraint reasoning")
            raise
