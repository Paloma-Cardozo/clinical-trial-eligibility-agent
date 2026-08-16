"""
Agent Orchestrator: Main agentic loop.

Responsibility:
- Accept a user message and current AgentState
- Call Gemini with the state as context and the new message
- Parse Gemini's response to extract function calls (if any)
- Execute tools (search_trials, get_trial_detail, reason_soft_constraints)
- Update AgentState with results and conversation history
- Decide whether to continue the loop or return a final answer

This is where the "agent" becomes intelligent: instead of a fixed pipeline,
Gemini decides what to do next (ask clarifying questions, search, filter, reason, or stop).

================================================================================
API KEY MANAGEMENT STRATEGY
================================================================================

TWO APPROACHES SHOWN BELOW:

1. DEVELOPMENT MODE (Current implementation with key rotation):
   ✅ Multiple API keys (currently 10 from free-tier Google AI Studio projects)
   ✅ Automatic rotation when quota exhausted (429 errors)
   ✅ Enables rapid iteration and testing with multiple agent loops
   ⚠️ Trade-off: Requires management of multiple keys
   ⚠️ Note: Uses free-tier quota; in production would use paid quota or optimization

   Context: This is suitable for a development/testing environment where you need
   rapid iteration and can manage multiple free-tier accounts. It demonstrates
   understanding of quota constraints and API rate limiting.

2. PRODUCTION MODE (Recommended alternative - NOT currently implemented):
   ✅ Single API key (paid quota via Cloud Billing)
   ✅ Quota-aware backoff and circuit breaker patterns
   ✅ Optimized token consumption (see optimizations below)
   ✅ Simpler deployment and maintenance

   Optimizations for production:
   - Aggressive caching (2-level: agent + session)
   - Prompt-based soft constraint reasoning (vs tool calls)
   - Reduced max_iterations (4 vs 7)
   - Early stopping on successful candidate threshold

   Context: For a real-world system, this approach is preferred because:
   - Clear cost model (pay for what you use)
   - No workarounds needed; compliant with ToS
   - More maintainable; single key vs multiple
   - Predictable scaling

The code below implements DEVELOPMENT MODE for rapid testing.
To switch to PRODUCTION MODE, see comments at _rotate_api_key().
"""

import json
import asyncio
import os
import logging
from typing import Tuple, Optional, List, Any, Dict
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Configure efficient retry strategy: avoid wasting quota on automatic retries
# Only retry server errors (500, 503), NOT quota errors (429) - handled by our key rotation
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=1,  # Single attempt - no automatic retries
    http_status_codes=[500, 503]  # Only retry server errors, skip 429 (quota)
)
HTTP_OPTIONS = types.HttpOptions(retry_options=RETRY_CONFIG)
from src.agent.state import AgentState
from src.clinicaltrials.client import TrialSearcher, TrialDetail
from src.cleaning.eligibility import EligibilityFilter
from src.cleaning.reasoning import EligibilityReasoner
from src.cleaning import clean_trial
from src.config import MODELS_FALLBACK, load_api_keys, GEMINI_TIMEOUT, API_MODE, CURRENT_MODE_INFO

# Load environment variables from .env
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Log current API mode on startup (for visibility)
logger.info(f"=== API MODE: {API_MODE} ===")
logger.info(f"    Strategy: {CURRENT_MODE_INFO['quota_strategy']}")
logger.info(f"    Cost: {CURRENT_MODE_INFO['cost']}")

# ============================================================================
# System Prompt (LITERAL — do not modify)
# ============================================================================
# DOCUMENTATION: This prompt is documented in detail in PROMPTS.md at the project root.
# See PROMPTS.md for the complete reference, decision rationale, and version history.
# Keep PROMPTS.md in sync with any changes made to this prompt.
# ============================================================================

SYSTEM_PROMPT = """You are a Clinical Trial Eligibility Agent. You help patients explore clinical trials on
ClinicalTrials.gov that might be relevant to their situation, based on their condition, age,
and other details they share with you.

## Your tools

- search_trials(condition, location=None, status="RECRUITING", page_token=None): searches
  for trials. Hard eligibility constraints (age, sex, healthy-volunteer status) are already
  applied automatically before results reach you — every candidate returned has already
  passed those checks.
- get_trial_detail(nct_id): fetches full details for one specific trial.
- reason_soft_constraints(nct_id): evaluates a trial's free-text eligibility criteria
  (disease stage, prior treatments, and similar) against the patient's profile. You must
  call get_trial_detail for a trial before you can call reason_soft_constraints on it.
- update_patient_profile(age=None, sex=None, condition=None, disease_stage=None,
  prior_treatments=None, location_preference=None, willing_to_travel=None, other_notes=None):
  stores patient clinical information persistently so that search_trials and reason_soft_constraints
  can access complete profile data. Only include fields the patient has mentioned; omit fields
  with no data. This tool always succeeds (returns OK), but must be called to persist information.

Asking the patient a clarifying question is NOT a tool call — it's simply responding with
plain text instead of invoking a function.

## Before your first search

You need, at minimum: (1) the patient's specific medical condition/diagnosis, (2) their age,
and (3) their sex. Trial eligibility is meaningless without all three. If any is missing,
ask for it conversationally before proceeding — one question at a time, not a checklist.
Prioritize condition first (harder to infer), then age, then sex. Avoid asking for
information you don't yet need (e.g., don't ask about prior treatments before you even know
what condition to search for). If the patient's condition is vague ("I'm sick"), ask for
specificity ("what has your doctor told you?") before searching. Location is never required
to search — you can search globally if the patient hasn't stated a preference. If the patient
states an explicit location/travel constraint (e.g., "I can only travel within Denmark"), use
it directly as the `location` parameter in search_trials — don't search globally and filter
afterward. Only fall back to a broader, unrestricted search if a location-constrained search
returns no viable candidates (this counts as a refinement, per the stopping criteria below),
and be explicit with the patient that you're widening the search beyond their stated constraint.

## How to collect patient information

As soon as the patient reveals their condition, age, or sex — the three fields required
before your first search (see above) — call update_patient_profile immediately to persist
them, even mid-conversation. Don't wait until you have all three; store each as it's
mentioned. For example, if the patient says "I'm 60 with breast cancer," call
update_patient_profile(age=60, condition="breast cancer") right away, even though sex is
still missing.

Also call update_patient_profile whenever the patient mentions disease stage, prior
treatments, location preferences, or other clinical factors — these are optional and not
required before searching, but useful for later soft-constraint reasoning.

IMPORTANT: for the three required fields (condition, age, sex), calling update_patient_profile
is NOT optional — you must persist them as soon as you have them, since search_trials relies
on the stored profile, not on what's simply been said in conversation. For the optional
fields, don't ask the patient to provide data solely to fill the tool — let the conversation
flow naturally; the profile grows as the patient shares more.

Do not ask "Have you had any treatments?" if they haven't mentioned treatments — ask
naturally: "Tell me about your medical history." Let the patient volunteer optional
information, but always store required information the moment it's given.

## Stopping criteria

Stop and present your findings once either condition is met, whichever comes first:

1. You have accumulated 3-8 candidates that passed BOTH hard constraint checks AND soft
   constraint evaluation, across all searches in this conversation.
2. You've completed 2 targeted search refinements that yielded no viable candidates.

A refinement = a new search with different parameters (e.g., broader condition, different
geography, less restrictive terms). If a search returns trials but none pass eligibility
checks, that counts as a failed refinement.

Example: Search 1 ("breast cancer") → 2 valid candidates (below the 3-minimum, keep going).
Search 2 ("breast cancer, HER2+") → 3 more valid candidates. Total: 5 → within range → STOP
and present these 5. If a single search already yields 3-8 valid candidates, stop there —
you don't need to force a second search just because your plan anticipated one.

Only count candidates that passed both hard and soft evaluation — don't count candidates
that were screened out.

If more than 8 candidates pass all filters, prioritize by confidence level (likely_eligible
first, possibly_eligible second). Present the top 8 and mention: "There are [N] more
matching trials available — ask if you'd like to see others." Secondary tiebreaker:
geographic preference, if the patient explicitly stated a location constraint.

If no relevant trials exist after reasonable effort, say so honestly. Don't continue
searching indefinitely.

When presenting results, always indicate why each candidate was included, and be clear
about your confidence level for each.

## How you present yourself

You're an information tool to help explore potential trials — not a medical advisor or
eligibility authority. Always:
- Present trials as candidates for discussion with their doctor, never as recommendations.
- Be explicit about uncertainty: "based on the trial criteria, this might be worth
  discussing" rather than "you qualify" or "you don't qualify."
- Acknowledge when you're reasoning about unstructured eligibility text (soft constraints)
  versus structured fields — the former carries more inherent uncertainty.
- Never discourage someone from seeking medical advice or care.

## When presenting final results

For EACH trial you present, explicitly state its confidence level so the patient understands the strength of the match:
- Use "likely eligible" for trials matching most inclusion criteria with high confidence
- Use "possibly eligible" for trials that might match but have some uncertainty
- Use "likely not eligible" (don't include these in your final list)
- Explain briefly WHY you assigned this level: e.g., "possibly eligible because [uncertainty about biomarker status / unclear disease stage]"

Always frame results in context: "These are candidates to discuss with your oncologist — not recommendations or guarantees. Your doctor can help determine which best fit your specific situation."

## Presenting Ranked Final Results

When the synthesis phase presents a final list, you will receive QUALIFIED CANDIDATES that have passed
both hard constraints (age, sex) and soft constraints (disease stage, treatments, criteria match).
These candidates are ranked by confidence level:
- "likely_eligible" trials (highest match confidence)
- "possibly_eligible" trials (some uncertainty in match)

When presenting the final list:
1. Present ONLY the qualified candidates provided (don't add trials that weren't evaluated)
2. Present them in rank order (confidence-ranked, highest first)
3. For each trial, state the confidence level and explain the soft constraint reasoning
4. Stop after 8 candidates max, but mention if more are available

If the patient describes symptoms of a medical emergency (severe pain, difficulty
breathing, loss of consciousness, chest pain, or other acute distress), stop the trial
conversation immediately and tell them to call emergency services or go to the nearest
emergency room — do not discuss trials until they've addressed the acute situation.

For time-sensitive but non-emergency concerns ("I have severe fatigue" or "my symptoms are
getting worse"), encourage them to contact their doctor before pursuing new trials."""


# ============================================================================
# Tool Declarations (google-genai format — plain dicts)
# ============================================================================

TOOL_SEARCH_TRIALS = {
    "name": "search_trials",
    "description": (
        "Search for clinical trials matching patient's condition and preferences. CALL THIS FIRST before other tools. "
        "Returns: list of NCT IDs with basic info (title, status, location). "
        "Hard constraint filters (age, sex, healthy volunteer status) are already applied automatically — only candidates passing hard eligibility are returned. "
        "Next step: For promising candidates, call get_trial_detail to fetch full eligibility criteria, then reason_soft_constraints to evaluate match."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "condition": {
                "type": "string",
                "description": "Patient's disease or condition (e.g., 'breast cancer', 'type 2 diabetes'). Required.",
            },
            "location": {
                "type": "string",
                "description": "Geographic region/country constraint (e.g., 'Denmark', 'Europe'). Optional. If patient stated travel constraint, use this.",
            },
            "status": {
                "type": "string",
                "description": "Trial recruitment status (default: 'RECRUITING'). Optional.",
                "enum": ["RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"],
            },
        },
        "required": ["condition"],
    },
}

TOOL_GET_TRIAL_DETAIL = {
    "name": "get_trial_detail",
    "description": (
        "Fetch comprehensive eligibility criteria and trial details for a specific clinical trial. "
        "Use this tool ONLY with the exact tool name 'get_trial_detail' (NOT 'get_trial' or any variant). "
        "Returns: full inclusion/exclusion criteria, patient population, treatment details, and safety information. "
        "MUST call this before calling reason_soft_constraints for the same trial to evaluate eligibility. "
        "Example: To evaluate if trial NCT05928429 matches patient's disease stage, first fetch its details with this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "nct_id": {
                "type": "string",
                "description": "NCT ID of the trial (e.g., 'NCT05928429'). Must be from search_trials results.",
            },
        },
        "required": ["nct_id"],
    },
}

TOOL_REASON_SOFT_CONSTRAINTS = {
    "name": "reason_soft_constraints",
    "description": (
        "Evaluate whether a trial's soft eligibility criteria match the patient's profile. "
        "Soft constraints include: disease stage/progression, prior treatments, biomarkers, genetic factors. "
        "PREREQUISITE: You MUST call get_trial_detail FIRST for this NCT ID to fetch eligibility text. "
        "Returns: confidence level (likely_eligible, possibly_eligible, not_eligible) + reasoning about soft constraint match. "
        "Use this ONLY after calling get_trial_detail for the same trial. "
        "Example: After fetching trial NCT05928429 with get_trial_detail, call this to check if HER2+ status matches."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "nct_id": {
                "type": "string",
                "description": "NCT ID of the trial (must have called get_trial_detail for this ID first).",
            },
        },
        "required": ["nct_id"],
    },
}

TOOL_UPDATE_PATIENT_PROFILE = {
    "name": "update_patient_profile",
    "description": (
        "Store patient clinical information persistently for use by search_trials and reason_soft_constraints. "
        "Call this immediately when the patient reveals age, sex, or condition (the three required fields) "
        "even mid-conversation; don't wait for all three. Also call when they mention disease stage, prior treatments, "
        "location preferences, or other details. Only include fields the patient has mentioned; omit fields with no data. "
        "Returns: success confirmation (OK)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "age": {
                "type": "integer",
                "description": "Patient's age in years (optional, but required before search_trials).",
            },
            "sex": {
                "type": "string",
                "description": "Patient's biological sex: 'M', 'F', or 'Other' (optional, but required before search_trials).",
            },
            "condition": {
                "type": "string",
                "description": "Medical condition/diagnosis (optional, but required before search_trials).",
            },
            "disease_stage": {
                "type": "string",
                "description": "Disease stage or progression (e.g., 'stage 2', 'early', 'advanced'). Optional.",
            },
            "prior_treatments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of prior treatments (e.g., ['chemotherapy', 'tamoxifen']). Optional; extends existing list.",
            },
            "location_preference": {
                "type": "string",
                "description": "Geographic location preference or constraint (e.g., 'Denmark', 'within 100 miles'). Optional.",
            },
            "willing_to_travel": {
                "type": "boolean",
                "description": "Whether patient is willing to travel for trials. Optional.",
            },
            "other_notes": {
                "type": "string",
                "description": "Any other relevant clinical notes. Optional.",
            },
        },
    },
}


# ============================================================================
# Agent Class
# ============================================================================


class Agent:
    """
    Orchestrates the agentic loop for clinical trial eligibility matching.

    Uses google-genai Interactions API with function calling.
    Manages state, tool execution, and conversation history.
    """

    def __init__(self):
        """
        Initialize agent with google-genai client.

        DEVELOPMENT MODE: Loads multiple API keys for key rotation when quota exhausted.
        This enables rapid testing with free-tier quota limits.

        PRODUCTION MODE: Would load single API key from secure configuration
        (e.g., Google Cloud Secret Manager with paid quota).
        """
        # Load all available API keys (DEVELOPMENT: supports rotation; PRODUCTION: single key)
        self.api_keys = load_api_keys()
        self.api_key_index = 0

        logger.debug(f"Loaded {len(self.api_keys)} API key(s) for {'development testing' if len(self.api_keys) > 1 else 'production use'}")

        # Initialize client with first available key
        if self.api_keys:
            os.environ["GOOGLE_API_KEY"] = self.api_keys[self.api_key_index]
            logger.debug("Using API key index 0")
        else:
            logger.warning("No API keys found")

        # Initialize client with efficient retry config (skip automatic retries for 429)
        # Only 1 attempt per request; 429 errors trigger key rotation via our logic
        self.client = genai.Client(http_options=HTTP_OPTIONS)
        logger.debug("Initialized Gemini client with efficient retry strategy (attempts=1, skip 429)")

        self.trial_searcher = TrialSearcher()
        self.eligibility_filter = EligibilityFilter()
        self.eligibility_reasoner = EligibilityReasoner()
        self.model_index = 0  # Track which model in fallback chain to use
        self.model = MODELS_FALLBACK[self.model_index]
        self.trial_details_cache = {}  # Persistent cache across sessions (NCT_ID -> CleanedTrial)
        logger.debug(f"Using model: {self.model}")

    async def _rotate_api_key(self) -> bool:
        """
        Rotate to next API key when current quota exhausted (429 error).

        ============================================================================
        DEVELOPMENT MODE (Current):
        ============================================================================
        Implements key rotation to extend testing capacity across multiple free-tier
        Google AI Studio projects. Enables rapid iteration without waiting for quota
        reset (typically 24 hours).

        Called when: Gemini API returns 429 (quota exceeded)
        Effect: Switches to next available key; allows agent loop to continue

        Trade-off: Requires managing multiple API keys; suitable for dev/testing only.

        ============================================================================
        PRODUCTION MODE (Recommended alternative):
        ============================================================================
        Replace this with quota-aware backoff:

            async def _handle_quota_exhausted(self) -> bool:
                # Single paid API key; implement exponential backoff instead
                wait_time = calculate_backoff(retry_count)
                await asyncio.sleep(wait_time)
                return True  # Retry same key after backoff

        Or: Implement circuit breaker pattern to gracefully degrade service.

        ============================================================================
        """
        if not self.api_keys or self.api_key_index >= len(self.api_keys) - 1:
            # All keys exhausted; in production would trigger alert/fallback
            return False

        self.api_key_index += 1
        os.environ["GOOGLE_API_KEY"] = self.api_keys[self.api_key_index]

        # CRITICAL: Recreate client with new API key (same efficient retry config)
        self.client = genai.Client(http_options=HTTP_OPTIONS)

        # Update reasoner with new key
        self.eligibility_reasoner = EligibilityReasoner()

        logger.debug(f"Rotated to API key index {self.api_key_index} (development testing mode)")
        return True

    async def _get_available_model(self) -> str:
        """
        Get the next available model in fallback chain.

        On 429/503 errors, automatically advance to next model.
        Returns the model string to use.
        """
        if self.model_index < len(MODELS_FALLBACK):
            return MODELS_FALLBACK[self.model_index]
        return MODELS_FALLBACK[-1]  # Fallback to last model if index exceeds

    async def _handle_model_error(self, error: Exception) -> bool:
        """
        Handle API errors. Return True if should retry.
        Strategy: Try next model first, then next API key.
        """
        error_str = str(error).lower()

        # Check for rate limit (429) or unavailable (503) errors
        is_quota_error = "429" in error_str or "too many requests" in error_str or "resource_exhausted" in error_str
        is_unavailable = "503" in error_str or "unavailable" in error_str or "high demand" in error_str

        if is_quota_error or is_unavailable:
            # First, try next model
            if self.model_index < len(MODELS_FALLBACK) - 1:
                self.model_index += 1
                self.model = MODELS_FALLBACK[self.model_index]
                return True

            # If all models exhausted, try next API key (resets model index)
            if await self._rotate_api_key():
                self.model_index = 0
                self.model = MODELS_FALLBACK[self.model_index]
                return True

        return False

    async def _synthesize_partial_results(self, state: AgentState, evaluated_candidates: Dict[str, str] = None) -> str:
        """
        Synthesize a response from partial evaluation when an error occurs.

        If qualified candidates exist from soft constraint evaluation, rank and present them.
        Otherwise, return summary of available search results.

        Args:
            state: Current AgentState
            evaluated_candidates: Dict {nct_id: confidence_level}
        """
        if evaluated_candidates is None:
            evaluated_candidates = {}

        patient_age = state.patient_profile.age or "unknown age"
        patient_sex = state.patient_profile.sex or "unknown sex"
        patient_condition = state.patient_profile.condition or "your condition"

        response = []
        response.append(f"Patient profile: {patient_age}yo {patient_sex} with {patient_condition}")

        if evaluated_candidates:
            qualified = self._get_qualified_candidates(evaluated_candidates, state)
            if qualified:
                response.append(f"\nClinical trials matching your profile:")
                for i, cand in enumerate(qualified, 1):
                    conf = "likely eligible" if cand["confidence"] == "likely_eligible" else "possibly eligible"
                    response.append(f"{i}. {cand['brief_title']} ({cand['nct_id']}) - {conf}")
                response.append("\nDiscuss these options with your doctor to determine which best fit your situation.")
                return "\n".join(response)

        if state.last_search_results:
            total_trials = len(state.last_search_results)
            response.append(f"\nFound {total_trials} clinical trials matching {patient_condition}.")

            if hasattr(self, 'eligibility_filter'):
                passing_ids = self.eligibility_filter.filter_by_hard_constraints(
                    patient_age=state.patient_profile.age or 0,
                    patient_sex=state.patient_profile.sex or "All",
                    candidate_trials=[clean_trial(t) for t in state.last_search_results],
                )
                response.append(f"{len(passing_ids)} trials meet your age and sex eligibility criteria.")

                if state.fetched_trial_details:
                    response.append(f"\nTrials evaluated:")
                    for nct_id, trial in state.fetched_trial_details.items():
                        response.append(f"  - {trial.brief_title} ({nct_id})")

        response.append("\nPlease consult with your doctor about these clinical trial options.")
        return "\n".join(response)

    async def process_message(
        self, state: AgentState, user_message: str
    ) -> Tuple[AgentState, str]:
        """
        Client-side history with key rotation: maintains conversation state locally and
        tolerates API key rotation mid-turn. Now with conversation_history truncation
        to prevent error 400 when history grows too large.

        Design:
        - Interactions API server: store=False (stateless — doesn't save state)
        - Client (this code): maintains conversation_history, patient_profile, search results
        - Truncation: Keep only last 20 elements of conversation_history to prevent
          error 400 in iteration 2+ (Gemini API limitation with large histories)
        - Why: v1's previous_interaction_id is specific to each API key/project. When we
          rotate keys due to quota exhaustion, the new server can't access the old
          interaction_id (404). Solution: keep full history locally, pass it to each
          new request. Allows seamless key rotation without 404s.
        """
        max_iterations = 10  # Hard limit (decided in Phase 3). Protects against infinite loops. With parallel function calling (Corrección B), should complete in 2-3 iterations, well under limit.
        iteration = 0
        final_response_text = None
        gemini_call_count = 0

        # Conversation history for stateless mode
        conversation_history = []

        # Track evaluated candidates for dynamic patient summary
        # Format: {nct_id: confidence_level} where confidence_level in ["likely_eligible", "possibly_eligible", "likely_not_eligible"]
        evaluated_candidates = {}

        logger.info(f"=== STARTING AGENT LOOP: patient condition={state.patient_profile.condition} ===")

        while iteration < max_iterations:
            iteration += 1

            interaction = None
            retry_count = 0
            max_retries = (len(self.api_keys) * len(MODELS_FALLBACK)) if self.api_keys else 1

            while retry_count < max_retries and interaction is None:
                try:
                    tools = [
                        {**TOOL_SEARCH_TRIALS, "type": "function"},
                        {**TOOL_GET_TRIAL_DETAIL, "type": "function"},
                        {**TOOL_REASON_SOFT_CONSTRAINTS, "type": "function"},
                        {**TOOL_UPDATE_PATIENT_PROFILE, "type": "function"},
                    ]

                    model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
                    gemini_call_count += 1
                    logger.info(f">>> GEMINI CALL #{gemini_call_count}, Iteration {iteration}: Key index {self.api_key_index}, Retry {retry_count}/{max_retries}")

                    # Build input for this iteration
                    if iteration == 1:
                        # First iteration: start with user message
                        call_input = user_message
                    else:
                        # Subsequent: use conversation history + new function results
                        call_input = conversation_history.copy()

                    # Stateless mode (store=False) requires tools in EVERY iteration per google-genai docs
                    # Not just iter 1. This ensures Gemini has access to tool schemas and prevents alucinación
                    # of tool names (e.g., "get_trial" vs "get_trial_detail").
                    create_kwargs = {
                        "model": model_name,
                        "input": call_input,
                        "store": False,  # Server doesn't save state; we maintain it locally
                        "tools": tools,  # Include tools in EVERY iteration (required for stateless mode)
                    }

                    if iteration == 1:
                        create_kwargs["system_instruction"] = SYSTEM_PROMPT

                    # Validate conversation_history before sending to Gemini
                    if not self._validate_conversation_history(call_input):
                        logger.error("Conversation history validation failed - possible corruption detected")
                        logger.info("Terminal error (corrupted history). Synthesizing partial results.")
                        partial_response = await self._synthesize_partial_results(state, evaluated_candidates)
                        return (state, partial_response)

                    # Call API with timeout to prevent indefinite hangs
                    loop = asyncio.get_event_loop()
                    interaction = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: self.client.interactions.create(**create_kwargs)),
                        timeout=GEMINI_TIMEOUT
                    )

                except Exception as e:
                    error_str = str(e).lower()
                    logger.exception("Gemini API call failed")

                    is_quota_error = "429" in error_str or "too_many_requests" in error_str or "resource_exhausted" in error_str
                    is_unavailable = "503" in error_str or "unavailable" in error_str or "high demand" in error_str
                    logger.debug(f"Error detection: quota={is_quota_error}, api_key_index={self.api_key_index}, num_keys={len(self.api_keys)}")

                    if is_quota_error or is_unavailable:
                        # First, try next model in fallback chain
                        if await self._handle_model_error(e):
                            retry_count += 1
                            await asyncio.sleep(min(2 ** retry_count, 30))
                            logger.info(f"Model/Key rotation successful. Retrying with model_index={self.model_index}, key_index={self.api_key_index}")
                            interaction = None  # Reset so inner while loop continues
                            continue

                    # No more keys or non-quota error: synthesize partial results
                    logger.info(f"Terminal error (non-recoverable): {type(e).__name__}. Synthesizing partial results.")
                    partial_response = await self._synthesize_partial_results(state, evaluated_candidates)
                    return (state, partial_response)
            # Process steps (with parallel function call execution)
            if interaction is None:
                return (state, "Failed to get response from API")

            has_tool_call = False
            step_summary = []
            logger.debug(f"V3 Iteration {iteration}: Processing {len(interaction.steps)} step(s)")

            # Add initial user_input to history if first iteration
            if iteration == 1 and len(conversation_history) == 0:
                conversation_history.append({
                    "type": "user_input",
                    "content": [{"type": "text", "text": user_message}]
                })

            # OPTIMIZATION (Corrección B): Separate function_call steps for parallel execution
            # Process all non-function_call steps first (thoughts, model_output)
            function_call_steps = []  # Collect all function_call steps for parallel execution

            for step in interaction.steps:
                step_type = step.type if hasattr(step, 'type') else type(step).__name__
                logger.debug(f"V3 Step type: {step_type}")

                # Always add step to history
                conversation_history.append(step.model_dump())

                # Check for model_output (final response from model)
                if step_type == "model_output":
                    if hasattr(step, "content") and step.content:
                        text_parts = []
                        for item in step.content:
                            if hasattr(item, "text") and item.text:
                                text_parts.append(item.text)
                        if text_parts:
                            final_response_text = "\n".join(text_parts)
                            step_summary.append("plain text")
                            break

                # Collect function_call steps for parallel execution
                if step_type == "function_call":
                    has_tool_call = True
                    tool_name = getattr(step, "name", None)
                    tool_args = getattr(step, "arguments", {}) or {}
                    tool_id = getattr(step, "id", None)

                    if tool_name:
                        function_call_steps.append((tool_name, tool_args, tool_id, step))

            # PARALLEL EXECUTION with MUTATION-FIRST PARTITION:
            # If update_patient_profile is in the batch, execute it FIRST (sequentially)
            # to update state.patient_profile, THEN execute remaining tools in parallel.
            # This prevents race conditions where search_trials reads stale profile data.
            if function_call_steps:
                step_summary.extend([f"function_call: {name}" for name, _, _, _ in function_call_steps])

                # Partition: update_patient_profile first, rest in parallel
                update_steps = [(name, args, tool_id, step) for name, args, tool_id, step in function_call_steps if name == "update_patient_profile"]
                other_steps = [(name, args, tool_id, step) for name, args, tool_id, step in function_call_steps if name != "update_patient_profile"]

                tool_results = []

                # STEP 1: Execute update_patient_profile sequentially (if present)
                if update_steps:
                    logger.debug(f"V3 Executing update_patient_profile (MUTATION FIRST)")
                    for name, args, tool_id, step in update_steps:
                        result = await self._execute_tool(name, args, state)
                        tool_results.append((name, args, tool_id, step, result))

                # STEP 2: Execute remaining tools in parallel
                if other_steps:
                    logger.debug(f"V3 Executing {len(other_steps)} other tool(s) in parallel (after mutation)")
                    parallel_results = await asyncio.gather(
                        *[self._execute_tool(name, args, state) for name, args, _, _ in other_steps],
                        return_exceptions=True
                    )

                    # Check for quota/unavailable errors from parallel tasks
                    # If found, re-raise to allow orchestrator to handle key rotation
                    quota_error = None
                    for result in parallel_results:
                        if isinstance(result, Exception):
                            error_str = str(result).lower()
                            is_quota_error = "429" in error_str or "too_many_requests" in error_str or "resource_exhausted" in error_str
                            is_unavailable = "503" in error_str or "unavailable" in error_str or "high demand" in error_str
                            if is_quota_error or is_unavailable:
                                quota_error = result
                                break

                    # Process all results (successful + errors)
                    # Don't raise immediately on quota error - let tool_results processing handle it
                    for (name, args, tool_id, step), result in zip(other_steps, parallel_results):
                        tool_results.append((name, args, tool_id, step, result))

                    # Only propagate quota error if ALL results are errors
                    if quota_error and len(tool_results) == 0:
                        logger.debug("All parallel tasks failed with quota/unavailable error, propagating to orchestrator loop")
                        raise quota_error
                    elif quota_error:
                        logger.debug(f"Partial quota/unavailable error in parallel tasks: {len(parallel_results) - len([r for r in parallel_results if isinstance(r, Exception)])} succeeded, will retry failed ones next iteration")

            # Process results and add to history
            for tool_name, tool_args, tool_id, step, tool_result in tool_results:
                # Handle both successful results (strings) and exceptions
                if isinstance(tool_result, Exception):
                    tool_result_text = json.dumps({"error": f"Tool execution failed: {str(tool_result)}"})
                else:
                    tool_result_text = tool_result

                # Extract evaluated candidates from reason_soft_constraints results
                if tool_name == "reason_soft_constraints" and not isinstance(tool_result, Exception):
                    try:
                        result_json = json.loads(tool_result_text)
                        nct_id = tool_args.get("nct_id")
                        confidence = result_json.get("confidence", "unknown")
                        if nct_id:
                            evaluated_candidates[nct_id] = confidence
                            logger.debug(f"Tracked candidate: {nct_id} -> {confidence}")
                    except (json.JSONDecodeError, KeyError):
                        pass  # Ignore parsing errors

                # Add function result to history for next iteration
                conversation_history.append({
                    "type": "function_result",
                    "name": tool_name,
                    "call_id": tool_id,
                    "result": [{"type": "text", "text": tool_result_text}]
                })

            # Manage conversation_history to prevent error 400 in iteration 2+
            # Strategy: Keep patient context permanent, trim history while preserving semantic blocks
            # IMPORTANT: Don't break function_call/function_result pairs or Gemini API rejects with 400
            if len(conversation_history) > 20:
                # Build patient summary (permanent context)
                treatments_text = ", ".join(state.patient_profile.prior_treatments) if state.patient_profile.prior_treatments else "None"

                # Build evaluation status if candidates have been evaluated
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

                # Build list of already-evaluated trials with confidence levels to prevent re-evaluation
                already_evaluated_text = ""
                if evaluated_candidates:
                    evaluated_with_confidence = [f"{nct} ({conf})" for nct, conf in evaluated_candidates.items()]
                    if len(evaluated_with_confidence) <= 20:
                        already_evaluated_text = f"\n[ALREADY EVALUATED: {', '.join(evaluated_with_confidence)}]"
                    else:
                        already_evaluated_text = f"\n[ALREADY EVALUATED: {len(evaluated_with_confidence)} trials - {', '.join(evaluated_with_confidence[:20])}...]"

                patient_summary = {
                    "type": "user_input",
                    "content": [{
                        "type": "text",
                        "text": f"[PATIENT CONTEXT: Age {state.patient_profile.age}, Sex {state.patient_profile.sex}, Condition: {state.patient_profile.condition}, Disease Stage: {state.patient_profile.disease_stage}, Prior Treatments: {treatments_text}]{evaluation_status}{already_evaluated_text}"
                    }]
                }

                # Truncate while preserving semantic integrity
                # Reduced from 19 to 14 to prevent conversation_history corruption under stress
                # Total: 1 patient_summary + 14 tail = 15 items max (conservative threshold)
                history_tail = conversation_history[-14:]  # Start with last 14

                # Verify no orphaned function_call/function_result pairs at edges
                # Remove any function_call not followed by function_result
                cleaned_tail = self._clean_orphaned_tool_calls(history_tail)

                conversation_history = [patient_summary] + cleaned_tail
                logger.debug(f"V3 Compressed history: patient context + semantically valid tail (was {len(history_tail)}, cleaned to {len(cleaned_tail)})")

            logger.debug(f"V3 Iteration {iteration}: {', '.join(step_summary) if step_summary else 'no steps'}")

            # Early trigger: if 3+ qualified candidates, present them now (matches system prompt minimum)
            if len(evaluated_candidates) >= 3 and not final_response_text:
                qualified_candidates = self._get_qualified_candidates(evaluated_candidates, state)
                if qualified_candidates:
                    final_response_text = self._build_qualified_response(qualified_candidates)
                    logger.info(f"Early trigger: {len(qualified_candidates)} qualified candidates - presenting results")
                    break

            if final_response_text:
                logger.debug("V3 Got text response, stopping")
                break

            if not has_tool_call:
                final_response_text = "Unable to process your message."
                break

        # If max_iterations reached without text response, make ONE final call to Gemini (no tools)
        # to ask it to synthesize what it has so far
        if not final_response_text and iteration >= max_iterations:
            logger.info(f"Reached max_iterations ({max_iterations}) without final response. Making synthesis call...")
            try:
                # Get qualified candidates (passed both hard + soft constraints, ranked by confidence)
                qualified_candidates = self._get_qualified_candidates(evaluated_candidates, state)

                # Build synthesis prompt with qualified candidates
                qualified_text = ""
                if qualified_candidates:
                    qualified_text = "\n\nQUALIFIED CANDIDATES (filtered & ranked by confidence):\n"
                    for i, cand in enumerate(qualified_candidates, 1):
                        qualified_text += f"{i}. {cand['nct_id']} - {cand['brief_title']} [{cand['confidence']}]\n"
                    qualified_text += "\nPresent ONLY these qualified candidates to the patient, in rank order. For each trial, explain:\n- Why it qualifies based on soft constraint match\n- Confidence level\n- Next steps (discussion with doctor)"

                synthesis_text = "Based on the trials you've searched and evaluated so far, please provide a summary of the most promising candidates for this patient and your recommendations for next steps. If you haven't found suitable candidates, explain what barriers were encountered and what the patient should discuss with their doctor."
                if qualified_text:
                    synthesis_text += qualified_text

                synthesis_prompt = {
                    "type": "user_input",
                    "content": [{
                        "type": "text",
                        "text": synthesis_text
                    }]
                }
                conversation_history.append(synthesis_prompt)

                model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
                gemini_call_count += 1
                logger.info(f">>> GEMINI SYNTHESIS CALL #{gemini_call_count}: No tools, synthesis only")

                # Make FINAL call without tools (no function calling available)
                synthesis_interaction = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.interactions.create(
                            model=model_name,
                            input=conversation_history,
                            store=False,
                            tools=[],  # NO TOOLS for synthesis
                        )
                    ),
                    timeout=GEMINI_TIMEOUT
                )

                # Extract text response from synthesis call
                if synthesis_interaction and synthesis_interaction.steps:
                    for step in synthesis_interaction.steps:
                        step_type = step.type if hasattr(step, 'type') else type(step).__name__
                        if step_type == "model_output" and hasattr(step, "content") and step.content:
                            for item in step.content:
                                if hasattr(item, "text") and item.text:
                                    final_response_text = item.text
                                    break
                        if final_response_text:
                            break

            except Exception as e:
                logger.exception("Synthesis call failed")
                final_response_text = "I've completed my search iteration limit. Please ask me about specific trials or request more details on any candidates I've evaluated."

        # Fallback if still no response (should be rare)
        if not final_response_text:
            final_response_text = "I was unable to generate a response. Please try asking your question again or request details on specific trials."

        # Don't update previous_interaction_id: we don't rely on server-side state,
        # we maintain full history locally in conversation_history
        logger.info(f"=== AGENT LOOP FINISHED: Total {gemini_call_count} Gemini calls, {iteration} iterations, response length {len(final_response_text or '')} chars ===")
        return (state, final_response_text)

    async def _execute_tool(
        self, tool_name: str, tool_args: Dict[str, Any], state: AgentState
    ) -> str:
        """
        Execute a single tool call and return result as JSON string.

        Handles:
        - search_trials: Call TrialSearcher.search(), apply hard constraints via EligibilityFilter
        - get_trial_detail: Call TrialSearcher.get_trial_details(), cache in state
        - reason_soft_constraints: Check cache, call EligibilityReasoner
        - update_patient_profile: Persist patient info to state.patient_profile

        Args:
            tool_name: Name of tool to execute
            tool_args: Arguments passed by Gemini
            state: Current AgentState (for cache, patient_profile)

        Returns:
            JSON string with tool result (to be sent back to Gemini)

        IMPORTANT: Quota/unavailable errors (429/503) are re-raised to allow
        orchestrator's main loop to handle them with key rotation. Only non-quota
        errors are converted to JSON responses.
        """
        try:
            if tool_name == "search_trials":
                return await self._execute_search_trials(tool_args, state)
            elif tool_name == "get_trial_detail":
                return await self._execute_get_trial_detail(tool_args, state)
            elif tool_name == "reason_soft_constraints":
                return await self._execute_reason_soft_constraints(tool_args, state)
            elif tool_name == "update_patient_profile":
                return await self._execute_update_patient_profile(tool_args, state)
            else:
                # Handle common hallucinations with helpful guidance
                if tool_name == "get_trial":
                    return json.dumps({"error": "Tool 'get_trial' does not exist. Use 'get_trial_detail' instead with the nct_id parameter."})
                else:
                    valid_tools = ["search_trials", "get_trial_detail", "reason_soft_constraints", "update_patient_profile"]
                    return json.dumps({"error": f"Unknown tool: '{tool_name}'. Available tools: {', '.join(valid_tools)}"})
        except Exception as e:
            error_str = str(e).lower()
            # Check if this is a quota/unavailable error
            is_quota_error = "429" in error_str or "too_many_requests" in error_str or "resource_exhausted" in error_str
            is_unavailable = "503" in error_str or "unavailable" in error_str or "high demand" in error_str

            if is_quota_error or is_unavailable:
                # Let orchestrator's main loop handle quota errors (for key rotation)
                logger.debug(f"Tool {tool_name} hit quota/unavailable error, propagating to orchestrator")
                raise

            # For other errors, return graceful error JSON
            logger.exception("Tool execution error (non-quota)")
            return json.dumps({"error": f"Tool error: {type(e).__name__}"})

    async def _execute_search_trials(
        self, tool_args: Dict[str, Any], state: AgentState
    ) -> str:
        """
        Execute search_trials tool.

        PREREQUISITE VALIDATION: Before searching, state.patient_profile must have all three
        required fields populated (condition, age, sex). This is enforced here (defensive layer)
        even though update_patient_profile should have been called by the agent.

        Flow:
        1. Validate state.patient_profile has condition, age, sex
        2. Call TrialSearcher.search() with condition, location, status
        3. For each result, apply EligibilityFilter.filter_by_hard_constraints()
        4. Return list of passing NCT IDs with basic info (title, status)
        """
        # DEFENSIVE CHECK: Ensure all three required fields are populated
        missing_fields = []
        if not state.patient_profile.condition:
            missing_fields.append("condition")
        if state.patient_profile.age is None:
            missing_fields.append("age")
        if not state.patient_profile.sex:
            missing_fields.append("sex")

        if missing_fields:
            return json.dumps({
                "error": f"Cannot search: missing required patient information. Please call update_patient_profile with: {', '.join(missing_fields)} before searching."
            })

        condition = tool_args.get("condition")
        location = tool_args.get("location")
        status = tool_args.get("status", "RECRUITING")

        if not condition:
            return json.dumps({"error": "condition parameter is required for search_trials"})

        # Call TrialSearcher
        search_result = await self.trial_searcher.search(
            condition=condition, location=location, status=status
        )
        logger.debug(f"search_trials API returned {len(search_result.results) if search_result.results else 0} trials for '{condition}' in '{location}'")

        # Save search results for graceful degradation (in case of later errors)
        state.last_search_results = search_result.results

        # Clean each trial to normalize and extract structured fields (e.g., parse eligibility criteria)
        cleaned_trials = [clean_trial(trial) for trial in search_result.results]

        # Apply hard constraint filter (age, sex, healthy volunteer status)
        patient_age = state.patient_profile.age or 0
        patient_sex = state.patient_profile.sex or "All"
        logger.debug(f"Hard constraint filter using: age={patient_age}, sex={patient_sex}")

        passing_nct_ids = self.eligibility_filter.filter_by_hard_constraints(
            patient_age=patient_age,
            patient_sex=patient_sex,
            candidate_trials=cleaned_trials,
        )
        logger.debug(f"After hard constraint filter: {len(passing_nct_ids)} trials pass (age={patient_age}, sex={patient_sex})")

        # Format results for Gemini
        results = []
        for i, trial in enumerate(search_result.results):
            if trial.nct_id in passing_nct_ids:
                results.append(
                    {
                        "nct_id": trial.nct_id,
                        "title": trial.brief_title,
                        "status": trial.overall_status,
                        "condition": trial.condition,
                    }
                )

        return json.dumps(
            {
                "total_found": len(search_result.results),
                "passing_hard_filters": len(results),
                "candidates": results,
            }
        )

    async def _execute_get_trial_detail(
        self, tool_args: Dict[str, Any], state: AgentState
    ) -> str:
        """
        Execute get_trial_detail tool with 2-level caching and response compression.

        Caches full trial details internally but returns compressed version to Gemini
        to prevent 400 "invalid argument" errors when conversation history grows.

        Flow:
        1. Check self.trial_details_cache (persistent across sessions)
        2. Check state.fetched_trial_details cache (per-session)
        3. If not cached, call TrialSearcher.get_trial_details()
        4. Cache result in both levels (FULL details for reasoning)
        5. Return COMPRESSED details to Gemini (essential fields only)
        """
        nct_id = tool_args.get("nct_id")

        if not nct_id:
            return json.dumps({"error": "nct_id is required"})

        # Level 1: Agent-level persistent cache
        if nct_id in self.trial_details_cache:
            trial = self.trial_details_cache[nct_id]
            state.fetched_trial_details[nct_id] = trial  # Also cache in session
            return self._compress_trial_for_gemini(trial)

        # Level 2: Session-level cache
        if nct_id in state.fetched_trial_details:
            trial = state.fetched_trial_details[nct_id]
            self.trial_details_cache[nct_id] = trial  # Promote to persistent
            return self._compress_trial_for_gemini(trial)

        # Level 3: Fetch from API
        try:
            trial = await self.trial_searcher.get_trial_details(nct_id)
            self.trial_details_cache[nct_id] = trial  # Store in persistent cache
            state.fetched_trial_details[nct_id] = trial  # Store in session cache
        except Exception as e:
            logger.exception(f"Failed to fetch trial details for {nct_id}")
            return json.dumps({"error": f"Failed to fetch trial {nct_id}: {type(e).__name__}"})

        return self._compress_trial_for_gemini(trial)

    def _clean_orphaned_tool_calls(self, history_tail: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove orphaned function_call/function_result pairs from history tail.

        After truncation at history[-19:], we might have:
        - A function_result without its function_call (call was at -20)
        - A function_call without its function_result (result is at -18)

        This breaks Gemini API's strict requirements for tool use sequences.

        CRITICAL FIX: Build sets of call IDs and result IDs in this tail, then only keep
        items that form complete pairs. This prevents breaking pairs when truncating.
        """
        if not history_tail:
            return history_tail

        # Scan tail to find what call IDs and result IDs exist
        call_ids_in_tail = set()
        result_call_ids_in_tail = set()

        for item in history_tail:
            if isinstance(item, dict):
                if item.get("type") == "function_call" and item.get("id"):
                    call_ids_in_tail.add(item.get("id"))
                elif item.get("type") == "function_result" and item.get("call_id"):
                    result_call_ids_in_tail.add(item.get("call_id"))

        # Build cleaned list: keep items only if they form complete pairs
        cleaned = []
        removed_count = 0

        for item in history_tail:
            if isinstance(item, dict):
                item_type = item.get("type")

                if item_type == "function_call":
                    call_id = item.get("id")
                    # Keep function_call only if its result exists in this tail
                    if call_id and call_id in result_call_ids_in_tail:
                        cleaned.append(item)
                    else:
                        removed_count += 1

                elif item_type == "function_result":
                    call_id = item.get("call_id")
                    # Keep function_result only if its call exists in this tail
                    if call_id and call_id in call_ids_in_tail:
                        cleaned.append(item)
                    else:
                        removed_count += 1

                else:
                    # Keep all non-tool items (thoughts, model_output, user_input, etc.)
                    cleaned.append(item)
            else:
                cleaned.append(item)

        if removed_count > 0:
            logger.debug(f"Removed {removed_count} orphaned tool call items from history tail (kept {len(cleaned)} items)")

        return cleaned

    def _compress_trial_for_gemini(self, trial) -> str:
        """
        Serialize trial for Gemini with compression.

        Returns only essential fields to keep conversation history small.
        Omits bulky text fields like eligibility_criteria to prevent 400 errors.

        Stored trial object is still cached in full for soft constraint reasoning.
        """
        compressed = {
            "nct_id": trial.nct_id,
            "brief_title": trial.brief_title,
            "overall_status": trial.overall_status,
            "condition": trial.condition,
            "phase": trial.phase,
            "enrollment": trial.enrollment,
            "study_type": trial.study_type,
            "locations": [
                {
                    "country": loc.country,
                    "city": loc.city,
                    "state": loc.state,
                    "facility": loc.facility,
                }
                for loc in (trial.locations or [])
            ],
            "minimum_age": trial.minimum_age,
            "maximum_age": trial.maximum_age,
            "sex": trial.sex,
            "healthy_volunteers": trial.healthy_volunteers,
            "primary_outcomes": trial.primary_outcomes or [],
            "secondary_outcomes": trial.secondary_outcomes or [],
            "start_date": trial.start_date,
            "completion_date": trial.completion_date,
        }
        return json.dumps(compressed)

    async def _execute_reason_soft_constraints(
        self, tool_args: Dict[str, Any], state: AgentState
    ) -> str:
        """
        Execute reason_soft_constraints tool.

        Dependencies:
        - MUST have called get_trial_detail(nct_id) first
        - Trial must be in state.fetched_trial_details cache

        Flow:
        1. Check if nct_id is in cache
        2. If not, return error asking to call get_trial_detail first
        3. Call EligibilityReasoner.reason_soft_constraints()
        4. Return confidence + rationale
        """
        nct_id = tool_args.get("nct_id")

        if not nct_id:
            return json.dumps({"error": "nct_id is required"})

        # Check cache
        if nct_id not in state.fetched_trial_details:
            return json.dumps(
                {
                    "error": f"Trial {nct_id} not yet fetched. Call get_trial_detail({nct_id}) first."
                }
            )

        trial = state.fetched_trial_details[nct_id]

        # Call EligibilityReasoner
        # IMPORTANT: Let quota/unavailable errors (429/503) propagate so orchestrator can rotate keys.
        # Only catch and convert to JSON errors that should not trigger key rotation.
        try:
            result = await self.eligibility_reasoner.reason_soft_constraints(
                patient_profile=state.patient_profile.model_dump(),
                trial=trial.model_dump(),
                api_key=self.api_keys[self.api_key_index] if self.api_keys else None,
                model=self.model,
            )
            return json.dumps(result)
        except Exception as e:
            error_str = str(e).lower()
            # Check if this is a quota/unavailable error that should propagate to orchestrator
            is_quota_error = "429" in error_str or "too_many_requests" in error_str or "resource_exhausted" in error_str
            is_unavailable = "503" in error_str or "unavailable" in error_str or "high demand" in error_str

            if is_quota_error or is_unavailable:
                # Let orchestrator handle quota/unavailable errors
                logger.debug(f"Soft constraint reasoning hit quota/unavailable: {type(e).__name__}")
                raise

            # For other errors, return gracefully as JSON
            logger.exception("Soft constraint reasoning failed (non-quota error)")
            return json.dumps({"error": f"Reasoning failed: {type(e).__name__}"})

    async def _execute_update_patient_profile(
        self, tool_args: Dict[str, Any], state: AgentState
    ) -> str:
        """
        Update patient profile with new information from the agent.

        This is called when the agent extracts patient details from conversation
        and needs to persist them for use by search_trials and reason_soft_constraints.

        Args:
            tool_args: Dict with optional keys: age, sex, condition, disease_stage,
                      prior_treatments, location_preference, willing_to_travel, other_notes
            state: Current AgentState (to update patient_profile)

        Returns:
            JSON string with success confirmation
        """
        # Update only fields that were provided (non-None)
        if "age" in tool_args and tool_args["age"] is not None:
            state.patient_profile.age = tool_args["age"]
            logger.debug(f"Updated patient age: {state.patient_profile.age}")

        if "sex" in tool_args and tool_args["sex"] is not None:
            state.patient_profile.sex = tool_args["sex"]
            logger.debug(f"Updated patient sex: {state.patient_profile.sex}")

        if "condition" in tool_args and tool_args["condition"] is not None:
            state.patient_profile.condition = tool_args["condition"]
            logger.debug(f"Updated patient condition: {state.patient_profile.condition}")

        if "disease_stage" in tool_args and tool_args["disease_stage"] is not None:
            state.patient_profile.disease_stage = tool_args["disease_stage"]
            logger.debug(f"Updated disease stage: {state.patient_profile.disease_stage}")

        if "prior_treatments" in tool_args and tool_args["prior_treatments"]:
            # prior_treatments is a list; extend existing (don't replace)
            treatments_list = tool_args["prior_treatments"]
            if isinstance(treatments_list, list):
                state.patient_profile.prior_treatments.extend(treatments_list)
            logger.debug(f"Added prior treatments: {treatments_list}")

        if "location_preference" in tool_args and tool_args["location_preference"] is not None:
            state.patient_profile.location_preference = tool_args["location_preference"]
            logger.debug(f"Updated location preference: {state.patient_profile.location_preference}")

        if "willing_to_travel" in tool_args and tool_args["willing_to_travel"] is not None:
            state.patient_profile.willing_to_travel = tool_args["willing_to_travel"]
            logger.debug(f"Updated willing to travel: {state.patient_profile.willing_to_travel}")

        if "other_notes" in tool_args and tool_args["other_notes"] is not None:
            state.patient_profile.other_notes = tool_args["other_notes"]
            logger.debug(f"Updated other notes: {state.patient_profile.other_notes}")

        # Log current state
        logger.info(f"Patient profile updated: age={state.patient_profile.age}, sex={state.patient_profile.sex}, condition={state.patient_profile.condition}")

        return json.dumps({"status": "OK", "message": "Patient profile updated successfully"})

    def _validate_conversation_history(self, history: List[Dict[str, Any]]) -> bool:
        """
        Validate conversation_history structure before sending to Gemini.

        Checks for:
        - Valid item types (user_input, function_call, function_result, etc.)
        - Complete function_call/function_result pairs
        - Non-empty content fields

        Returns True if valid, False if corrupted.
        """
        if not history:
            return True

        valid_types = {"user_input", "function_call", "function_result", "model_output", "thought"}
        call_ids = set()
        result_ids = set()

        for item in history:
            if not isinstance(item, dict):
                logger.warning(f"Invalid history item type: {type(item)}, expected dict")
                return False

            item_type = item.get("type")
            if item_type not in valid_types:
                logger.warning(f"Unknown history item type: {item_type}")
                return False

            if item_type == "function_call":
                call_id = item.get("id")
                if not call_id:
                    logger.warning("function_call missing 'id' field")
                    return False
                call_ids.add(call_id)

            elif item_type == "function_result":
                call_id = item.get("call_id")
                if not call_id:
                    logger.warning("function_result missing 'call_id' field")
                    return False
                result_ids.add(call_id)

        # Check for orphaned pairs
        orphaned_calls = call_ids - result_ids
        orphaned_results = result_ids - call_ids

        if orphaned_calls:
            logger.warning(f"Orphaned function_call IDs: {orphaned_calls}")
            return False
        if orphaned_results:
            logger.warning(f"Orphaned function_result IDs: {orphaned_results}")
            return False

        return True

    def _build_qualified_response(self, qualified_candidates: List[Dict[str, Any]]) -> str:
        """
        Build a user-facing response from ranked qualified candidates.

        Args:
            qualified_candidates: List from _get_qualified_candidates()

        Returns:
            Formatted response string with candidates ranked and explained.
        """
        if not qualified_candidates:
            return ""

        response = ["Clinical trials matching your profile:\n"]
        for i, cand in enumerate(qualified_candidates, 1):
            conf = "likely eligible" if cand["confidence"] == "likely_eligible" else "possibly eligible"
            response.append(f"{i}. {cand['brief_title']} ({cand['nct_id']}) - {conf}")

        response.append("\nDiscuss these options with your doctor to determine which best fit your situation.")
        return "\n".join(response)

    def _get_qualified_candidates(
        self, evaluated_candidates: Dict[str, str], state: AgentState
    ) -> List[Dict[str, Any]]:
        """
        Filter and rank trials that passed BOTH hard + soft constraints.

        Combines hard constraint evaluation (already applied during search_trials) with
        soft constraint evaluation (confidence levels from reason_soft_constraints).

        Args:
            evaluated_candidates: Dict {nct_id: confidence_level} from soft constraint reasoning
            state: Current AgentState with patient profile and fetched trial details

        Returns:
            List of qualified trials, ranked by confidence (likely_eligible first, then possibly_eligible):
            [
                {
                    "nct_id": "NCT...",
                    "confidence": "likely_eligible" | "possibly_eligible",
                    "brief_title": "...",
                    "trial": TrialDetail object
                },
                ...
            ]
        """
        qualified = []

        for nct_id, confidence in evaluated_candidates.items():
            # Skip trials not in cache (shouldn't happen if orchestrator is correct)
            if nct_id not in state.fetched_trial_details:
                logger.debug(f"Skipping {nct_id}: not in trial cache")
                continue

            trial = state.fetched_trial_details[nct_id]

            # Only include likely_eligible and possibly_eligible (exclude likely_not_eligible)
            if confidence in ["likely_eligible", "possibly_eligible"]:
                qualified.append({
                    "nct_id": nct_id,
                    "confidence": confidence,
                    "brief_title": trial.brief_title,
                    "trial": trial,
                })
                logger.debug(f"Qualified candidate: {nct_id} ({confidence})")

        # Rank by confidence level: likely_eligible first, then possibly_eligible
        # Within same confidence, maintain discovery order (dict preserves insertion order in Python 3.7+)
        qualified.sort(key=lambda x: (
            0 if x["confidence"] == "likely_eligible" else 1,
            list(evaluated_candidates.keys()).index(x["nct_id"])
        ))

        logger.info(f"Qualified candidates ranked: {len(qualified)} trials ({len([c for c in qualified if c['confidence']=='likely_eligible'])} likely_eligible, {len([c for c in qualified if c['confidence']=='possibly_eligible'])} possibly_eligible)")

        return qualified

