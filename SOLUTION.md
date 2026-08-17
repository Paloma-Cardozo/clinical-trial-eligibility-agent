# Clinical Trial Eligibility Agent — Solution Approach

This document explains how each requirement from the specification was addressed, with references to the actual code implementation. This is a technical portfolio artifact showing design decisions, architecture, and problem-solving.

---

## Overview

**Problem:** Patients need an efficient way to discover clinical trials they may qualify for from the public ClinicalTrials.gov registry.

**Solution:** Full-stack agent with FastAPI backend, Vanilla JS frontend, LLM-driven control flow, deterministic hard constraint filtering, and LLM-powered soft constraint reasoning for robust trial matching.

---

## 1. Patient-Facing Interface

### Requirement
A simple chat-style interface where patients describe their situation in free text.

### Solution

**Frontend:** [src/ui/index.html](src/ui/index.html)
- Vanilla HTML/JS (no build process, no framework overhead)
- Session persistence using `sessionStorage` for tab-scoped session IDs
- Chat UI with automatic scrolling and clean message formatting
- Server-authoritative session minting ([main.py:45-65](main.py#L45-L65))

**Design Decision:** No framework (React/Vue) — vanilla JS is simpler, no build process required. One command runs both UI and API.

**Key Code:**
- [main.py:45-130](main.py#L45-L130): FastAPI endpoint serving UI and `/chat` API
- [main.py:150-205](main.py#L150-L205): Chat handler, session management, LLM orchestration

---

## 2. Agent That Drives the Process

### Requirement
Agent decides what to do based on incomplete input: ask clarifying questions, formulate API queries, select trials for detail fetch, decide when to stop.

### Solution

**Core Agent Loop:** [src/agent/orchestrator.py](src/agent/orchestrator.py) (~1400 lines)

The agent runs a tool-use loop where the LLM genuinely drives control flow:

```
Initialize agent with system prompt
↓
LLM decides: ask question? search trials? fetch details? reason constraints? present results?
↓
Execute LLM's choice (tool call)
↓
Return results to LLM
↓
Repeat until stopping criteria met
```

#### System Prompt & Instruction Following

**System Prompt:** [src/agent/orchestrator.py:92-220](src/agent/orchestrator.py#L92-L220)

Defines agent behavior in sections:
- **Your tools:** 4 tools (search_trials, get_trial_detail, reason_soft_constraints, update_patient_profile)
- **Before your first search:** must collect condition, age, sex before querying
- **How to collect patient information:** persist required fields as mentioned
- **Stopping criteria:** accumulate 3-8 qualified candidates OR exhaust 2 refinements
- **How you present yourself:** frame as information tool, not medical advisor
- **Presenting Ranked Final Results:** explain confidence levels and ranking

This prompt is authoritative and documented in [PROMPTS.md](PROMPTS.md) for transparency.

#### Dynamic Patient Profile & Conversation History

**State Management:** [src/agent/state.py](src/agent/state.py)
- `AgentState`: Pydantic model tracking patient profile, conversation history, search results
- **Stateless API design:** all state passed to every request, stored in in-memory dict by session_id

**Patient Profile Tracking:** [src/agent/orchestrator.py:545-550](src/agent/orchestrator.py#L545-L550)
```python
evaluated_candidates = {}  # {nct_id: confidence_level}
```

After each `reason_soft_constraints` call, confidence level extracted and tracked ([lines 723-733](src/agent/orchestrator.py#L723-L733)). Model sees progress via enriched `patient_summary` showing what's been evaluated.

#### Tool Definitions & LLM Control

**Tool Definitions:** [src/agent/orchestrator.py:230-544](src/agent/orchestrator.py#L230-L544)

4 tools available to LLM:
1. **search_trials(condition, location, status, page_token)** → trials matching patient condition
   - Hard constraints pre-applied (age, sex filtered before results reach agent)
   - Uses [src/clinicaltrials/client.py:TrialSearcher](src/clinicaltrials/client.py)
   
2. **get_trial_detail(nct_id)** → full trial details, including free-text eligibility criteria
   - Fetches details selectively (LLM decides which trials warrant deep read)
   
3. **reason_soft_constraints(nct_id)** → confidence level (likely_eligible, possibly_eligible, likely_not_eligible)
   - LLM reads trial's eligibility criteria and evaluates against patient profile
   - Returns JSON: `{confidence: str, rationale: str}`
   - Uses [src/cleaning/reasoning.py:EligibilityReasoner](src/cleaning/reasoning.py)
   
4. **update_patient_profile(age, sex, condition, disease_stage, prior_treatments, location_preference, ...)** → persists data
   - Called immediately when patient mentions required fields (condition, age, sex)
   - Optional fields stored as patient volunteers info

#### Stopping Criteria & Early Termination

**Stopping Logic:** [src/agent/orchestrator.py:825-860](src/agent/orchestrator.py#L825-L860)

Agent stops when either condition is met:
1. **Natural stop:** 3-8 candidates passed both hard AND soft constraints
2. **Refinement exhaustion:** 2 targeted searches yielded no viable candidates

**Early Termination on Sufficient Candidates**
- After each iteration, checks: if 3+ evaluated candidates, present and exit ([lines 832-838](src/agent/orchestrator.py#L832-L838))
- Reduces latency without waiting for synthesis phase
- Helper method `_build_qualified_response()` formats ranked output

#### API Key Rotation & Retry Strategy

**Two Modes:** [src/config.py](src/config.py)

**DEVELOPMENT MODE** (default):
- 10 free-tier Google API keys with 2-model fallback
- Automatic rotation on 429 (quota exhausted)
- Exponential backoff: `min(2^retry_count, 30)` seconds max ([src/agent/orchestrator.py:607](src/agent/orchestrator.py#L607))
- If primary model quota exhausted, tries fallback model (gemini-3.6-flash)

**PRODUCTION MODE**:
- Single paid API key
- Exponential backoff with 30-second cap
- Circuit breaker pattern for persistent failures

**Key Code:** [src/agent/orchestrator.py:555-625](src/agent/orchestrator.py#L555-L625)
- Retry loop with proper backoff capping
- Parallel execution handling ([lines 704-718](src/agent/orchestrator.py#L704-L718)): processes ALL results before raising quota error
- Partial failure resilience: if some parallel tasks hit 429, continues with successful results

#### Robustness & Error Handling

**Graceful Error Fallback**
- If terminal error (400, exhausted quota) interrupts main loop, synthesize results from evaluated candidates
- Method `_get_qualified_candidates()` filters and ranks by confidence
- Returns partial but useful results instead of error ([lines 845-858](src/agent/orchestrator.py#L845-L858))

**Conversation History Validation**
- Method `_validate_conversation_history()` detects structural corruption before Gemini API call
- Reduces history_tail from 19 to 14 items (line 821) — more conservative truncation
- If validation fails, triggers error fallback instead of risking error 400 ([lines 616-623](src/agent/orchestrator.py#L616-L623))

---

## 3. ClinicalTrials.gov Integration

### Requirement
Query public API to search for recruiting trials matching patient condition/location.

### Solution

**API Client:** [src/clinicaltrials/client.py](src/clinicaltrials/client.py)

**TrialSearcher** class wraps public API:
- Base URL: `https://clinicaltrials.gov/api/v2/studies`
- Selective fields fetching (reduces payload, improves performance)
- Retry logic with exponential backoff ([lines 85-120](src/clinicaltrials/client.py#L85-L120))
- Structured error handling: `ClinicalTrialsAPIError` distinguishes connection failures from empty results

**Search Query:**
```python
search_trials(
    condition="lung cancer",
    location="California",
    status="RECRUITING",
    page_token=None
)
```

Returns trial summaries with: NCT ID, title, status, phases, recruitment status, nearest location.

**Design Decision:** Retry with exponential backoff (1s, 2s, 4s) rather than external library — keeps dependencies minimal, behavior explicit and testable.

---

## 4. Data Cleaning

### Requirement
Handle messy data: structure eligibility criteria blob, normalize dates, pull only needed fields.

### Solution

**Two-Part Cleaning Strategy:**

#### Hard Constraints → Deterministic Filtering

**EligibilityFilter:** [src/cleaning/eligibility.py](src/cleaning/eligibility.py)

Checks structured fields before ANY LLM call:
- **Age range:** patient age within trial's min/max
- **Sex:** patient sex matches trial requirement
- **Healthy volunteer status:** filtered by trial's `healthyVolunteers` flag

Example:
```python
filter = EligibilityFilter(min_age=50, max_age=75, sex="Female", healthy_volunteers_only=False)
filter.check_patient_eligible(age=55, sex="Female")  # True
```

**Why separate:** These checks are deterministic, cheap (few lines of Python), and filter obvious non-matches before wasting LLM calls. Only trials passing hard constraints reach the reasoner.

**Integration:** Hard filters applied server-side during `search_trials` call ([src/agent/orchestrator.py:656-670](src/agent/orchestrator.py#L656-L670)), so agent never sees ineligible trials.

#### Soft Constraints → LLM Reasoning

**EligibilityReasoner:** [src/cleaning/reasoning.py](src/cleaning/reasoning.py)

Reads trial's free-text eligibility criteria and reasons about:
- Disease stage match (patient stage vs trial inclusion/exclusion)
- Prior treatments (which treatments disqualify? which help?)
- Biomarker status (required? available?)
- Disease subtype match
- Other unstructured clinical criteria

**Example:**
```python
reasoner = EligibilityReasoner(patient_profile={
    "condition": "lung cancer",
    "disease_stage": "stage 3b",
    "prior_treatments": ["chemotherapy", "radiation"]
})

confidence = reasoner.reason_trial(
    trial_criteria="Inclusion: NSCLC stage II-IV, prior chemo OK. Exclusion: active infection",
    trial_nct_id="NCT06686771"
)
# Returns: "likely_eligible" or "possibly_eligible" with rationale
```

**Why LLM:** Free-text eligibility criteria cannot be checked programmatically. LLM reads context, understands medical nuance, assigns confidence level. This is where LLM's judgment adds real value.

#### Eligibility Criteria Parsing

**Hybrid Regex+LLM Approach:** [src/cleaning/reasoning.py:90-150](src/cleaning/reasoning.py#L90-L150)

- **Regex first:** standard headers ("Inclusion Criteria:", "Exclusion Criteria:") parsed deterministically for speed
- **LLM fallback:** non-standard formats sent to Gemini API to structure
- **Benefit:** fast on well-formatted trials, handles edge cases, explicit and testable

#### Date Normalization

Raw API dates ("2024-03", "2024-03-15") normalized to Python `date` objects:
```python
start_date: Optional[date]  # Pydantic model field
# Parsed in trial_from_json() helper
```

**Design Decision:** Normalize even though nothing consumes dates yet — keeps cleaning module cohesive and ready for future use without refactoring. Dates are structurally similar to ages (unstructured strings requiring parsing).

---

## 5. Eligibility Reasoning & Output (inherent in Agent Loop)

### Requirement
For each candidate trial: determine fit, return label (likely/possibly/likely not eligible), plain-language rationale, link to trial.

### Solution

**Ranking & Presentation:** [src/agent/orchestrator.py:845-880](src/agent/orchestrator.py#L845-L880)

**Helper Method: `_get_qualified_candidates()`**
- Filters `evaluated_candidates` dict for candidates with confidence >= "possibly_eligible"
- Sorts by confidence: `likely_eligible` first, then `possibly_eligible`
- Returns top 8 (spec limit)
- If more available, notes "There are [N] more matching trials"

**Output Format:**
```json
{
  "session_id": "...",
  "response": "Clinical trials matching your profile:\n\n
    1. [Trial Title] (NCT...) - possibly eligible\n
    2. [Trial Title] (NCT...) - likely eligible\n
    ...\n
    Discuss these with your doctor to determine best fit."
}
```

Each trial includes:
- ✅ **NCT ID + Title** (from trial details)
- ✅ **Status** (RECRUITING, ACTIVE_NOT_RECRUITING, etc.)
- ✅ **Confidence level** (likely_eligible, possibly_eligible)
- ✅ **Plain-language rationale** (from soft constraint reasoning)
- ✅ **Link to trial** (implicit in NCT ID: `https://clinicaltrials.gov/study/NCT...`)

**Design Decision:** Confidence levels assigned by LLM during soft constraint reasoning, not post-hoc. Each trial has explicit reasoning attached (from `reason_soft_constraints` rationale field).

---

## 6. Safety & Framing

### Requirement
Patient-facing: frame as candidate matches for doctor discussion, never as definitive eligibility or medical advice.

### Solution

**System Prompt Guidance:** [src/agent/orchestrator.py:138-165](src/agent/orchestrator.py#L138-L165)

Agent trained to:
- Present trials as candidates for discussion with doctor, not recommendations
- Be explicit about uncertainty ("based on trial criteria, this might be worth discussing")
- Acknowledge soft-constraint reasoning uncertainty vs. structured field certainty
- Never discourage care-seeking

**Emergency Handling:** [src/agent/orchestrator.py:166-175](src/agent/orchestrator.py#L166-L175)

If patient describes medical emergency (severe pain, difficulty breathing, chest pain), agent:
1. Stops trial discussion immediately
2. Directs to call 911 or go to ER
3. Does not re-engage on trials until emergency addressed

**Example Output Frame:**
```
These are candidates to discuss with your oncologist — not recommendations or guarantees.
Your doctor can help determine which best fit your specific situation.
```

---

## 7. Testing

### Coverage

**Test Suite:** [tests/](tests/)
- **54 unit tests passing** (excludes LLM tests by default)
- **Smoke tests:** server start, session persistence, `/chat` endpoint
- **Orchestrator tests:** agent decision logic, stopping criteria, candidate tracking
- **Cleaning tests:** hard constraints, soft constraint reasoning, date parsing
- **Integration tests:** ClinicalTrials.gov API client (note: 5 SSL tests skipped on Windows)

**Pytest Configuration:** [pytest.ini](pytest.ini)
- `asyncio_mode = auto` for async test support
- LLM tests marked and excluded by default (require API keys, cost money)
- Run all: `pytest -v`
- Run with LLM: `pytest -m llm -v`

**Key Test Files:**
- [tests/test_orchestrator.py](tests/test_orchestrator.py): agent loop, stopping, candidate tracking
- [tests/test_cleaning.py](tests/test_cleaning.py): hard/soft constraints, date parsing
- [tests/test_candidate_tracking.py](tests/test_candidate_tracking.py): dynamic evaluation tracking
- [tests/test_smoke.py](tests/test_smoke.py): server, session persistence

---

## 8. Robustness & Optimization Features

### Graceful Error Fallback
When terminal error occurs (quota exhausted, API error 400), agent synthesizes results from evaluated candidates instead of returning error to patient.
- Code: [src/agent/orchestrator.py:845-858](src/agent/orchestrator.py#L845-L858)

### Early Termination
Agent exits as soon as 3+ qualified candidates accumulate, without waiting for max iterations.
- Reduces response latency
- Aligns with system prompt requirement (3-8 candidates)
- Code: [src/agent/orchestrator.py:832-838](src/agent/orchestrator.py#L832-L838)

### Conversation History Validation
Detects and prevents corruption of conversation history before API calls.
- Conservative truncation: 14 items max (line 821)
- Validates structure before Gemini API call
- Triggers error fallback if corruption detected
- Code: [src/agent/orchestrator.py:616-623](src/agent/orchestrator.py#L616-L623), [_validate_conversation_history](src/agent/orchestrator.py#L870-885)

### Complete Implementation
All 6 assignment requirements fully implemented:
- Hard/soft constraint filtering producing ranked shortlist
- Dynamic candidate tracking with confidence levels
- API key rotation with exponential backoff (capped at 30 seconds)
- Multi-mode support (DEVELOPMENT/PRODUCTION)

---

## Key Architectural Decisions

| Decision | Benefit |
|----------|---------|
| Manual LLM loop (not framework) | Every decision point visible; LLM genuinely drives control flow |
| Hard/soft constraint split | Cheap deterministic checks first; LLM only on trials worth reasoning |
| Stateless API (state in request) | Simple, horizontally scalable, no session store needed |
| server-authoritative session IDs | Prevents session fixation; security-first design |
| Hybrid regex+LLM eligibility parsing | Fast on well-formatted data, handles edge cases |
| In-memory session store | Fits scope (single patient, ephemeral); upgrade to Redis for production |
| google-genai SDK | Avoids API credit requirements during dev; same philosophy as other LLM SDKs |

---

## Performance & Scale Considerations

**Optimization Implemented:**
- **Early trigger:** presents 3+ candidates without waiting for synthesis (reduces latency)
- **Parallel trial detail fetches:** `asyncio.gather()` fetches multiple trial details concurrently
- **Selective API fields:** only fetch fields needed (reduces payload, improves API performance)
- **Hard constraint pre-filtering:** eliminates obvious non-matches before LLM reasoning

**For Production Scale:**
1. Move session store to Redis (in-memory dict only safe for single process)
2. Add per-session locking (in-memory dict not concurrent-safe)
3. Cache trial details (avoid re-fetching same trial across sessions)
4. Monitor API quota usage and plan key rotation strategy

---

## Conclusion

This solution delivers a **complete, production-adjacent clinical trial discovery tool** that genuinely demonstrates LLM agent design, data cleaning, and full-stack engineering:

1. ✅ **Agent design:** LLM drives control flow; visible, testable decision points
2. ✅ **Data cleaning:** hard/soft constraint separation; structured + unstructured handling
3. ✅ **Engineering quality:** clean code, comprehensive tests, clear docs, architectural coherence
4. ✅ **Safety & UX:** frames results appropriately, handles edge cases, graceful degradation

