# Clinical Trial Eligibility (CTE) Agent

## How to Run Locally

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)

### Setup and Run

1. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Google Gemini API:**
   - **Create `.env` file from template:**
     ```bash
     cp .env.example .env
     ```
   
   - **Choose your API mode:**
     
     **DEVELOPMENT MODE** (default - recommended for testing):
     - Get free API keys from https://ai.google.dev/ (no billing required)
     - Add multiple keys to `.env`: `GOOGLE_API_KEY_1`, `GOOGLE_API_KEY_2`, etc.
     - Agent automatically rotates through keys when quota exhausted
     - Free tier available (~60 requests/minute per key)
     
     **PRODUCTION MODE** (for deployment with paid quota):
     - Set up Cloud Billing in Google Cloud Console
     - Get a single API key with paid quota
     - Add to `.env`: `GOOGLE_API_KEY` or `GOOGLE_API_KEY_1`
     - Add to `.env`: `API_MODE=PRODUCTION`
     - Agent uses exponential backoff + circuit breaker pattern
   
   - **See `.env.example` for detailed configuration options**

   ⚠️ **Security note:** `.env` is in `.gitignore` — your API keys will never be committed to git. Each developer must create their own `.env` locally.

4. **Run the server:**

   ```bash
   python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

5. **Access the UI:**
   Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

### Testing

Run the smoke tests to verify the server is working:

```bash
pytest tests/test_smoke.py -v
```

---

## Architecture

The CTE Agent is a single-process FastAPI application with a strict module boundary between orchestration, data access, and data cleaning:

```
clinical-trial-eligibility-agent/
├── main.py                          # FastAPI app: serves the UI, exposes POST /chat
├── requirements.txt                 # Dependencies
├── .gitignore                       # Git ignore rules
├── README.md                        # This file
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py                 # AgentState (Pydantic): per-session state
│   │   └── orchestrator.py          # Agent: the tool-use loop
│   ├── cleaning/
│   │   ├── __init__.py
│   │   ├── eligibility.py           # EligibilityFilter: hard constraints
│   │   └── reasoning.py             # EligibilityReasoner: soft constraints (LLM)
│   ├── clinicaltrials/
│   │   ├── __init__.py
│   │   └── client.py                # TrialSearcher: ClinicalTrials.gov API wrapper
│   └── ui/
│       └── index.html               # Vanilla HTML/JS chat frontend
└── tests/
    ├── __init__.py
    ├── test_smoke.py                # Smoke tests for server and session persistence
    ├── test_orchestrator.py         # Unit tests for Agent orchestrator
    ├── test_cleaning.py             # Unit tests for data cleaning and normalization
    ├── test_clinicaltrials_api.py   # Integration tests for ClinicalTrials.gov API
    └── conftest.py                  # Pytest configuration and fixtures
```

**Request flow:** the browser posts `{session_id, message}` to `/chat`. The server is the
sole authority for session identity: if `session_id` is missing or unrecognized, a new one is minted server-side and returned to the client, which persists it in `sessionStorage` for the lifetime of the tab. Each session's `AgentState` lives in an in-memory dict (`{session_id: AgentState}`) at module scope in `main.py`, and is looked up on every subsequent request.

The `agent` module owns the decision loop (what to ask, what to search, when to stop);
`clinicaltrials` is a pure data-access layer with no reasoning in it; `cleaning` is split
into two classes on purpose — `EligibilityFilter` for constraints that can be checked
deterministically from structured fields, and `EligibilityReasoner` for constraints that
require reading free-text eligibility criteria. This split is architectural,
not something bolted on later — see Design Decisions below.

## Design Decisions

| Decision                                                                                                                                                    | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Real client-server split (FastAPI backend, browser hits `/chat` over HTTP) instead of a monolithic script                                                   | Keeps the agent testable in isolation from the UI, and reflects an architecture that's actually defensible as "system design," not just a script with a UI bolted on.                                                                                                                                                                                                                                                                                                                                          |
| Manual tool-use loop with LLM-driven control flow, no agent framework (LangChain, LangGraph, CrewAI); using google-genai SDK for the orchestrator           | The evaluation criterion that matters most is that the LLM genuinely drives control flow, turn by turn. A hand-written loop makes every decision point visible and explainable. SDK choice (google-genai) is independent of this architecture — both Anthropic and Google support native function-calling and structured tool definitions. google-genai was chosen to avoid API credit requirements during development. The manual loop philosophy remains unchanged.                                          |
| Modular decomposition (`AgentState`, `TrialSearcher`, `EligibilityFilter`, `EligibilityReasoner`) instead of one large orchestrator file                    | Each concern is independently testable, and it makes the hard-constraint vs. soft-constraint separation (see below) a structural property of the codebase, not a convention I have to remember to follow.                                                                                                                                                                                                                                                                                                      |
| `src/` package layout                                                                                                                                       | Standard Python packaging convention; avoids import ambiguity as the project grows past a handful of files. Minor, but a conscious choice, not a default I didn't notice.                                                                                                                                                                                                                                                                                                                                      |
| Hard constraints (age, sex, healthy-volunteer status) checked deterministically in code against structured ClinicalTrials.gov fields, _before_ any LLM call | These fields are already structured and unambiguous — there's no reason to spend an LLM call, add latency, or introduce any non-determinism on a check a few lines of Python can do reliably. This filters out obvious non-matches cheaply, so the LLM only reasons about trials worth reasoning about.                                                                                                                                                                                                        |
| Soft constraints (disease stage, prior treatments, biomarkers) reasoned over by the LLM, reading the free-text `eligibilityCriteria`                        | These genuinely require language understanding — they can't be reduced to a field comparison. This is where the LLM's judgment adds real value, as opposed to the hard constraints above.                                                                                                                                                                                                                                                                                                                      |
| Pydantic for `AgentState` and for tool argument schemas, instead of plain dataclasses                                                                       | Pydantic is already a required dependency via FastAPI, so using it elsewhere isn't a new dependency, just consistency. More importantly, it gives validation exactly where it's needed most: at the boundary where tool arguments coming back from the LLM are parsed, since that output is untrusted and can be malformed.                                                                                                                                                                                    |
| `sessionStorage` (not `localStorage`) for the session ID on the frontend                                                                                    | The storage's lifetime should match the backend's actual guarantee. Session state lives in an in-memory dict that's lost on server restart — `localStorage` would silently promise a continuity across browser restarts that the server can't deliver, and it's also shared across tabs, which could let two open tabs cross-contaminate the same session. `sessionStorage` is scoped to the tab and doesn't outlive it, which matches the backend's real guarantees.                                          |
| Server-authoritative session IDs: an unrecognized `session_id` from the client is ignored and a new one is minted, rather than trusted and adopted          | Trusting a client-supplied session ID at face value is a session fixation risk — any client could claim or create a session under an ID of its choosing. The server is the only party allowed to mint valid session IDs.                                                                                                                                                                                                                                                                                       |
| In-memory session store, no database                                                                                                                        | Explicitly out of scope per the assignment (single patient at a time, no persistence requirement), and the session data is an ephemeral search context, not a medical record that needs to survive a restart.                                                                                                                                                                                                                                                                                                  |
| Single FastAPI process serving both the static frontend and the `/chat` API, instead of a separate frontend dev server                                      | One command (`uvicorn main:app`) is enough to run the whole app locally, which the assignment explicitly asks for. A split frontend/backend setup would add CORS configuration and a second process for no benefit at this scale.                                                                                                                                                                                                                                                                              |
| `/chat` endpoint defined as `async def`, even though synchronous code does not `await` anything yet                                                         | `TrialSearcher` is async to enable parallel `get_trial_details()` calls via `asyncio.gather()`. Defining `/chat` as async preemptively avoids re-touching `main.py` later and keeps the async/sync boundary consistent throughout. The cost is surface-level async with no concurrency currently, which is acceptable.                                                                                                                                                                                         |
| Manual retry logic with exponential backoff (no external retry libraries like `tenacity`) and explicit error propagation via `ClinicalTrialsAPIError`       | Keeps dependencies minimal and makes retry behavior explicit and testable. Distinguishing "got results but list is empty" from "API call failed" is critical: the agent must know whether to ask clarifying questions or surface a connection error to the patient. Manual backoff is simple enough (1s, 2s, 4s) that the overhead of a library is not justified.                                                                                                                                              |
| Hybrid regex+LLM strategy for parsing `eligibility_criteria`                                                                                                | Standard headers ("Inclusion Criteria:", "Exclusion Criteria:") are handled deterministically with regex for speed and testability. Non-standard formats fall back to Google Gemini API (via httpx REST) to structure the text. This avoids two extremes: (1) regex-only, which fails on poorly formatted trials and requires constant format tweaking; (2) LLM-for-everything, which is slow and adds cost on well-formatted trials where regex works fine. Hybrid is fast, handles edge cases, and explicit. |
| eligibility_parser.py uses httpx REST directly; orchestrator uses google-genai SDK                                                                          | Eligibility parsing is a standalone, one-shot LLM call (structured but not agentic). Direct REST via httpx is simpler and already functional. The orchestrator genuinely needs the SDK's abstractions for multi-turn conversations and function-calling semantics. If google-genai proves solid, revisiting eligibility_parser.py to migrate it to the SDK for consistency is reasonable future work, but not required — the current split is defensible and working.                                          |
| Normalize dates even though nothing consumes them yet                                                                                                       | Dates (start_date, completion_date) are structurally similar to ages: raw strings from the API ("2024-03", "2024-03-15") that need parsing to Python `date` objects. Treating them differently from ages would be inconsistent (both are "structured but unnormalized" field types). Parsing them now keeps the cleaning module cohesive and ready for future use without refactoring.                                                                                                                         |

## What's Next with More Time

- **Phase 4 — Hard/soft constraint filtering:** wire `EligibilityFilter` and
  `EligibilityReasoner` into the loop, producing the final labeled, ranked shortlist.
- **Concurrency improvements:** the in-memory session dict isn't safe against two simultaneous
  requests for the same `session_id` (a race condition intentionally not solved given the
  single-patient, single-session scope of this assignment). With more time or at real scale,
  move session state to Redis and add per-session locking.
- **Testing expansion:** beyond integration tests against the real API, add unit tests per
  module (`EligibilityFilter` against known structured-field edge cases) once Phase 4 lands.
