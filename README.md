# Clinical Trial Eligibility (CTE) Agent

## How to Run Locally

### Prerequisites

- Python 3.9 or higher
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

3. **Set up your Anthropic API key:**

   Create a `.env` file in the root of the project:

   ```bash
   echo "ANTHROPIC_API_KEY=your_key_here" > .env
   ```

   Then open the `.env` file and replace `your_key_here` with your actual Anthropic API key.
   (Note: `.env` is covered by `.gitignore`, so it will never be committed to the repository.)

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
│   │   └── orchestrator.py          # Agent: the tool-use loop (Phase 3+)
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
    └── test_smoke.py                # Smoke tests for server and session persistence

**Request flow:** the browser posts `{session_id, message}` to `/chat`. The server is the
sole authority for session identity: if `session_id` is missing or unrecognized, a new one is minted server-side and returned to the client, which persists it in `sessionStorage` for the lifetime of the tab. Each session's `AgentState` lives in an in-memory dict (`{session_id: AgentState}`) at module scope in `main.py`, and is looked up on every subsequent request.

The `agent` module owns the decision loop (what to ask, what to search, when to stop);
`clinicaltrials` is a pure data-access layer with no reasoning in it; `cleaning` is split
into two classes on purpose — `EligibilityFilter` for constraints that can be checked
deterministically from structured fields, and `EligibilityReasoner` for constraints that
require reading free-text eligibility criteria. This split is architectural from Phase 0,
not something bolted on later — see Design Decisions below.

## Design Decisions

| Decision                                                                                                                                                    | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Real client-server split (FastAPI backend, browser hits `/chat` over HTTP) instead of a monolithic script                                                   | Keeps the agent testable in isolation from the UI, and reflects an architecture that's actually defensible as "system design," not just a script with a UI bolted on.                                                                                                                                                                                                                                                                                                 |
| Native Anthropic SDK with a manual tool-use loop, no agent framework (LangChain, LangGraph, CrewAI)                                                         | The evaluation criterion that matters most here is that the LLM genuinely drives control flow, turn by turn. A hand-written loop makes every decision point visible and explainable; a framework's `AgentExecutor` hides that behind internal prompting I didn't write and can't fully account for.                                                                                                                                                                   |
| Considered and rejected: Claude Managed Agents (Anthropic's hosted agent runtime, public beta)                                                              | It's built for multi-agent coordination (a coordinator delegating to sub-agents), not a single agent with a handful of tools — using it here would be scope mismatch, not a fit. It's also a recent beta feature, which adds delivery risk for a fixed-scope deliverable.                                                                                                                                                                                             |
| Modular decomposition (`AgentState`, `TrialSearcher`, `EligibilityFilter`, `EligibilityReasoner`) instead of one large orchestrator file                    | Each concern is independently testable, and it makes the hard-constraint vs. soft-constraint separation (see below) a structural property of the codebase, not a convention I have to remember to follow.                                                                                                                                                                                                                                                             |
| `src/` package layout                                                                                                                                       | Standard Python packaging convention; avoids import ambiguity as the project grows past a handful of files. Minor, but a conscious choice, not a default I didn't notice.                                                                                                                                                                                                                                                                                             |
| Hard constraints (age, sex, healthy-volunteer status) checked deterministically in code against structured ClinicalTrials.gov fields, _before_ any LLM call | These fields are already structured and unambiguous — there's no reason to spend an LLM call, add latency, or introduce any non-determinism on a check a few lines of Python can do reliably. This filters out obvious non-matches cheaply, so the LLM only reasons about trials worth reasoning about.                                                                                                                                                               |
| Soft constraints (disease stage, prior treatments, biomarkers) reasoned over by the LLM, reading the free-text `eligibilityCriteria`                        | These genuinely require language understanding — they can't be reduced to a field comparison. This is where the LLM's judgment adds real value, as opposed to the hard constraints above.                                                                                                                                                                                                                                                                             |
| Pydantic for `AgentState` and (from Phase 3+) for tool argument schemas, instead of plain dataclasses                                                       | Pydantic is already a required dependency via FastAPI, so using it elsewhere isn't a new dependency, just consistency. More importantly, it gives me validation exactly where I need it most: at the boundary where I parse `tool_use` arguments coming back from the LLM, which is untrusted output that can be malformed.                                                                                                                                           |
| `sessionStorage` (not `localStorage`) for the session ID on the frontend                                                                                    | The storage's lifetime should match the backend's actual guarantee. Session state lives in an in-memory dict that's lost on server restart — `localStorage` would silently promise a continuity across browser restarts that the server can't deliver, and it's also shared across tabs, which could let two open tabs cross-contaminate the same session. `sessionStorage` is scoped to the tab and doesn't outlive it, which matches the backend's real guarantees. |
| Server-authoritative session IDs: an unrecognized `session_id` from the client is ignored and a new one is minted, rather than trusted and adopted          | Trusting a client-supplied session ID at face value is a session fixation risk — any client could claim or create a session under an ID of its choosing. The server is the only party allowed to mint valid session IDs.                                                                                                                                                                                                                                              |
| In-memory session store, no database                                                                                                                        | Explicitly out of scope per the assignment (single patient at a time, no persistence requirement), and the session data is an ephemeral search context, not a medical record that needs to survive a restart.                                                                                                                                                                                                                                                         |
| Single FastAPI process serving both the static frontend and the `/chat` API, instead of a separate frontend dev server                                      | One command (`uvicorn main:app`) is enough to run the whole app locally, which the assignment explicitly asks for. A split frontend/backend setup would add CORS configuration and a second process for no benefit at this scale.                                                                                                                                                                                                                                     |

## What's Next with More Time

- **Phase 1 — ClinicalTrials.gov integration:** implement `TrialSearcher` against the
  `/studies` endpoint with a constrained `fields` list, `status=RECRUITING` filtering, and
  pagination handling.
- **Phase 2 — Data cleaning:** parse the free-text `eligibilityCriteria` blob into structured
  `inclusion_criteria` / `exclusion_criteria` lists, with a regex-based first pass and an
  LLM-assisted fallback for studies that don't follow the common header format; normalize
  age fields and dates.
- **Phase 3 — Agent loop:** implement the actual tool-use loop in `orchestrator.py` —
  clarification requests, search, drill-down into promising candidates, explicit stopping
  condition.
- **Phase 4 — Hard/soft constraint filtering:** wire `EligibilityFilter` and
  `EligibilityReasoner` into the loop, producing the final labeled, ranked shortlist.
- **Concurrency:** the in-memory session dict isn't safe against two simultaneous requests
  for the same `session_id` (a race condition I'm aware of but chose not to solve, given the
  single-patient, single-session scope of this assignment). With more time, or at real
  scale, I'd move session state to Redis and add per-session locking.
- **Resilience:** no retry/backoff logic yet around ClinicalTrials.gov calls; would add
  timeout handling and a small retry policy before treating this as production-ready.
- **Testing:** current tests are smoke tests only; would add unit tests per module
  (`EligibilityFilter` against known structured-field edge cases, criteria parsing against a
  sample of real `eligibilityCriteria` text formats) once Phases 1-4 land.
