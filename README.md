# Clinical Trial Eligibility (CTE) Agent

A FastAPI-based agent that helps patients discover clinical trials they may be eligible for. Patients describe their medical situation in plain language, and the agent reasons through eligibility criteria to surface matched trials ranked by confidence.

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Add Google Gemini API keys to .env (see Configuration below)

# Run
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Open browser
http://localhost:8000
```

---

## Prerequisites

- **Python 3.12+**
- **pip** (Python package manager)
- **Google Gemini API keys** (free tier available at https://ai.google.dev/)

---

## Configuration

### API Mode Selection

The agent supports two modes to fit different deployment scenarios:

#### **DEVELOPMENT MODE** (default — recommended for testing)
- Uses **10 free-tier Google Gemini API keys** with automatic rotation
- Free quota: ~60 requests/minute per key
- Agent automatically switches to next key when quota exhausted (429 response)
- Fallback model: if primary model (gemini-3.7-flash) quota exhausted, tries gemini-3.6-flash
- Best for: local development, testing, prototyping

**Setup:**
```bash
# In .env
GOOGLE_API_KEY_1=your-key-1
GOOGLE_API_KEY_2=your-key-2
# ... up to GOOGLE_API_KEY_10
API_MODE=DEVELOPMENT  # optional (default)
```

#### **PRODUCTION MODE** (for deployment with billing)
- Uses **single paid-tier Google Gemini API key**
- Implements exponential backoff (max 30s cap) for rate limiting
- Circuit-breaker pattern: graceful degradation on persistent failures
- Best for: production deployment, high-traffic scenarios

**Setup:**
```bash
# In .env
GOOGLE_API_KEY=your-paid-key
API_MODE=PRODUCTION
```

⚠️ **Security:** `.env` is in `.gitignore` — API keys are never committed. Each developer creates their own `.env` locally.

### Complete Configuration Options

See [.env.example](.env.example) for all available environment variables and detailed explanations.

---

## Running the Agent

### Start Server
```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Test with curl
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I am 55 years old, female with lung cancer. Had chemotherapy. Looking for trials in California."
  }'
```

### Run Tests
```bash
# Run all tests (excludes LLM tests by default)
pytest -v

# Run with LLM tests (requires valid API keys and costs money)
pytest -m llm -v

# Run specific test file
pytest tests/test_orchestrator.py -v
```

---

## System Prompts and Instructions

The agent's decision-making is governed by a **system prompt** that defines how it collects patient information, when to search for trials, and how to present results.

**See [PROMPTS.md](PROMPTS.md)** for the complete, authoritative system prompt, including:
- How the agent asks clarifying questions
- When the agent stops and presents findings
- How confidence levels are assigned to trial matches
- How results are framed for patient discussion with doctors

The system prompt is the source of truth for agent behavior and is maintained in [src/agent/orchestrator.py](src/agent/orchestrator.py#L92-L220) for transparency.

---

## Architecture

The CTE Agent is a single-process FastAPI application with clear separation of concerns:

```
clinical-trial-eligibility-agent/
├── main.py                          # FastAPI: serves UI, exposes POST /chat
├── src/
│   ├── agent/
│   │   ├── orchestrator.py          # Core agent loop: decision logic, tool calls
│   │   └── state.py                 # AgentState (Pydantic): per-session state
│   ├── cleaning/
│   │   ├── eligibility.py           # EligibilityFilter: hard constraints (deterministic)
│   │   └── reasoning.py             # EligibilityReasoner: soft constraints (LLM)
│   ├── clinicaltrials/
│   │   └── client.py                # TrialSearcher: ClinicalTrials.gov API client
│   └── ui/
│       └── index.html               # Frontend: vanilla HTML/JS chat interface
└── tests/                           # Unit and integration tests
```

### Request Flow

1. Browser sends `{session_id, message}` to `/chat` endpoint
2. Server mints new session if `session_id` unknown (server is sole authority)
3. Server looks up `AgentState` in in-memory dict, runs agent loop
4. Agent decides: ask clarifying question? search trials? evaluate soft constraints? present results?
5. Server returns response + session_id (client persists in `sessionStorage`)

### Module Responsibilities

| Module | Responsibility | Reasoning |
|--------|---|---|
| **orchestrator.py** | Tool-use loop, LLM control flow, stopping criteria | LLM must genuinely drive decisions, not execute a fixed script |
| **state.py** | Session-scoped patient data, conversation history | Stateless API; state passed to every request |
| **eligibility.py** | Hard constraints (age, sex, healthy-volunteer status) | Deterministic checks on structured fields; no LLM overhead |
| **reasoning.py** | Soft constraints (disease stage, treatments, biomarkers) | Free-text eligibility criteria require language understanding |
| **client.py** | ClinicalTrials.gov API, data fetching, retry logic | Pure data-access layer; no reasoning |
| **index.html** | Chat UI, session persistence | Vanilla JS; no build process required

## Key Design Decisions

### Agent Architecture
- **Manual tool-use loop** (not LangChain/LangGraph): LLM genuinely drives control flow, every decision point is explicit
- **google-genai SDK** (not Anthropic): avoids API credit requirements during dev, philosophy remains the same

### Data Handling
- **Hard constraints** (age, sex) checked deterministically in Python before LLM calls — filters obvious non-matches cheaply
- **Soft constraints** (disease stage, treatments) reasoned by LLM over free-text eligibility criteria — requires language understanding
- **Hybrid regex+LLM** for parsing eligibility criteria: regex for standard headers, LLM fallback for unstructured formats

### Session & Security
- **In-memory session store**: no database needed, ephemeral session scope
- **Server-authoritative session IDs**: server mints new IDs, client-supplied IDs rejected (prevents session fixation)
- **sessionStorage** (not localStorage): respects backend lifetime guarantees, tab-scoped

### Infrastructure
- **Single FastAPI process**: serves UI + API from one command (no separate frontend server, no CORS overhead)
- **Pydantic validation**: validates LLM-generated tool arguments at system boundary
- **Async-ready**: `/chat` endpoint async to support parallel trial detail fetches via `asyncio.gather()`

---

## What's Implemented

This solution implements the full assignment scope:

### ✅ 1. Patient-Facing Interface  
Vanilla HTML/JS chat frontend — no build process. Patients describe their situation in free text, see ranked trial recommendations.

### ✅ 2. Intelligent Agent Loop
- Asks clarifying questions when eligibility info is missing (condition, age, sex)
- Decides its own API queries based on patient input
- Fetches full trial details selectively, not exhaustively
- Decides when sufficient candidates found (3-8 qualified) or searches exhausted
- **Robustness features:**
  - Presents results early when 3+ qualified candidates accumulated (no unnecessary delays)
  - Validates conversation state before API calls; graceful fallback if corruption detected
  - Graceful degradation: if error occurs mid-evaluation, presents best results from evaluated candidates

### ✅ 3. ClinicalTrials.gov Integration  
Queries public API, extracts recruiting trials matching patient condition/location/eligibility.

### ✅ 4. Data Cleaning  
- Eligibility criteria: parsed into structured inclusion/exclusion via regex + LLM
- Dates: normalized to Python `date` objects
- Selective fields: only fetches what's needed via API `fields=` parameter

### ✅ 5. Eligibility Reasoning & Output
- **Hard constraint filtering** ([EligibilityFilter](src/cleaning/eligibility.py)): age, sex, healthy-volunteer checks
- **Soft constraint reasoning** ([EligibilityReasoner](src/cleaning/reasoning.py)): disease stage, treatments, biomarkers via LLM
- **Ranked output**: candidates labeled as `likely_eligible`, `possibly_eligible`, ranked by confidence
- **Plain-language rationale**: explains why each trial was included
- **Links to ClinicalTrials.gov**: patients can verify details

### ✅ 6. Safety & Framing  
- Frames results as candidates for doctor discussion, never as medical advice
- Transparent about uncertainty in soft constraint reasoning
- Does not discourage care-seeking
- Handles medical emergencies (stops agent, redirects to 911)

### ✅ Testing  
- 54 unit tests passing (excludes integration tests by default)
- Smoke tests for server and session persistence
- Unit tests for orchestrator, data cleaning, API integration
- See [pytest.ini](pytest.ini) for configuration

---

## Future Improvements

- **Persistence**: move in-memory session dict to Redis for multi-process/multi-server deployments
- **Concurrency**: add per-session locking to handle simultaneous requests safely
- **Testing**: expand unit tests per module (e.g., `EligibilityFilter` with known edge cases)
- **Analytics**: track trial matches, confidence levels, and patient outcomes to improve reasoning
