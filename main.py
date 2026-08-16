"""
CTE Agent: Main FastAPI application.

This is the entry point for the Clinical Trial Eligibility Agent.
It serves the HTML/JS frontend and exposes the /chat endpoint for agent interactions.

Session management:
- Sessions are stored in memory as {session_id: AgentState}
- Session IDs are UUIDs generated server-side. Server is the sole authority for valid session IDs.
- State persists across multiple POST requests from the same session
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import uuid
import asyncio
import logging
import os
from pathlib import Path
from datetime import datetime, UTC, timedelta

from src.agent.state import AgentState
from src.agent.orchestrator import Agent

# Configure logging to see all levels
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Session storage: {session_id: AgentState}
sessions: dict[str, AgentState] = {}

# Session TTL: sessions older than this are cleaned up
SESSION_TTL_HOURS = 24
CLEANUP_INTERVAL_SECONDS = 3600  # Run cleanup every hour

# Agent instance (stateless; all state lives in AgentState)
try:
    agent = Agent()
except Exception as e:
    raise RuntimeError(f"Failed to initialize Agent: {e}") from e

app = FastAPI(title="Clinical Trial Eligibility Agent")


# ============================================================================
# Session Cleanup (prevent memory leak)
# ============================================================================

async def cleanup_expired_sessions():
    """
    Periodically remove sessions that haven't been accessed in SESSION_TTL_HOURS.
    Prevents memory leak from accumulating inactive sessions.
    Runs every CLEANUP_INTERVAL_SECONDS.
    """
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        now = datetime.now(UTC)
        cutoff_time = now - timedelta(hours=SESSION_TTL_HOURS)
        expired_ids = [
            sid for sid, state in sessions.items()
            if state.created_at < cutoff_time
        ]
        for sid in expired_ids:
            del sessions[sid]
        if expired_ids:
            logger.info(f"Cleanup: removed {len(expired_ids)} expired session(s), {len(sessions)} active")


@app.on_event("startup")
async def startup_cleanup_task():
    """Start background cleanup task on server startup."""
    asyncio.create_task(cleanup_expired_sessions())


# ============================================================================
# Request/Response Models
# ============================================================================


class ChatRequest(BaseModel):
    """
    Client request to /chat endpoint.

    Fields:
    - session_id: Optional unique session identifier. If None, a new session is created.
      Must be a valid UUID if provided.
    - message: Patient's message (free text). Must be 1-5000 characters.
    """

    session_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=5000)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                uuid.UUID(v)
            except ValueError:
                raise ValueError("session_id must be a valid UUID")
        return v

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be empty or whitespace-only")
        return v.strip()


class ChatResponse(BaseModel):
    """
    Server response from /chat endpoint.

    Fields:
    - session_id: Session identifier (new or existing)
    - response: Text response from the agent
    """

    session_id: str
    response: str


# ============================================================================
# Routes
# ============================================================================


@app.get("/", response_class=FileResponse)
def serve_ui():
    """
    Serve the frontend HTML.

    Returns the main UI at src/ui/index.html.
    This is where patients interact with the agent via browser.
    """
    ui_path = Path(__file__).parent / "src" / "ui" / "index.html"
    return FileResponse(ui_path)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint. Receives patient message and returns agent response.

    Security note: Session IDs are server-authoritative. If the client provides
    a session_id that doesn't exist, we ignore it and generate a new one.
    This prevents session fixation attacks.

    Flow:
    1. Validate session_id (server-side check)
    2. Retrieve or create AgentState for this session
    3. Call agent.process_message() with error handling
    4. Return response or user-friendly error message

    Error handling: If agent fails, return HTTP 200 with error message
    instead of 500, so client can display graceful error to patient.
    """
    try:
        # Validate session_id: if not provided OR doesn't exist in our sessions dict,
        # generate a new one. Server is the only authority for valid session IDs.
        if request.session_id is None or request.session_id not in sessions:
            session_id = str(uuid.uuid4())
        else:
            session_id = request.session_id

        # Get or create state
        if session_id not in sessions:
            sessions[session_id] = AgentState(session_id=session_id)

        state = sessions[session_id]

        # Process message through agent
        updated_state, agent_response = await agent.process_message(state, request.message)

        # Update session state
        sessions[session_id] = updated_state

        return ChatResponse(session_id=session_id, response=agent_response)

    except Exception as e:
        logger.exception(f"Chat endpoint error for session {session_id}")

        # Graceful degradation: if we have session state, return partial results
        if session_id in sessions:
            state = sessions[session_id]

            # Build partial response from whatever we completed
            partial_response = []

            # If we have patient profile, acknowledge it
            if state.patient_profile.age:
                partial_response.append(
                    f"Patient profile recorded: {state.patient_profile.age}yo "
                    f"{state.patient_profile.sex or '?'} from "
                    f"{state.patient_profile.location_preference or 'unknown location'}"
                )

            # If we completed a search, mention results
            if state.last_search_results:
                partial_response.append(
                    f"\nFound {len(state.last_search_results)} clinical trials matching your condition. "
                    f"Trial evaluation is temporarily unavailable."
                )

            # Add error note
            partial_response.append(
                f"\n⚠️ Service temporarily unavailable. Please try again in a moment."
            )

            response_text = "".join(partial_response) if partial_response else (
                "I apologize, but I encountered an error processing your message. Please try again in a moment."
            )
        else:
            response_text = "I apologize, but I encountered an error processing your message. Please try again in a moment."

        return ChatResponse(session_id=session_id, response=response_text)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)
