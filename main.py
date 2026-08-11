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
from pydantic import BaseModel
from typing import Optional
import uuid
from pathlib import Path

from src.agent.state import AgentState

# Session storage: {session_id: AgentState}
sessions: dict[str, AgentState] = {}

app = FastAPI(title="Clinical Trial Eligibility Agent")


# ============================================================================
# Request/Response Models
# ============================================================================


class ChatRequest(BaseModel):
    """
    Client request to /chat endpoint.

    Fields:
    - session_id: Optional unique session identifier. If None, a new session is created.
    - message: Patient's message (free text).
    """

    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    """
    Server response from /chat endpoint.

    Fields:
    - session_id: Session identifier (new or existing)
    - response: Text response from the agent (placeholder for now)
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
    3. Return agent response (currently a placeholder; will call
       Agent.process_message() once the orchestrator loop is implemented)
    """

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

    # TODO: Replace with agent.process_message(state, request.message)
    # once the orchestrator loop is implemented (see src/agent/orchestrator.py)

    response = f"[Placeholder] You said: {request.message}"

    sessions[session_id] = state

    return ChatResponse(session_id=session_id, response=response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
