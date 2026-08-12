"""
AgentState: Represents the state of a single patient conversation session.

This dataclass will hold:
- patient_profile: Structured info extracted from patient's free-text input (age, condition, location, etc.)
- conversation_history: List of (role, content) tuples for context in LLM calls
- candidate_trials: List of trials deemed potentially relevant to this patient
- search_attempts: Metadata about API calls made (to avoid redundant searches)

Implementation details deferred to Phase 3 (eligibility reasoning logic).
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, UTC


class AgentState(BaseModel):
    """
    Encapsulates the state of a single patient's conversation with the agent.

    Fields will include:
    - patient_profile (Phase 3)
    - conversation_history (Phase 3)
    - candidate_trials (Phase 4)
    - search_attempts (Phase 3)
    """
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))