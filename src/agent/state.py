"""
AgentState: Represents the state of a single patient conversation session.

Holds:
- patient_profile: Structured info extracted from patient's free-text input
- fetched_trial_details: Cache of TrialDetail objects to avoid redundant API calls
- session_id, created_at: Session metadata
- last_search_results: Most recent trial search results for graceful degradation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime, UTC
from src.clinicaltrials.client import TrialDetail, TrialSummary


class PatientProfile(BaseModel):
    """
    Structured patient information extracted from conversation.

    All fields are optional because the agent decides via system prompt
    what is indispensable. Validation ensures data integrity without
    blocking extraction flow.
    """
    age: Optional[int] = None
    sex: Optional[str] = None
    condition: Optional[str] = None
    disease_stage: Optional[str] = None
    prior_treatments: List[str] = Field(default_factory=list)
    location_preference: Optional[str] = None
    willing_to_travel: Optional[bool] = None
    other_notes: Optional[str] = None

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 0 or v > 150):
            raise ValueError("Age must be between 0 and 150")
        return v

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            valid_sexes = {"M", "F", "Other", "Unknown"}
            if v not in valid_sexes:
                raise ValueError(f"Sex must be one of {valid_sexes}")
        return v

    @field_validator("prior_treatments")
    @classmethod
    def validate_prior_treatments(cls, v: List[str]) -> List[str]:
        # Remove empty strings and strip whitespace
        cleaned = [item.strip() for item in v if item.strip()]
        return cleaned

    @field_validator("other_notes")
    @classmethod
    def validate_other_notes(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > 1000:
            raise ValueError("other_notes must be 1000 characters or less")
        return v


class AgentState(BaseModel):
    """
    Encapsulates the state of a single patient's conversation with the agent.

    Conversation history is maintained locally within the agent loop; this object
    persists patient profile and trial details across multiple turns.

    Fields:
    - session_id: Unique session identifier (server-authoritative)
    - created_at: When this session started
    - patient_profile: Extracted patient info (built up over turns)
    - fetched_trial_details: Cache of TrialDetail objects by NCT ID
    - last_search_results: Most recent trial search results
    """
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    patient_profile: PatientProfile = Field(default_factory=PatientProfile)
    fetched_trial_details: Dict[str, TrialDetail] = Field(default_factory=dict)
    last_search_results: List[TrialSummary] = Field(default_factory=list, description="Most recent trial search results for graceful degradation")