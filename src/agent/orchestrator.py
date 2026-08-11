"""
Agent Orchestrator: Main agentic loop.

Responsibility (to be implemented in Phase 2):
- Accept a user message and current AgentState
- Call Claude with the state as context and the new message
- Parse Claude's response to extract tool_use calls (if any)
- Execute tools (TrialSearcher, EligibilityFilter, EligibilityReasoner)
- Update AgentState with results and conversation history
- Decide whether to continue the loop or return a final answer

This is where the "agent" becomes intelligent: instead of a fixed pipeline,
Claude decides what to do next (ask clarifying questions, search, filter, reason, or stop).
"""

from src.agent.state import AgentState
from typing import Tuple


class Agent:
    """
    Orchestrates the agentic loop for clinical trial eligibility matching.

    Methods (to be implemented):
    - process_message(state: AgentState, user_message: str) -> Tuple[AgentState, str]
      Runs one iteration of the agent loop: LLM call -> tool execution -> state update -> response.
    """

    def process_message(self, state: AgentState, user_message: str) -> Tuple[AgentState, str]:
        """
        Process a single user message and advance the agent loop.

        Args:
            state: Current AgentState for this session
            user_message: Patient's input (free text or clarification)

        Returns:
            (updated_state, agent_response): Updated state and text response to send to patient

        Implementation details (Phase 2):
        1. Append user_message to conversation_history
        2. Build prompt with patient context + conversation history
        3. Call Claude with tool definitions (search, fetch detail, ask question)
        4. Parse response for tool_use blocks
        5. Execute each tool, collect results
        6. Feed results back to Claude for synthesis (if needed)
        7. Extract final text response
        8. Update state and return
        """
        raise NotImplementedError("Agent loop implemented in Phase 2")
