from typing import Optional
from langgraph.graph import MessagesState


# ---------- Custom Graph State for the Manager Agent ----------
class GraphState(MessagesState):
    user_query: Optional[str]
    current_page: Optional[dict] = None
    agent: Optional[str] = None
    modals: Optional[list] = None  # List of modals
    
# ---------- Sub-agent State----------
class ReactAgentState(GraphState):
    remaining_steps: int