from typing import Optional
from langgraph.graph import MessagesState


# ---------- Custom Graph State for the Manager Agent ----------
class GraphState(MessagesState):
    user_query: Optional[str]
    current_page: Optional[dict] = None
    agent: Optional[str] = None
    relevant_articles_for_user: Optional[list] = None  # List of articles id found for the user query
    
# ---------- Sub-agent State----------
class ReactAgentState(GraphState):
    remaining_steps: int