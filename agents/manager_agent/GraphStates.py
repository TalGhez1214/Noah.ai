from typing import Optional
from langgraph.graph import MessagesState
from typing import Any, Optional, TypedDict

# ---------- Custom Graph State for the Manager Agent ----------
class UIItems(TypedDict):
    type: str            # e.g., "summary", "article", "highlight" etc.
    data: Any            # the render-ready payload for your UI component

class GraphState(MessagesState):
    user_query: Optional[str]
    current_page: Optional[dict] = None
    agent: Optional[str] = None
    ui_items: Optional[UIItems] = None 
# ---------- Sub-agent State----------
class ReactAgentState(GraphState):
    remaining_steps: int