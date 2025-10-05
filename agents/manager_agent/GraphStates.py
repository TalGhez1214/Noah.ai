# agents/manager_agent/GraphStates.py

from typing import TypedDict, Optional, Sequence, Any, Annotated
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages  # default reducer

# ---- sliding window settings ----
LAST_TURNS = 10                 # Human+AI turns
LAST_MSGS = LAST_TURNS * 2

def _coherent_tail(msgs: Sequence[BaseMessage], limit: int) -> list[BaseMessage]:
    """
    Take the last `limit` messages and ensure OpenAI's tool-call invariant:
    - If a ToolMessage is kept, its corresponding AI tool_call must also be present.
    Otherwise, drop the ToolMessage.
    """
    tail = list(msgs)[-limit:]

    # Collect tool_call ids that exist in the kept window
    ai_tool_ids: set[str] = set()
    for m in tail:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                # tc can be a dict or a pydantic-like obj
                if isinstance(tc, dict):
                    tid = tc.get("id")
                else:
                    tid = getattr(tc, "id", None)
                if tid:
                    ai_tool_ids.add(tid)

    # Keep only ToolMessages that match a tool_call in the window
    filtered: list[BaseMessage] = []
    for m in tail:
        if isinstance(m, ToolMessage):
            if getattr(m, "tool_call_id", None) in ai_tool_ids:
                filtered.append(m)
            else:
                # Drop orphan tool output (its AI tool_call was trimmed out)
                continue
        else:
            filtered.append(m)

    return filtered

def capped_add_messages(existing: Sequence[BaseMessage],
                        new: Sequence[BaseMessage]) -> list[BaseMessage]:
    merged = list(add_messages(existing, new))
    # 1) naive cap
    if len(merged) <= LAST_MSGS:
        # Still run coherence filter in case earlier code appended bad order
        return _coherent_tail(merged, len(merged))
    # 2) coherent tail (prevents orphan ToolMessages)
    return _coherent_tail(merged, LAST_MSGS)

# ---------- Custom Graph State ----------
class UIItems(TypedDict, total=False):
    type: str            # e.g., "summary", "article", "highlight"
    data: Any            # render-ready payload for your UI

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], capped_add_messages]
    user_query: Optional[str]
    current_page: Optional[dict]
    agent: Optional[str]
    ui_items: Optional[UIItems]
