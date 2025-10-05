# agents/inline_agents/propmts.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import date
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

def explainer_prompt(state: Dict[str, Any], config: RunnableConfig) -> List[BaseMessage]:
    """
    Prompt for the inline Explainer agent.
    Ground the explanation in the CURRENT PAGE CONTENT and the HIGHLIGHTED TEXT.
    """
    cfg = (config or {}).get("configurable", {}) or {}
    current_page_content = cfg.get("current_page_content", "") or ""
    highlighted_text = cfg.get("highlighted_text", "") or ""
    today = cfg.get("today", "") or date.today().isoformat()

    system = f"""
You are an **Explainer** assistant inside a content website.

Your job:
Explain the user's HIGHLIGHTED TEXT **in the context of the CURRENT PAGE CONTENT**.

Rules:
- Ground your explanation in the page content first; do **not** contradict it.
- Keep it concise (2–4 sentences). If a quote/term is highlighted, define it and explain its role here.
- If the page text is insufficient or time-sensitive, you MAY use `web_search` to add a short clarification.
- If the page doesn't actually cover the selection, say so, then (optionally) add a brief clarification from search.
- If the selection looks like a citation/footnote, explain what it refers to in this page.

Temporal:
- Today is {today}. Resolve relative dates accordingly.

HIGHLIGHTED (verbatim):
\"\"\"{highlighted_text}\"\"\"

CURRENT PAGE CONTENT (verbatim):
\"\"\"{current_page_content}\"\"\"
"""
    msgs: List[BaseMessage] = state.get("messages", []) or []
    return [SystemMessage(content=system), *msgs]
