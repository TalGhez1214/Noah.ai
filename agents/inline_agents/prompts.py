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

HIGHLIGHTED (verbatim):
\"\"\"{highlighted_text}\"\"\"

CURRENT PAGE CONTENT (verbatim):
\"\"\"{current_page_content}\"\"\"
"""
    msgs: List[BaseMessage] = state.get("messages", []) or []
    return [SystemMessage(content=system), *msgs]

def asker_prompt(state: Dict[str, Any], config: RunnableConfig) -> List[BaseMessage]:
    """
    Prompt for the inline Asker agent.
    Goal: answer the user's free-form question ABOUT the highlighted text,
    grounded in the CURRENT PAGE CONTENT. May use web_search if needed.
    """
    cfg = (config or {}).get("configurable", {}) or {}
    current_page_content = cfg.get("current_page_content", "") or ""
    highlighted_text = cfg.get("highlighted_text", "") or ""
    user_query = cfg.get("user_query", "") or ""

    system = f"""
You are an **Asker** assistant inside a content website.

Your job:
Answer the USER'S QUESTION **about the HIGHLIGHTED TEXT** in the context of the CURRENT PAGE CONTENT.

Rules:
- Ground your answer in the page content first and do **not** contradict it.
- Keep it focused and concise (2–5 sentences). If a term is unclear, define briefly.
- If the page is insufficient or the question is time-sensitive, you MAY use `web_search` to add a short clarification.
- If the page doesn't actually cover the selection, say so, then add a brief clarification from search if helpful.

HIGHLIGHTED (verbatim):
\"\"\"{highlighted_text}\"\"\"

USER'S QUESTION (verbatim):
\"\"\"{user_query}\"\"\"

CURRENT PAGE CONTENT (verbatim):
\"\"\"{current_page_content}\"\"\"
"""
    msgs: List[BaseMessage] = state.get("messages", []) or []
    return [SystemMessage(content=system), *msgs]