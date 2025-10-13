# # agents/sub_agents/highlighter.py
# from __future__ import annotations
# from typing import Any, Dict, List, Optional
# from dataclasses import dataclass
# import json
# import re

# from langchain_openai import ChatOpenAI
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from langgraph.prebuilt import create_react_agent

# from .base import BaseSubAgent
# from agents.prompts import HIGHLIGHTER_PROMPT


# @dataclass
# class HighlightSpan:
#     start: int
#     end: int
#     sentence: str
#     reason: str
#     score: float


# _USER_MSG_TEMPLATE = """\
# USER QUERY:
# {query}

# ARTICLE CONTENT (do NOT alter, and compute offsets against THIS TEXT):
# {content}
# """


# def _extract_json_block(text: str) -> Optional[str]:
#     # Prefer fenced ```json ... ```; fallback to first [...] array
#     m = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
#     if m:
#         return m.group(1).strip()
#     m = re.search(r"\[[\s\S]*\]", text)
#     if m:
#         return m.group(0).strip()
#     return None


# def _parse_spans(raw: Optional[str]) -> List[Dict[str, Any]]:
#     if not raw:
#         return []
#     try:
#         data = json.loads(raw)
#         return data if isinstance(data, list) else []
#     except Exception:
#         return []


# def _normalize_spans(content: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     Snap LLM-proposed spans to the real content, then expand to full sentence boundaries:
#       - clamp indices
#       - try to locate 'sentence' near start hint, then globally (case-sensitive then insensitive)
#       - expand to full sentence (no mid-sentence highlights)
#       - de-duplicate and sort
#     """
#     if not spans:
#         return []
#     text = content or ""
#     L = len(text)

#     def clamp(n: int, lo: int, hi: int) -> int:
#         return max(lo, min(hi, n))

#     def find_near(sentence: str, start_hint: int, window: int = 300) -> int:
#         if not sentence:
#             return -1
#         a = clamp(start_hint - window, 0, L)
#         b = clamp(start_hint + window, 0, L)
#         local = text[a:b]
#         idx = local.find(sentence)
#         if idx != -1:
#             return a + idx
#         idx = local.lower().find(sentence.lower())
#         if idx != -1:
#             return a + idx
#         idx = text.find(sentence)
#         if idx != -1:
#             return idx
#         idx = text.lower().find(sentence.lower())
#         return idx

#     boundary_re = re.compile(r"[.!?]")

#     def expand_to_sentence(start_idx: int, end_idx: int) -> (int, int):
#         # left boundary
#         prev = -1
#         for m in boundary_re.finditer(text, 0, start_idx):
#             prev = m.end()
#         left = prev if prev != -1 else 0
#         while left < L and text[left] in " \t\r\n":
#             left += 1
#         # right boundary
#         right = L
#         m = boundary_re.search(text, end_idx)
#         if m:
#             right = m.end()
#         while right < L and text[right] in " \t":
#             right += 1
#         left = clamp(left, 0, L)
#         right = clamp(right, left, L)
#         return left, right

#     fixed: List[Dict[str, Any]] = []
#     for h in spans:
#         sentence = str(h.get("sentence") or "")
#         start_hint = int(h.get("start") or 0)
#         end_hint = int(h.get("end") or 0)
#         reason = h.get("reason") or "keyword-match"
#         score = float(h.get("score") or 0.0)

#         start = -1
#         if sentence:
#             start = find_near(sentence, start_hint)
#         if start < 0:
#             start = clamp(start_hint, 0, L)

#         end = start + len(sentence) if sentence and start >= 0 else clamp(end_hint, start, L)
#         start = clamp(start, 0, L)
#         end = clamp(end, start, L)
#         if end <= start:
#             continue

#         s_full, e_full = expand_to_sentence(start, end)
#         if e_full <= s_full:
#             continue

#         fixed.append({
#             "start": s_full,
#             "end": e_full,
#             "sentence": text[s_full:e_full],
#             "reason": reason,
#             "score": score,
#         })

#     fixed.sort(key=lambda x: (x["start"], x["end"]))
#     dedup: List[Dict[str, Any]] = []
#     for h in fixed:
#         if not dedup:
#             dedup.append(h)
#             continue
#         last = dedup[-1]
#         if abs(last["start"] - h["start"]) < 2 and abs(last["end"] - h["end"]) < 2:
#             if h.get("score", 0.0) > last.get("score", 0.0):
#                 dedup[-1] = h
#         else:
#             dedup.append(h)
#     return dedup[:6]


# class HighlighterSubAgent(BaseSubAgent):
#     """
#     ReAct sub-agent (no tools) that asks the LLM to select highlight spans from the article,
#     normalizes them to sentence boundaries, then returns them via ui_items so the UI can apply.
#     """

#     def __init__(self, model: str = "gpt-4o-mini", prompt: str = HIGHLIGHTER_PROMPT) -> None:
#         self.name = "highlighter_agent"
#         self.description = "Highlight the most relevant passages in the current article for the user's query."
#         self.prompt = prompt

#         llm = ChatOpenAI(model=model, temperature=0)
#         self.agent = create_react_agent(
#             model=llm,
#             tools=[],
#             prompt=self.prompt,
#             name="highlighter",
#         )

#     def get_knowledge_for_answer(self, query: str) -> str:
#         return ""

#     def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
#         user_query: str = state.get("user_query") or ""
#         page: Optional[Dict[str, Any]] = state.get("current_page") or {}
#         content: str = page.get("content") or ""

#         messages = [
#             SystemMessage(content=self.prompt),
#             HumanMessage(content=_USER_MSG_TEMPLATE.format(query=user_query, content=content)),
#         ]

#         out = self.agent.invoke({"messages": messages})
#         msgs = out.get("messages") or []
#         last_text = msgs[-1].content if msgs and isinstance(msgs[-1], AIMessage) else ""

#         raw_json = _extract_json_block(last_text)
#         spans = _parse_spans(raw_json)
#         normalized = _normalize_spans(content, spans)

#         # 👇 Return highlights via ui_items (just like summarizer returns summary ui_items)
#         return {
#             "ui_items": {"type": "highlights", "data": normalized},
#             "agent": self.name,
#         }

# agents/sub_agents/highlighter.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
import re

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from .base import BaseSubAgent
from agents.prompts import HIGHLIGHTER_PROMPT
from agents.manager_agent.agents_models import AGENTS_MODELS


@dataclass
class HighlightSpan:
    start: int
    end: int
    sentence: str
    reason: str
    score: float


_USER_MSG_TEMPLATE = """\
USER QUERY:
{query}

ARTICLE CONTENT (do NOT alter, and compute offsets against THIS TEXT):
{content}
"""


def _extract_json_block(text: str) -> Optional[str]:
    # Prefer fenced ```json ... ```; fallback to first [...] array
    m = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        return m.group(0).strip()
    return None


def _parse_spans(raw: Optional[str]) -> List[Dict[str, Any]]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _normalize_spans(content: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Snap LLM-proposed spans to the real content, then expand to full sentence boundaries:
      - clamp indices
      - try to locate 'sentence' near start hint, then globally (case-sensitive then insensitive)
      - expand to full sentence (no mid-sentence highlights)
      - de-duplicate and sort
    """
    if not spans:
        return []
    text = content or ""
    L = len(text)

    def clamp(n: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, n))

    def find_near(sentence: str, start_hint: int, window: int = 300) -> int:
        if not sentence:
            return -1
        a = clamp(start_hint - window, 0, L)
        b = clamp(start_hint + window, 0, L)
        local = text[a:b]
        idx = local.find(sentence)
        if idx != -1:
            return a + idx
        idx = local.lower().find(sentence.lower())
        if idx != -1:
            return a + idx
        idx = text.find(sentence)
        if idx != -1:
            return idx
        idx = text.lower().find(sentence.lower())
        return idx

    boundary_re = re.compile(r"[.!?]")

    def expand_to_sentence(start_idx: int, end_idx: int) -> (int, int):
        # left boundary
        prev = -1
        for m in boundary_re.finditer(text, 0, start_idx):
            prev = m.end()
        left = prev if prev != -1 else 0
        while left < L and text[left] in " \t\r\n":
            left += 1
        # right boundary
        right = L
        m = boundary_re.search(text, end_idx)
        if m:
            right = m.end()
        while right < L and text[right] in " \t":
            right += 1
        left = clamp(left, 0, L)
        right = clamp(right, left, L)
        return left, right

    fixed: List[Dict[str, Any]] = []
    for h in spans:
        sentence = str(h.get("sentence") or "")
        start_hint = int(h.get("start") or 0)
        end_hint = int(h.get("end") or 0)
        reason = h.get("reason") or "keyword-match"
        score = float(h.get("score") or 0.0)

        start = -1
        if sentence:
            start = find_near(sentence, start_hint)
        if start < 0:
            start = clamp(start_hint, 0, L)

        end = start + len(sentence) if sentence and start >= 0 else clamp(end_hint, start, L)
        start = clamp(start, 0, L)
        end = clamp(end, start, L)
        if end <= start:
            continue

        s_full, e_full = expand_to_sentence(start, end)
        if e_full <= s_full:
            continue

        fixed.append({
            "start": s_full,
            "end": e_full,
            "sentence": text[s_full:e_full],
            "reason": reason,
            "score": score,
        })

    fixed.sort(key=lambda x: (x["start"], x["end"]))
    dedup: List[Dict[str, Any]] = []
    for h in fixed:
        if not dedup:
            dedup.append(h)
            continue
        last = dedup[-1]
        if abs(last["start"] - h["start"]) < 2 and abs(last["end"] - h["end"]) < 2:
            if h.get("score", 0.0) > last.get("score", 0.0):
                dedup[-1] = h
        else:
            dedup.append(h)
    return dedup[:6]


class HighlighterSubAgent(BaseSubAgent):
    """
    ReAct sub-agent (no tools) that asks the LLM to select highlight spans from the article,
    normalizes them to sentence boundaries, then returns them via ui_items so the UI can apply.
    """

    def __init__(self, model: Optional[str] = None, prompt: str = HIGHLIGHTER_PROMPT) -> None:
        self.name = "highlighter_agent"
        self.description = "Highlight the most relevant passages in the current article for the user's query."
        self.prompt = prompt

        # Use Groq model from centralized config (fallback provided)
        model_id = model or AGENTS_MODELS.get("highlighter", "openai/gpt-oss-20b")
        llm = ChatGroq(model=model_id, temperature=0)

        self.agent = create_react_agent(
            model=llm,
            tools=[],
            prompt=self.prompt,
            name="highlighter",
        )

    def get_knowledge_for_answer(self, query: str) -> str:
        return ""

    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_query: str = state.get("user_query") or ""
        page: Optional[Dict[str, Any]] = state.get("current_page") or {}
        content: str = page.get("content") or ""

        messages = [
            SystemMessage(content=self.prompt),
            HumanMessage(content=_USER_MSG_TEMPLATE.format(query=user_query, content=content)),
        ]

        out = self.agent.invoke({"messages": messages})
        msgs = out.get("messages") or []
        last_text = msgs[-1].content if msgs and isinstance(msgs[-1], AIMessage) else ""

        raw_json = _extract_json_block(last_text)
        spans = _parse_spans(raw_json)
        normalized = _normalize_spans(content, spans)

        # 👇 Return highlights via ui_items (just like summarizer returns summary ui_items)
        return {
            "ui_items": {"type": "highlights", "data": normalized},
            "agent": self.name,
        }
