# # agents/sub_agents/highlighter.py
# from __future__ import annotations
# from typing import Any, Dict, List, Optional
# from dataclasses import dataclass

# from langchain_openai import ChatOpenAI
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from langgraph.prebuilt import create_react_agent

# from .base import BaseSubAgent  # your base class


# @dataclass
# class HighlightSpan:
#     start: int
#     end: int
#     sentence: str
#     reason: str
#     score: float


# _SYSTEM_PROMPT = """\
# You are a highlighting assistant. Your job is to read an ARTICLE CONTENT and a USER QUERY
# and return the most relevant passages as exact character spans within the article.

# OUTPUT FORMAT (mandatory):
# Return a single fenced JSON block with an array of objects, each:
# {
#   "start": <0-based character offset in ORIGINAL CONTENT>,
#   "end": <exclusive offset>,
#   "sentence": "<the exact substring selected>",
#   "reason": "<'keyword-match'|'importance'|'lede' or a short reason>",
#   "score": <0.0-1.0>
# }

# STRICT RULES:
# - Offsets MUST match the ORIGINAL CONTENT EXACTLY (0-based chars; end is exclusive).
# - "sentence" MUST be exactly the substring between start..end from the ORIGINAL CONTENT.
# - Choose at most 6 highlights. Prefer non-overlapping spans.
# - If the user asks for “most important phrase(s)”, pick concise, high-signal lines.
# - If the user asks “where it mentions <X>”, return passages mentioning <X> (or nearest definitional sentences).
# - If nothing relevant exists, return an empty JSON array: []

# Do not add commentary outside the JSON block.
# """

# # We will format a user message like:
# # USER QUERY:
# # <query>
# #
# # ARTICLE CONTENT (do not alter, count offsets against THIS TEXT):
# # <content>
# _USER_MSG_TEMPLATE = """\
# USER QUERY:
# {query}

# ARTICLE CONTENT (do NOT alter, and compute offsets against THIS TEXT):
# {content}
# """


# class HighlighterSubAgent(BaseSubAgent):
#     """
#     ReAct sub-agent (no tools) that asks the LLM to select highlight spans from the article
#     based on the user's query, and emits a JSON array with start/end indices.
#     """

#     def __init__(self, model: str = "gpt-4o-mini") -> None:
#         self.name = "highlighter_agent"
#         self.description = "Highlight the most relevant passages in the current article for the user's query."

#         llm = ChatOpenAI(model=model, temperature=0)
#         # ReAct agent with no tools — we still use the prebuilt structure for parity with your other agents.
#         self.agent = create_react_agent(
#             model=llm,
#             tools=[],                 # <- no tools; LLM decides spans itself
#             prompt=_SYSTEM_PROMPT,    # acts as system; we’ll pass the content as a user message
#             name="highlighter",
#         )

#     def get_knowledge_for_answer(self, query: str) -> str:
#         # not used for this sub-agent
#         return ""

#     def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Expects in state:
#           - 'user_query': str
#           - 'current_page': dict with 'content': str
#         Returns:
#           - {"messages": [AIMessage], "agent": self.name}
#         """
#         user_query: str = state.get("user_query") or ""
#         page: Optional[Dict[str, Any]] = state.get("current_page") or {}
#         content: str = page.get("content") or ""

#         # Build a one-shot interaction: system + user
#         messages = [
#             SystemMessage(content=_SYSTEM_PROMPT),
#             HumanMessage(content=_USER_MSG_TEMPLATE.format(query=user_query, content=content)),
#         ]

#         # Even though create_react_agent builds a LangGraph under the hood, with no tools it just does a single LLM turn.
#         out = self.agent.invoke({"messages": messages})

#         # Ensure we always return an AIMessage for the manager graph
#         msgs = out.get("messages") or []
#         if not msgs or not isinstance(msgs[-1], AIMessage):
#             # Fallback: synthesize an AIMessage with an empty JSON array
#             msgs = [AIMessage(content="```\n[]\n```")]

#         # OPTIONAL: You may parse & validate the JSON here if you want to enforce shape.
#         return {"messages": msgs, "agent": self.name}


# agents/sub_agents/highlighter.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from .base import BaseSubAgent  # your base class


@dataclass
class HighlightSpan:
    start: int
    end: int
    sentence: str
    reason: str
    score: float


_SYSTEM_PROMPT = """\
You are a highlighting assistant. Your job is to read an ARTICLE CONTENT and a USER QUERY
and return the most relevant passages as exact character spans within the article.

OUTPUT FORMAT (mandatory):
Return a single fenced JSON block with an array of objects, each:
{
  "start": <0-based character offset in ORIGINAL CONTENT>,
  "end": <exclusive offset>,
  "sentence": "<the exact substring selected>",
  "reason": "<'keyword-match'|'importance'|'lede' or a short reason>",
  "score": <0.0-1.0>
}

STRICT RULES:
- Offsets MUST match the ORIGINAL CONTENT EXACTLY (0-based chars; end is exclusive).
- "sentence" MUST be exactly the substring between start..end from the ORIGINAL CONTENT.
- Choose at most 6 highlights. Prefer non-overlapping spans.
- If the user asks for “most important phrase(s)”, pick concise, high-signal lines.
- If the user asks “where it mentions <X>”, return passages mentioning <X> (or nearest definitional sentences).
- If nothing relevant exists, return an empty JSON array: []

Do not add commentary outside the JSON block.
"""

_USER_MSG_TEMPLATE = """\
USER QUERY:
{query}

ARTICLE CONTENT (do NOT alter, and compute offsets against THIS TEXT):
{content}
"""


def _extract_json_block(text: str) -> Optional[str]:
    # Prefer fenced ```json ... ```; fallback to first [...] array
    import re
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


# def _normalize_spans(content: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     Snap LLM-proposed spans to the real content:
#       - clamp indices
#       - if 'sentence' exists, search near the suggested start (±300 chars), then globally, case-sensitive then -insensitive
#       - set end = start + len(sentence) when found
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
#         # exact in window
#         idx = local.find(sentence)
#         if idx != -1:
#             return a + idx
#         # case-insensitive in window
#         idx = local.lower().find(sentence.lower())
#         if idx != -1:
#             return a + idx
#         # exact global
#         idx = text.find(sentence)
#         if idx != -1:
#             return idx
#         # ci global
#         idx = text.lower().find(sentence.lower())
#         return idx

#     fixed: List[Dict[str, Any]] = []
#     for h in spans:
#         sentence = str(h.get("sentence") or "")
#         start_hint = int(h.get("start") or 0)
#         end_hint = int(h.get("end") or 0)
#         reason = h.get("reason") or "keyword-match"
#         score = float(h.get("score") or 0.0)

#         # compute start
#         start = -1
#         if sentence:
#             start = find_near(sentence, start_hint)
#         if start < 0:
#             start = clamp(start_hint, 0, L)

#         # compute end
#         if sentence and start >= 0:
#             end = start + len(sentence)
#         else:
#             end = clamp(end_hint, start, L)

#         start = clamp(start, 0, L)
#         end = clamp(end, start, L)
#         if end <= start:
#             continue

#         fixed.append({
#             "start": start,
#             "end": end,
#             "sentence": text[start:end],
#             "reason": reason,
#             "score": score,
#         })

#     # sort and de-dup close duplicates
#     fixed.sort(key=lambda x: (x["start"], x["end"]))
#     dedup: List[Dict[str, Any]] = []
#     for h in fixed:
#         if not dedup:
#             dedup.append(h)
#             continue
#         last = dedup[-1]
#         if abs(last["start"] - h["start"]) < 2 and abs(last["end"] - h["end"]) < 2:
#             # keep the higher score
#             if h.get("score", 0.0) > last.get("score", 0.0):
#                 dedup[-1] = h
#         else:
#             dedup.append(h)
#     return dedup[:6]

def _normalize_spans(content: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Snap LLM-proposed spans to the real content, then expand to full sentence boundaries:
      - clamp indices
      - if 'sentence' exists, search near the suggested start (±300 chars), then globally, case-sensitive then -insensitive
      - set (start,end) to the FULL sentence that contains the match (no mid-sentence highlights)
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

    # --- NEW: expand a (start,end) to cover the full sentence ---
    import re
    # sentence boundary approximation: ., !, ? followed by space/newline or end-of-text
    # we’ll also treat start-of-text and end-of-text as boundaries.
    boundary_re = re.compile(r"[.!?]")

    def expand_to_sentence(start_idx: int, end_idx: int) -> (int, int):
        # Move left to previous boundary (or start)
        left = start_idx
        # search for last punctuation before start_idx
        prev = -1
        for m in boundary_re.finditer(text, 0, start_idx):
            prev = m.end()  # boundary is right after punctuation
        if prev != -1:
            left = prev
        else:
            left = 0
        # skip leading spaces/newlines
        while left < L and text[left] in " \t\r\n":
            left += 1

        # Move right to next boundary (or end)
        right = end_idx
        m = boundary_re.search(text, end_idx)
        if m:
            right = m.end()
        else:
            right = L
        # include trailing spaces immediately after punctuation (keeps words intact)
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

        # compute start by locating the sentence text near hint; fallback to hint
        start = -1
        if sentence:
            start = find_near(sentence, start_hint)
        if start < 0:
            start = clamp(start_hint, 0, L)

        # provisional end
        if sentence and start >= 0:
            end = start + len(sentence)
        else:
            end = clamp(end_hint, start, L)

        start = clamp(start, 0, L)
        end = clamp(end, start, L)
        if end <= start:
            continue

        # --- NEW: expand to full sentence bounds ---
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

    # sort and de-dup close duplicates
    fixed.sort(key=lambda x: (x["start"], x["end"]))
    dedup: List[Dict[str, Any]] = []
    for h in fixed:
        if not dedup:
            dedup.append(h)
            continue
        last = dedup[-1]
        # if almost identical, keep the higher score
        if abs(last["start"] - h["start"]) < 2 and abs(last["end"] - h["end"]) < 2:
            if h.get("score", 0.0) > last.get("score", 0.0):
                dedup[-1] = h
        else:
            dedup.append(h)
    return dedup[:6]




class HighlighterSubAgent(BaseSubAgent):
    """
    ReAct sub-agent (no tools) that asks the LLM to select highlight spans from the article
    and then fixes offsets to match the exact content before returning them.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.name = "highlighter_agent"
        self.description = "Highlight the most relevant passages in the current article for the user's query."

        llm = ChatOpenAI(model=model, temperature=0)
        self.agent = create_react_agent(
            model=llm,
            tools=[],              # LLM decides; no tools
            prompt=_SYSTEM_PROMPT, # system prompt
            name="highlighter",
        )

    def get_knowledge_for_answer(self, query: str) -> str:
        return ""

    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects:
          - state['user_query']: str
          - state['current_page']: {'content': str, ...}
        Returns (for manager graph):
          - {"messages": [AIMessage], "agent": self.name}
        """
        user_query: str = state.get("user_query") or ""
        page: Optional[Dict[str, Any]] = state.get("current_page") or {}
        content: str = page.get("content") or ""

        # Prepare a one-shot interaction
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_USER_MSG_TEMPLATE.format(query=user_query, content=content)),
        ]

        # Run LLM
        out = self.agent.invoke({"messages": messages})
        msgs = out.get("messages") or []
        last_text = msgs[-1].content if msgs and isinstance(msgs[-1], AIMessage) else ""

        # Parse + normalize JSON spans against the REAL content
        raw_json = _extract_json_block(last_text)
        spans = _parse_spans(raw_json)
        normalized = _normalize_spans(content, spans)

        # Emit ONLY a fenced JSON block so the UI parser remains simple & robust
        response_content = "```json\n" + json.dumps(normalized, ensure_ascii=False) + "\n```"
        return {"messages": [AIMessage(content=response_content)], "agent": self.name}
