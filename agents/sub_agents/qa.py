from __future__ import annotations
from typing import Any, Dict, List
from datetime import date
import json

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent, InjectedState
from typing_extensions import Annotated

from langchain_tavily import TavilySearch

# ---------- Parser schema (Pydantic) ----------
from pydantic import BaseModel, Field
from typing import List as _List


class ResourcePickerSchema(BaseModel):
    """
    Structured output for the parser agent.
    Select up to 3 relevant DB chunk IDs that best support the user's question and the draft answer.
    """
    resource_list: _List[str] = Field(
        default_factory=list,
        description="Up to 3 exact chunk IDs from get_data_for_answer_from_database_tool that truly support the answer. [] if none."
    )


class QASubAgent:
    """
    Simple Q&A agent + end-of-run parser:

    Flow:
      1) ReAct QA agent (can call DB tool and/or web_search) → produces a draft answer (free-form text).
      2) Parser agent (with_structured_output(ResourcePickerSchema)) → reads the draft + tool trace and returns
         up to 3 relevant DB chunk IDs (or []).
      3) Final output:
         - messages[-1].content = the original draft answer (unchanged),
         - relevant_articles_for_user = selected IDs.
    """

    def __init__(self, retriever, model: str, prompt) -> None:
        self.name = "qa_agent"
        self.description = "This agent is responsible for all the question answering requests."
        self.retriever = retriever
        self.prompt = prompt
        self.model = model

        # ----- Tools -----
        @tool(
            "get_data_for_answer_from_database_tool",
            description=(
                "Retrieve relevant website content (chunks) to answer the user's question. "
                "Returns a list of dicts with fields: 'id' and 'content'."
            ),
        )
        def get_data_for_answer_from_database_tool(
            state: Annotated[dict, InjectedState],
        ) -> List[Dict[str, Any]]:
            initial = {
                "messages": state.get("messages", []),
                "user_query": state.get("user_query", "") or "",
                "file_type": "chunks",
                "requested_k": 10,
            }
            out: Dict[str, Any] = self.retriever.invoke(initial)
            return out.get("top_results", []) or []

        web_search_tool = TavilySearch(
            max_results=5,
            name="web_search",
            description="Search the web for up-to-date information.",
        )

        self._tools = [get_data_for_answer_from_database_tool, web_search_tool]

        # ----- Base LLM + ReAct Agent -----
        self._llm = ChatOpenAI(model=self.model, temperature=0.2)
        self.agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=self.prompt,
            name="qa",
        )

        # ----- Parser Agent (strict schema; no tool calls) -----
        # IMPORTANT: This finisher enforces schema and MUST NOT change the answer text.
        self._parser_llm = ChatOpenAI(model=self.model, temperature=0).with_structured_output(ResourcePickerSchema)

    # ---------- Helpers ----------

    def _collect_db_tool_results(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """
        Gather the raw dict items returned by get_data_for_answer_from_database_tool across the run.
        Each ToolMessage content may be:
          - a Python list[dict], or
          - a JSON string representing list[dict]
        We return a flat list of dicts: [{"id": "...", "content": "..."}, ...]
        """
        results: List[Dict[str, Any]] = []
        for m in messages:
            if isinstance(m, ToolMessage) and getattr(m, "name", "") == "get_data_for_answer_from_database_tool":
                payload = m.content
                try:
                    data = json.loads(payload) if isinstance(payload, str) else payload
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "article_id" in item and "snippet" in item:
                                results.append({"id": str(item["article_id"]), "content": str(item["snippet"])})
                except Exception:
                    # Ignore malformed tool payloads gracefully
                    continue
        return results

    def _run_resource_picker(
        self,
        user_query: str,
        draft_answer_text: str,
        db_chunks: List[Dict[str, Any]],
        page_content: str,
    ) -> ResourcePickerSchema:
        """
        Parser agent: MUST NOT change the draft answer.
        It only selects up to 3 relevant IDs from db_chunks (or []) using the schema.
        """
        system = SystemMessage(content=(
            "You are a finisher that selects up to 3 website chunk IDs supporting the draft answer. "
            "Rules:\n"
            "- DO NOT change or rewrite the draft answer text.\n"
            "- Choose only IDs that are clearly relevant to the user's question and the draft answer; otherwise return [].\n"
            "- Never invent IDs. Only use IDs present in db_chunks.\n"
            "- Return resource_list with at most 3 string IDs.\n"
        ))
        # Provide only what's needed. db_chunks includes the full dicts {"id","content"}.
        human_payload = {
            "user_query": user_query,
            "draft_answer_text": draft_answer_text,
            "db_chunks": db_chunks,              # list of {"id": str, "content": str}
            "current_page_excerpt": page_content[:2000],
        }
        human = HumanMessage(content=json.dumps(human_payload, ensure_ascii=False))

        result: ResourcePickerSchema = self._parser_llm.invoke([system, human])
        # Normalize output just in case
        ids = result.resource_list or []
        if not isinstance(ids, list):
            ids = []
        ids = [str(x) for x in ids][:3]
        return ResourcePickerSchema(resource_list=ids)

    # ---------- Public API ----------

    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        page_content = (state.get("current_page") or {}).get("content", "") or ""
        messages = state.get("messages", []) or []
        user_query = state.get("user_query", "") or ""

        # 1) Run the main ReAct QA agent (free to call tools as needed)
        agent_answer = self.agent.invoke(
            {"messages": messages},
            config={
                "configurable": {
                    "current_page_content": page_content,
                    "today": date.today().isoformat(),
                    "user_query": user_query,
                }
            },
        )

        # 2) Get the draft assistant output (leave it EXACTLY as is)
        final_msg = agent_answer["messages"][-1]
        draft_answer_text = getattr(final_msg, "content", "") or ""

        # 3) Gather all DB tool results as list[dict] with "id" and "content"
        db_chunks = self._collect_db_tool_results(agent_answer["messages"])

        # 4) Parser: pick up to 3 relevant IDs (answer text is NOT changed)
        picker = self._run_resource_picker(
            user_query=user_query,
            draft_answer_text=draft_answer_text,
            db_chunks=db_chunks,
            page_content=page_content,
        )

        # 5) Replace the last assistant message with the original draft answer (unchanged)
        agent_answer["messages"][-1].content = draft_answer_text

        # 6) Return final state
        return {
            "messages": agent_answer["messages"],
            "agent": self.name,
            "relevant_articles_for_user": list(picker.resource_list or []),
        }
