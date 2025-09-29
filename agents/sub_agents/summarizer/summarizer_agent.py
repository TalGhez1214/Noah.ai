from __future__ import annotations
from typing import Any, Dict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..base import BaseSubAgent
from typing_extensions import Annotated
from agents.manager_agent.GraphStates import GraphState
from agents.sub_agents.summarizer.summarizer_tools import (
    summary_content_from_link_tool,
    summary_article_from_current_page_tool,
    get_articles_from_database_tool,
)

# ---------- Structured Output ----------
class RespondFormat(BaseModel):
    # Copy the field VALUES exactly as written in the FINAL assistant message.
    # Do not paraphrase, normalize, spell-correct, or change punctuation/casing/whitespace.

    answer: str = Field(
        ...,
        description=(
            "Verbatim answer from the 'Answer' field in the FINAL assistant message. "
            "Copy EXACTLY as it appears (no rewording or normalization)."
        ),
    )

    title: str = Field(
        ...,
        description=(
            "Verbatim article title from the FINAL assistant message. "
            "Copy EXACTLY as it appears (no rewording or normalization)."
        ),
    )
    summary: str = Field(
        ...,
        description=(
            "Verbatim article summary from the FINAL assistant message. "
            "Copy EXACTLY as it appears; do not add/remove words or adjust formatting."
        ),
    )
    url: str = Field(
        ...,
        description=(
            'Verbatim URL from the FINAL assistant message. '
            'Copy EXACTLY as it appears. If unknown, use an empty string "" (do not invent "N/A").'
        ),
    )
class SummarizerSubAgent(BaseSubAgent):
    def __init__(self, retriever, model: str, prompt: str) -> None:
        self.name = "summary_agent"
        self.description = "This agent is responsible for all article summarizing requests."
        self.retriever = retriever

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder("messages"),
        ])

        self.format_prompt = ChatPromptTemplate.from_messages([
                            (
                                "system",
                                "You are a strict extractor. Copy the Answer, Title, Summary, and URL **verbatim** "
                                "from the LAST ASSISTANT MESSAGE provided below. Do NOT rewrite, paraphrase, "
                                "spell-correct, normalize, or add/remove words. Preserve capitalization and punctuation. "
                                "If a field is missing/ there is a clarification question, set it to an empty string (for URL, empty string is allowed). "
                                "Return exactly the three fields required by the schema."
                            ),
                            (
                                "user",
                                "Last assistant message (the one you must copy from verbatim):\n\n{final_ai_text}"
                            ),
                            ])

        # ---------- TOOLS ----------
        self.tools = [
            summary_content_from_link_tool,
            summary_article_from_current_page_tool,
            get_articles_from_database_tool,
        ]

        # ---------- LLMs ----------
        self.llm = ChatOpenAI(model=model, temperature=0.2)
        self.tool_enabled_llm = self.llm.bind_tools(self.tools)
        self.parser_model = self.llm.with_structured_output(RespondFormat)

        # ---------- Build Graph ----------
        workflow = StateGraph(GraphState)

        # agent node
        def call_model(state: GraphState):
            chain = self.prompt | self.tool_enabled_llm
            response = chain.invoke({"messages": state["messages"]})
            return {"messages": [response]}

        # respond node (final structured answer)
        def respond(state: GraphState):
            """
            Parse the LAST AI message (the model's final text after all tools),
            which you've guided to contain title, summary, url.
            """
            final_ai = state["messages"][-1]           # <- last AIMessage (no tool_calls)
            formatter_chain = self.format_prompt | self.parser_model
            parsed = formatter_chain.invoke({"final_ai_text": final_ai.content})

            # Convert to dict safely
            payload = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)

            # Tag for your UI
            payload["type"] = "summary"

            # Append to modals (avoid in-place append returning None)
            modals = list(state.get("modals", []))
            modals.append(payload)

            return {"modals": modals}

        # routing logic
        def should_continue(state: GraphState):
            last_message = state["messages"][-1]
            if not last_message.tool_calls:
                return "respond" # no tool calls → we're done; route to respond, where we run the structured-output model
            return "continue" # there ARE tool calls → route to the tools node so the ToolNode will execute them

        # build workflow
        workflow.add_node("agent", call_model)
        workflow.add_node("respond", respond)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tools", # Go to executing tools
                "respond": "respond", # Go to structured-output model - finish workflow
            },
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("respond", END)

        self.graph = workflow.compile()

    