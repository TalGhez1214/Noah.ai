from __future__ import annotations
from typing import Any, Dict, List
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent, InjectedState
from typing_extensions import Annotated

# Output parser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Tavily web search tool (ONLY this one)
from langchain_community.tools.tavily_search import TavilySearchResults


class QASubAgent:
    """
    Q&A agent that answers user questions using:
    1) Conversation + current page content (highest priority)
    2) get_data_for_answer_from_database_tool (website content)
    3) Tavily web search (for up-to-date info)
    4) General knowledge (only if still necessary)

    Final output MUST follow the StructuredOutputParser schema:
      {
        "answer": "<short, simple answer>",
        "resource_list": ["<ids from DB tool used>"]
      }
    """

    def __init__(
        self,
        retriever,
        model: str,
        prompt: str,  # system prompt template with {current_page_content}
    ) -> None:
        self.name = "qa_agent"
        self.description = "Q&A agent that answers user questions using retrieved knowledge."
        self.retriever = retriever
        self.prompt_template = prompt

        # ---------- TOOLS ----------
        @tool(
            "get_data_for_answer_from_database_tool",
            description=(
                "Retrieve relevant website content (chunks) to answer the user's question. "
                "Returns a list of dicts with fields: 'id' and 'content'. Use these when the user asks "
                "about the current article/site topics or when conversation/current page isn't enough."
            ),
        )
        def get_data_for_answer_from_database_tool(
            state: Annotated[dict, InjectedState],  # injected automatically by LangGraph
        ) -> List[Dict[str, Any]]:
            initial = {
                "messages": state.get("messages", []),
                "user_query": state.get("user_query", "") or "",
                "file_type": "chunks",
                "requested_k": 10,  # Number of chunks to return
            }
            out: Dict[str, Any] = self.retriever.invoke(initial)
            top_results = out.get("top_results", []) or []
            # Expecting a list[{"id": "...", "content": "..."}]
            return top_results

        # Only Tavily web search
        web_search_tool = TavilySearchResults(
            max_results=5,
            name="web_search",
            description=(
                "Search the web for up-to-date information. Use ONLY if conversation/current page and "
                "database content are insufficient. Return concise results."
            ),
        )

        self._tools = [get_data_for_answer_from_database_tool, web_search_tool]

        # ---------- OUTPUT PARSER ----------
        # Define response schema: answer (str), resource_list (list[str])
        self._response_schemas = [
            ResponseSchema(
                name="answer",
                description="Short, simple answer text for the user. If web search or general knowledge was used, begin by stating that no relevant information was found on the website."
            ),
            ResponseSchema(
                name="resource_list",
                description="A JSON array of strings with the exact 'id' values of DB chunks actually used. If none were used, return an empty array []."
            ),
        ]
        self._output_parser = StructuredOutputParser.from_response_schemas(self._response_schemas)
        self._format_instructions = self._output_parser.get_format_instructions()

        # ---------- LLM + Agent ----------
        # Lower temperature for reliability/grounding; you can tune later.
        self._llm = ChatOpenAI(model=model, temperature=0.2)

        # We'll inject SystemMessage at call-time (so {current_page_content} is fresh).
        self.agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt="",  # we prepend a SystemMessage in call()
            name="qa",
        )

    def _format_system_prompt(self, state: Dict[str, Any]) -> str:
        current_page = state.get("current_page", "") or ""
        # Insert the parser's format instructions into the prompt so the model emits the correct JSON.
        return self.prompt_template.format(
            current_page_content=current_page["content"] if isinstance(current_page, dict) else "",
            format_instructions=self._format_instructions,
        )

    def structured_output(self, llm_output: str) -> Dict[str, Any]:
        """
        Convert the output string from the agent into a structured object per the parser schema.
        Retries with a self-fix prompt up to 3 times.
        """
        for parse_try in range(3):
            try:
                parsed = self._output_parser.parse(llm_output)
                # Normalize types
                answer = parsed.get("answer", "")
                resource_list = parsed.get("resource_list", [])
                if not isinstance(answer, str):
                    answer = str(answer)
                if not isinstance(resource_list, list):
                    resource_list = []
                resource_list = [str(x) for x in resource_list]
                return {"answer": answer, "resource_list": resource_list}
            except Exception as e:
                print(f"⚠️ Parsing attempt {parse_try+1} failed:", e)
                # Retry with format instructions
                fixed_prompt = (
                    "Your previous output was invalid:\n"
                    f"{llm_output}\n\n"
                    f"{self._format_instructions}\n"
                    "Fix this into valid JSON only, nothing else.\n"
                    'If you dont know the answer to one of the fields, just return "Invalid output" for that field.'
                )
                llm_output = self._llm.invoke(fixed_prompt).content.strip()

        # Only raise after all retries failed
        raise ValueError("Failed to parse output after multiple attempts.")

    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected state keys:
          - messages: list of LC messages (user/assistant)
          - current_page_content: str (raw text of current article/page)
          - user_query: str (best-effort copy of user's latest query)
        """
        system_prompt = self._format_system_prompt(state)
        messages = [SystemMessage(system_prompt)] + state.get("messages", [])

        out = self.agent.invoke({"messages": messages})

        # The last assistant message should be JSON per our parser.
        final_msg = out["messages"][-1]
        final_content = getattr(final_msg, "content", "") or ""

        parsed = self.structured_output(final_content)
        # Replace the last message with the coerced JSON text to keep graph state consistent
        out["messages"][-1].content = json.dumps(parsed, ensure_ascii=False)

        return {"messages": out["messages"], "agent": self.name}
