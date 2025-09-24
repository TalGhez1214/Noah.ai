from typing import TypedDict, Annotated, Literal
import operator
import re

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from agents.prompts import SUPERVISOR_PROMPT, qa_prompt, SUMMARY_PROMPT, article_finder_prompt, fallback_agent_prompt, HIGHLIGHTER_PROMPT

from agents.sub_agents.qa import QASubAgent
from agents.sub_agents.summarizer import SummarizerSubAgent
from agents.sub_agents.articles_finder import ArticalFinderSubAgent
from agents.sub_agents.fallback import FallbackSubAgent
from agents.sub_agents.highlighter import HighlighterSubAgent


# ---- import search graph module ----
import rag.rag_piplines.articles_finder_graph as M

from typing import Optional
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from agents.manager_agent.GraphStates import GraphState

# This allow us to save memory between different API calls and sessions - 
# This is in-memory only - It does not persist between server restarts, Good for dev/testing, not production
memory = MemorySaver() 



class ManagerAgent:
    def __init__(self, model: str = "gpt-4o-mini", user_query: str = "", user_id: Optional[str] = None, current_page: Optional[dict] = None):
        self.user_query = user_query
        self.user_id = user_id
        self.retriever = M.build_graph()
        self.current_page = current_page

        
        ## Available sub-agents ##
        self.qa_agent = QASubAgent(retriever=self.retriever, model="gpt-4o", prompt=qa_prompt)
        self.article_summary_agent = SummarizerSubAgent(retriever=self.retriever, model=model, prompt=SUMMARY_PROMPT)
        self.articles_finder_agent = ArticalFinderSubAgent(retriever=self.retriever, model=model, prompt=article_finder_prompt)
        self.fallback_agent = FallbackSubAgent(model=model, prompt=fallback_agent_prompt)
        self.highlighter_agent = HighlighterSubAgent(model=model, prompt=HIGHLIGHTER_PROMPT)

        self._agents = [
                        self.qa_agent,
                        self.article_summary_agent, 
                        self.articles_finder_agent, 
                        self.fallback_agent,
                        self.highlighter_agent, 
                        ]

        self._tools = self.create_tools(
            agents=self._agents
        )

        self._supervisor_agent = create_react_agent(
                            model="openai:gpt-4.1",
                            tools=self._tools,
                            prompt=SUPERVISOR_PROMPT,
                            name="supervisor",
                        )

        ## LangGraph build ##
        graph = StateGraph(GraphState)
        graph.add_node(self._supervisor_agent)
        graph.add_node(self.qa_agent.name, self.qa_agent.call)
        graph.add_node(self.article_summary_agent.name, self.article_summary_agent.call)
        graph.add_node(self.articles_finder_agent.name, self.articles_finder_agent.call)
        graph.add_node(self.fallback_agent.name, self.fallback_agent.call)
        graph.add_node(self.highlighter_agent.name, self.highlighter_agent.call)

        graph.add_edge(START, "supervisor")
        graph.add_edge(self.qa_agent.name, END)
        graph.add_edge(self.article_summary_agent.name, END)
        graph.add_edge(self.articles_finder_agent.name, END)
        graph.add_edge(self.fallback_agent.name, END)
        graph.add_edge(self.highlighter_agent.name, END)

        self.app = graph.compile(checkpointer=memory)


    def create_tools(self, agents):
        """Create tools for the supervisor agent."""
        tools = []
        for agent in agents:
            tool = self.create_handoff_tool(agent_name=agent.name, description=agent.description)
            tools.append(tool)
        return tools

    def create_handoff_tool(self, agent_name: str, description: str | None = None):
        name = f"transfer_to_{agent_name}"
        description = description or f"Ask {agent_name} for help."

        @tool(name, description=description)
        def handoff_tool(
            state: Annotated[MessagesState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            tool_message = {
                "role": "tool",
                "content": f"Successfully transferred to {agent_name}",
                "name": name,
                "tool_call_id": tool_call_id,
            }
            return Command(
                goto=agent_name,  
                update={**state, "messages": state["messages"] + [tool_message]},  
                graph=Command.PARENT,  
            )
        return handoff_tool


    # Chat entry point for full conversation
    def chat(self) -> list[BaseMessage]:
        """Chat entry point for full conversation."""
        
        thread = {"configurable": {"thread_id": self.user_id}} # Use user_id as thread_id to pull conversation history from RAM
        new_message = [HumanMessage(content=self.user_query)]

        graph_state = {
            "messages": new_message,
            "user_query": self.user_query,  # Store the user query in the state
            "agent": None,  # Initially no agent is assigned
            "current_page": self.current_page,
        }
        # calling manager agent to start working
        updated = self.app.invoke(graph_state, thread) # Here we added the user query as a new message to the Graph state
        
        # Print all messages in the updated state
        
        # print("\n🗨️ Conversation History:")
        # for m in updated["messages"]:
        #     m.pretty_print()
        print("\n🗨️ Conversation (no tool messages):")
        for m in updated["messages"]:
            if getattr(m, "type", "") not in {"tool"}:  # keep human/ai/system only
                m.pretty_print()
        return updated["messages"]
    
    

    
