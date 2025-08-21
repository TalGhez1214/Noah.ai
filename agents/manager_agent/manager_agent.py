from typing import TypedDict, Annotated, Literal
import operator
import re

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from agents.prompts import SUPERVISOR_PROMPT
from agents.sub_agents.qa import build_qa_agent
from agents.sub_agents.summarizer import build_summary_agent
from agents.sub_agents.articles_finder_agent import ArticalFinderAgent
from rag.rag_piplines.rag_retriever import RAGRetriever
from typing import Optional
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

# This allow us to save memory between different API calls and sessions - 
# This is in-memory only - It does not persist between server restarts, Good for dev/testing, not production
memory = MemorySaver() 

class AgentState(MessagesState):
    agent: Optional[str] = None


class ManagerAgent:
    def __init__(self, model: str = "gpt-4o-mini", user_query: str = "", user_id: Optional[str] = None):
        self.user_query = user_query
        self.user_id = user_id

        self.retriever = RAGRetriever()
        self.qa_agent = build_qa_agent(self.retriever, model)
        self.article_summary_agent = build_summary_agent(self.retriever, model)
        self.articles_finder_agent = ArticalFinderAgent(retriever=self.retriever, model=model)


        self._tools = self.create_tools(
            agents=[
                self.qa_agent,
                self.article_summary_agent,
                self.articles_finder_agent,
            ]
        )

        self._supervisor_agent = create_react_agent(
                            model="openai:gpt-4.1",
                            tools=self._tools,
                            prompt=SUPERVISOR_PROMPT,
                            name="supervisor",
                        )

        # LangGraph build
        graph = StateGraph(AgentState)
        graph.add_node(self._supervisor_agent)
        graph.add_node(self.qa_agent.name, self._qa_node)
        graph.add_node(self.article_summary_agent.name, self._summary_node)
        graph.add_node(self.articles_finder_agent.name, self._articles_finder_node)
        graph.add_node("reject", self._reject_node)

        graph.add_edge(START, "supervisor")
        graph.add_edge("qa", END)
        graph.add_edge("summary", END)
        graph.add_edge("reject", END)

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


    # StateGraph nodes for the ManagerAgent
    def _qa_node(self, state: AgentState) -> AgentState:

        out = self.qa_app.invoke({"messages": state["messages"]})
        return {"messages": out["messages"], "agent": "qa"}

    def _summary_node(self, state: AgentState) -> AgentState:
        out = self.summary_app.invoke({"messages": state["messages"]})
        return {"messages": out["messages"], "agent": "summary"}
    
    def _articles_finder_node(self, state: AgentState) -> AgentState:
        """
        This node handles the articles finder agent, which retrieves relevant articles
        based on the user's query and returns them as a dictionary response for each article:
        {
            "title": "Article Title",
            "Summary": "Article Summary",
            "Key Quote": "Key Quote from the article",
        }
        """
        articles = self.articles_finder_agent.get_knowledge_for_answer(user_query=self.user_query)
        articles_snippets = []

        for article in articles:
            agent_answer = self.articles_finder_agent.build_articles_finder_agent(user_query=self.user_query, article=article).invoke({"messages": state["messages"]})
            try:
                json_output = self.articles_finder_agent.structured_output(agent_answer["messages"][-1].content)
            except Exception as e:
                print(f"Error parsing structured output: {e}")
                json_output = {"Summary": "No summary available", "Key Quote": "No quote available"}
            articles_snippets.append(json_output)

        for i in range(len(articles_snippets)):
            articles_snippets[i]["title"] = articles[i]["title"]
            articles_snippets[i]["url"] = articles[i]["url"] # Make sure it's clickable in the UI

        return {
            "messages": [AIMessage(content=f"{articles_snippets}")], 
            "agent": "articles_finder"
        }
    

    def _reject_node(self, state: AgentState) -> AgentState:
        """Politely decline unsupported tasks using an LLM-generated message."""
        last_user_msg = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "")
        
        rejection_prompt = (
            "You are a helpful assistant specialized in news Q&A and summarization.\n"
            "A user has just asked a question or made a request that is outside your capabilities.\n"
            "You cannot help with this request, but you want to be kind and clear.\n\n"
            "Explain that you are limited to:\n"
            "- Answering news-related questions\n"
            "- Summarizing articles or topics\n"
            "Then give 1–2 example prompts the user *can* ask instead.\n\n"
            "User said:\n"
            f"{last_user_msg}\n\n"
            "Respond kindly and clearly:"
        )

        reply = self.router_llm.invoke([SystemMessage(content=rejection_prompt)]).content.strip()
        return {"messages": [AIMessage(content=reply)], "agent": "none"}

    # Chat entry point for full conversation
    def chat(self) -> list[BaseMessage]:
        """Chat entry point for full conversation."""
        
        thread = {"configurable": {"thread_id": self.user_id}} # Use user_id as thread_id to pull conversation history from RAM
        new_message = [HumanMessage(content=self.user_query)]

        # calling manager agent to start working
        updated = self.app.invoke({"messages": new_message}, thread) # Here we added the user query as a new message to the Graph state
        
        # Print all messages in the updated state
        print("\n🗨️ Conversation History:")
        for m in updated["messages"]:
            m.pretty_print()
        return updated["messages"]
    
    

    
