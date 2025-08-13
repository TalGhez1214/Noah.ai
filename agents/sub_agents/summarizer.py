# agents/sub_agents/summarizer.py
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from agents.prompts import SUMMARY_PROMPT

def build_summary_agent(model: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0.3)
    return create_react_agent(
        model=llm,
        tools=[],  # no tools for now
        prompt=SUMMARY_PROMPT,
        name="summary",
    )
