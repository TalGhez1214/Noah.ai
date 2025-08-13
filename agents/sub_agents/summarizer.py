# agents/sub_agents/summarizer.py

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from agents.prompts import SUMMARY_PROMPT

def build_summary_agent(model="gpt-4o"):
    return create_react_agent(
        model=ChatOpenAI(model=model, temperature=0.3),
        tools=[],  # no tools for now
        prompt=SUMMARY_PROMPT,
        name="summary",
    )
