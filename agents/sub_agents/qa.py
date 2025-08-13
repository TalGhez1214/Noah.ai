# agents/sub_agents/qa.py

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from agents.prompts import QA_PROMPT

def build_qa_agent(model="gpt-4o"):
    return create_react_agent(
        model=ChatOpenAI(model=model, temperature=0),
        tools=[],  # no tools for now
        prompt=QA_PROMPT,
        name="qa",
    )
