# agents/sub_agents/qa.py
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from agents.prompts import QA_PROMPT

def build_qa_agent(model: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0)
    return create_react_agent(
        model=llm,
        tools=[],  # no tools for now
        prompt=QA_PROMPT,
        name="qa",
    )
