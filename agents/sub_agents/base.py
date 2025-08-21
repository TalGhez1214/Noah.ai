from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_core.runnables import Runnable




class BaseSubAgent(ABC):
    """Abstract base for all sub-agents used by the supervisor graph.


    Required methods:
    1) __init__ – create the ReAct agent (store under self.agent) and any tools.
    2) get_knowledge_for_answer – call your RAG retriever (signature unified).
    3) call – invoke the agent (pass messages / inputs and return output).


    Optional:
    4) structured_output – parse LLM output into JSON-like dict.


    Conventions:
    - self.name: unique node name used inside the LangGraph (e.g., "qa_agent").
    - self.agent: a Runnable (from create_react_agent) added as a node in the graph.
    """


    name: str
    description: str 
    agent: Runnable
    prompt: str


def __init__(self, *args: Any, **kwargs: Any) -> None: # pragma: no cover
    super().__init__()


@abstractmethod
def get_knowledge_for_answer(self, query: str) -> str:
    """Return concatenated context string from RAG for the given query."""


@abstractmethod
def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke with MessagesState-like input and return {"messages": ..., "agent": <label>}"""


def structured_output(self, llm_output: str) -> Optional[Dict[str, Any]]: # optional
    """Parse agent output into a structured dict. Default: return None."""
    return None