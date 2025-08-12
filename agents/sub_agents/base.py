from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

class BaseSubAgent(ABC):
    """Base class for all sub-agents."""

    def __init__(self, deps: Optional[Dict[str, Any]] = None):
        # deps can hold shared tools like retriever, db, cache, etc.
        self.deps = deps or {}

    @abstractmethod
    def run(self, query: str) -> str:
        ...

    @abstractmethod
    def describe(self) -> str:
        ...
