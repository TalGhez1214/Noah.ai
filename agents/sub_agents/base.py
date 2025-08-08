from abc import ABC, abstractmethod

class BaseSubAgent(ABC):
    """Abstract base class for all sub-agents."""

    @abstractmethod
    def run(self, query: str) -> str:
        pass

    @abstractmethod
    def describe(self) -> str:
        """Short description of what this sub-agent does."""
        pass
