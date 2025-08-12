from typing import List
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .base import BaseSubAgent

class QASubAgent(BaseSubAgent):
    """
    Answers questions using RAG (chunk-level) + an LLM.
    Expects self.deps["retriever"] to be a RAGRetriever with .retrieve(...)
    """

    def __init__(self, deps=None):
        super().__init__(deps)
        self.llm = ChatOpenAI(temperature=0)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(
                """You are a precise news assistant. Use ONLY the provided context to answer.
                    If the answer is not in the context, say you don't know and suggest a follow-up.

                    Question:
                    {question}

                    Context (snippets with sources):
                    {context}

                    Return a concise, neutral answer. Include short in-text citations like [1], [2] that map to the sources below."""
            )
        )

    def _format_context(self, hits) -> str:
        # hits are tuples: (final_score, id, similarity, recency_weight, metadata_dict)
        blocks: List[str] = []
        for idx, h in enumerate(hits, 1):
            meta = h[4] or {}
            text = meta.get("text") or meta.get("content") or ""
            url  = meta.get("url") or ""
            title = meta.get("title") or ""
            blocks.append(f"[{idx}] {title}\n{text}\nSource: {url}")
        return "\n\n".join(blocks)

    def run(self, query: str) -> str:
        retriever = self.deps.get("retriever")
        if not retriever:
            # Fallback: no RAG available
            return self.chain.run(question=query, context="")

        try:
            # Chunk mode gives the most grounded answers for Q&A
            hits = retriever.retrieve(
                question=query,
                mode="chunk",
                k_initial_matches=80,
                k_final_matches=6
            )
            context = self._format_context(hits)
        except Exception as e:
            # If FAISS files are missing or any runtime error, degrade gracefully
            context = ""
            return f"(RAG unavailable: {e})\n" + self.chain.run(question=query, context=context)

        return self.chain.run(question=query, context=context)

    def describe(self) -> str:
        return "Answers questions using retrieval-augmented generation from recent news."
