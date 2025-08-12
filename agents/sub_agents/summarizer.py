from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .base import BaseSubAgent

class SummarySubAgent(BaseSubAgent):
    """
    Summarizes a topic/article using RAG (article- or title-level).
    If the user passes a topic/title, we search titles; if broad, we search articles.
    """

    def __init__(self, deps=None):
        super().__init__(deps)
        self.llm = ChatOpenAI(temperature=0.3)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(
                """You write crisp, factual news summaries (5–8 sentences).
                    Use ONLY the provided context. If context is thin, say so.

                    Topic / Request:
                    {question}

                    Context:
                    {context}

                    Write the summary and include a compact "Sources:" line with the URLs used."""
            )
        )

    def _format_context(self, hits) -> str:
        blocks = []
        for h in hits:
            m = h[4] or {}
            title = m.get("title") or ""
            url = m.get("url") or ""
            text = m.get("text") or m.get("content") or ""
            blocks.append(f"{title}\n{text}\nSource: {url}")
        return "\n\n".join(blocks)

    def run(self, query: str) -> str:
        retriever = self.deps.get("retriever")
        if not retriever:
            return self.chain.run(question=query, context="")

        # Heuristic: short queries → search titles, otherwise search articles
        mode = "title" if len(query) < 40 else "article"

        try:
            hits = retriever.retrieve(
                question=query,
                mode=mode,
                k_initial_matches=50,
                k_final_matches=5
            )
            context = self._format_context(hits)
        except Exception as e:
            context = ""
            return f"(RAG unavailable: {e})\n" + self.chain.run(question=query, context=context)

        return self.chain.run(question=query, context=context)

    def describe(self) -> str:
        return "Summarizes articles/topics using retrieved context (titles or full articles)."
