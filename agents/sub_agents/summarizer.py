from typing import List
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from agents.prompts import SUMMARY_PROMPT

def build_summary_agent(retriever, model: str = "gpt-4o-mini", top_k: int = 5):
    """
    Builds a ReAct agent for summarization with a single retrieval tool.
    """

    @tool("get_articles_for_summary")
    def get_articles_for_summary(topic: str) -> str:
        """
        Retrieve the articles for summarizing the user query
        from the news database.

        Input:
            topic (str): The topic or article subject provided by the user.

        Output:
            A plain text string containing ONLY the concatenated 'content' fields
            from the top retrieved articles.
        """
        try:
            hits = retriever.retrieve(
                question=topic,
                mode="article",  # article-level retrieval for summaries
                k_initial_matches=50,
                k_final_matches=top_k
            )
        except Exception:
            return ""

        contents: List[str] = []
        for h in hits:
            c = h.get("content") or ""
            if c:
                contents.append(c.strip())
        return "\n\n---\n\n".join(contents)

    llm = ChatOpenAI(model=model, temperature=0.3)
    return create_react_agent(
        model=llm,
        tools=[get_articles_for_summary],
        prompt=SUMMARY_PROMPT,
        name="summary",
    )
