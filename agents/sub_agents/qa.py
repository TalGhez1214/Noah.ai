from typing import List
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from agents.prompts import QA_PROMPT

def build_qa_agent(retriever, model: str = "gpt-4o-mini", top_k: int = 6):
    """
    Builds a ReAct agent for Q&A with a single retrieval tool.
    """

    @tool("get_knowledge_for_answer")
    def get_knowledge_for_answer(query: str) -> str:
        """
        Retrieve the most relevant pieces of knowledge (short text chunks)
        from the news database for answering a specific user question.
        
        Input:
            query (str): The exact question the user asked.

        Output:
            A plain text string containing ONLY the concatenated 'chunk' fields
            from the top retrieved snippets
        """
        try:
            hits = retriever.retrieve(
                question=query,
                mode="chunk",  # chunk-level retrieval
                k_initial_matches=80,
                k_final_matches=top_k
            )
        except Exception:
            return ""

        chunks: List[str] = []
        
        for h in hits:
            ch = h.get("chunk") or ""
            if ch:
                chunks.append(ch.strip())
        return "\n\n---\n\n".join(chunks)

    llm = ChatOpenAI(model=model, temperature=0)
    return create_react_agent(
        model=llm,
        tools=[get_knowledge_for_answer],
        prompt=QA_PROMPT,
        name="qa",
    )
