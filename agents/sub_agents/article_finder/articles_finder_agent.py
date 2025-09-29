from typing import List
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from agents.prompts import article_finder_prompt
from langchain_core.messages import AIMessage
from ..base import BaseSubAgent
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from agents.sub_agents.article_finder.prompts import AGENT_PROMPT

# ---------- Structured Output ----------
class RespondFormat(BaseModel):
    # Copy the field VALUES exactly as written in the FINAL assistant message.
    # Do not paraphrase, normalize, spell-correct, or change punctuation/casing/whitespace.

    summary: str = Field(
        ...,
        description=(
            "2-3 sentences summary of the article. Empty string if no summary found."
        ),
    )

    quote: str = Field(
        ...,
        description=(
            "A key direct quote from the article. Empty string if no quote found."
        ),
    )
    

class ArticalFinderSubAgent(BaseSubAgent):
    """
    Builds a ReAct agent for find relevant articles for a specific user query.
    """

    def __init__(self, 
                 retriever, 
                 prompt: str,
                 model: str = "gpt-4o-mini"
                 ) -> None:
        
        self.name = "articles_finder_agent"
        self.description = "This agent finds the most relevant articles for the user query"
        self.retriever = retriever

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_PROMPT),
            MessagesPlaceholder("messages"),
        ])


        # ---------- LLMs ----------
        self.llm = ChatOpenAI(model=model, temperature=0.2)
        self.parser_model = self.llm.with_structured_output(RespondFormat)
        
    def call(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        This node handles the articles finder agent, which retrieves relevant articles
        based on the user's query and returns them as a dictionary response for each article:
        {
            "title": "Article Title",
            "Summary": "Article Summary",
            "Key Quote": "Key Quote from the article",
        }
        """
        retrieve_state = {
            "messages": state.get("messages", []),
            "user_query": state.get("user_query", ""),
            "requested_k": 5,  # Number of articles to retrieve
        }
        articles = self.retriever.invoke(retrieve_state).get("top_results", [])
        modals = []
        relevant_articles = []

        parser_chain = self.prompt | self.parser_model

        for article in articles:
            try:
                response = parser_chain.invoke({"messages": state["messages"],
                                        "user_query": state["user_query"],
                                        "article": article,
                                        })
            
                # Convert to dict safely
                modal = response.model_dump() if hasattr(response, "model_dump") else dict(response)

                modal["title"] = article["title"]
                modal["author"] = article.get("author", "Unknown")
                modal["url"] = article.get("url", "")


                if modal["summary"] and modal["quote"]:
                    modals.append(modal)
                    relevant_articles.append(article)
            except Exception as e:
                print(f"Error parsing structured output: {e}")
                
        return {
            "messages": [AIMessage(content=f"{relevant_articles}")], 
            "agent": "articles_finder",
            "ui_payload": {"type": "articles","data": modals},
        }