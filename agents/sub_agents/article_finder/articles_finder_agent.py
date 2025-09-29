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
                 model: str = "gpt-4o-mini", 
                 prompt = article_finder_prompt) -> None:
        
        self.name = "articles_finder_agent"
        self.description = "This agent finds the most relevant articles for the user query"
        self.retriever = retriever

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """ ## Role ##
                        You will receive an article dicitionary with title, auther, publish date, content etc. Extract the following:
                        1. summary: A short 2–3 sentence summary.
                        2. quote: The most relevant quote from the article to the user query (1-2 sentences with more then 5 words).

                        user_query: {user_query}

                        ## Rules ##
                        - The quete should be a direct quote from the article and you are not allowed to paraphrase it. (verbatim, including punctuation/casing). If no suitable quote exists, do not invent one—leave it out.
                        - The quete should be the most relevant quete from the article to the user_query.
                        - The summary should be a concise overview of the article's main points.
                        - Do NOT add facts, claims, numbers, dates, or interpretations that are not explicitly present in the article.
                        - If the article does not contain any relevant information for the user query, return an empty string for both fields. but
                        prefer sharing articles over not, so emit only if the article is irrelevant..
                        - if the article dosn't address the user filters (in title, author, date, topics/tags), return empty strings for both fields.


                        ## Article ##

                        {article}
                    """),
            MessagesPlaceholder("messages"),
        ])

        self.validator_prompt = ChatPromptTemplate.from_messages([
            ("system", """ ## Role
                            You give a short, friendly final response to the user based only on the provided articles.

                            ## Inputs
                            - user_query: {user_query}
                            - articles: {articles}  // each has title, author, published_at, topics/tags, content, url, etc.

                            ## What to do
                            1) Check relevance:
                            - Topic aligns with the user’s intent.
                            - If the user asked for a specific author or timeframe in the query itself, respect that.
                            - Otherwise, judge relevance by how well the articles address the user’s question.

                            2) Decide:
                            - Success: articles clearly address the query.
                            - Partial: some do, some don’t (or they’re close).
                            - No Match: none address the query.

                            3) Respond (fun + friendly):
                            - Start with: "Answer: "
                            - Keep 1–3 short sentences.
                            - No article details at all (no titles, quotes, authors, dates, URLs, numbers).
                            - Be positive, encouraging, and human.

                            ## Style
                            - Warm, light, and supportive. Emojis are allowed (1–2 max).
                            - No speculation or external knowledge.
                            - If Partial/No Match, briefly say the gap in general terms and what you shared instead (e.g., “closest recent items”), without details.

                            ## Examples
                            - Success: "Answer: I’ve added below the articles you asked about. Hope they help 🙌"
                            - Partial: "Answer: It seems like we don’t have articles by (auther name the user asked for), but I added a few on a similar theme. 🙂"
                            - No Match: "Answer: "I'm sorry but I don't have articles from - time period the user mentioned - would you like articls from a different time period?" 🌟"

                            ## Output
                            - A single short paragraph starting with "Answer: ".
                            - If you find it relevant finish with a question inviting further queries (only regarding finding articles) - do not invent here topics or dates because 
                            myabe the website dosn't have this.
                    """),
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
        }
        articles = self.retriever.invoke(retrieve_state).get("top_results", [])
        modals = []

        parser_chain = self.prompt | self.parser_model

        for article in articles:
            try:
                response = parser_chain.invoke({"messages": state["messages"],
                                        "user_query": state["user_query"],
                                        # "title": article["title"],
                                        # "author": article.get("author", "Unknown"),
                                        # "content": article.get("content", ""),
                                        "article": article,
                                        })
            
                # Convert to dict safely
                modal = response.model_dump() if hasattr(response, "model_dump") else dict(response)

                modal["title"] = article["title"]
                modal["author"] = article.get("author", "Unknown")
                modal["url"] = article.get("url", "")
                # Tag for your UI
                modal["type"] = "article"
            except Exception as e:
                print(f"Error parsing structured output: {e}")
                modal = {"Title": "", "Summary": "", "Key Quote": "", "url": "", "auther": "", "type": "article"}

            modals.append(modal)
        
        # validator_chain = self.validator_prompt | self.llm
        # validation_response = validator_chain.invoke({"messages": state["messages"],
        #                                             "user_query": state["user_query"],
        #                                             "articles": articles})  

        return {
            "messages": [AIMessage(content=f"{articles}")], 
            "agent": "articles_finder",
            "modals": modals,
        }