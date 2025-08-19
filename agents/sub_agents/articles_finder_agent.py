from typing import List
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from agents.prompts import ARTICLES_FINDER_PROMPT

from pydantic import BaseModel


class ArticalFinderAgent():
    """
    Builds a ReAct agent for find relevant articles for a specific user query.
    """

    def __init__(self, retriever, model: str = "gpt-4o-mini"):
        
        self.retriever = retriever
        self._llm = ChatOpenAI(model=model, temperature=0.8)

        # Define the response schemas for the output parser
        self._response_schemas = [
            ResponseSchema(name="Summary", description="One-sentence summary of the text"),
            ResponseSchema(name="Key Quote", description="A key direct quote from the text"),
        ]

        self._output_parser = StructuredOutputParser.from_response_schemas(self._response_schemas)

        format_instructions = self._output_parser.get_format_instructions()
        # Define the prompt template for the agent
        self._prompt = PromptTemplate(
                input_variables=["user_query", "title", "author", "content"],
                partial_variables={"format_instructions": format_instructions},
                template=ARTICLES_FINDER_PROMPT,
            )
        
    def get_knowledge_for_answer(self, user_query: str) -> str:
        """
        Retrieve the most relevant pieces of knowledge (short text chunks)
        from the news database for answering a specific user question.
        
        Input:
            user_query (str): The exact question the user asked.

        Output:
            A plain text string containing ONLY the concatenated 'chunk' fields
            from the top retrieved snippets
        """
        try:
            hits = self.retriever.retrieve(
                query=user_query,
                semantic_file="full_content",  
                keywords_fields=["title", "author", "content"], 
                k_final_matches=3
            )
        except Exception as e:
            return f"Error retrieving knowledge: {e}"

        return hits

    def build_articles_finder_agent(self, user_query: str="", article: dict = {}):
        return create_react_agent(
            model=self._llm,
            tools=[],
            prompt=self._prompt.format(user_query=user_query, **article),
            name="articles_finder",
        )
    
    def structured_output(self, llm_output: str):
        """
        Convert the output string from the agent into a structured ArticleSummary object.
        """
        for parse_try in range(3):
            # Attempt to parse the output using the structured output parser
            # This will raise an exception if the output is not valid JSON
            # or does not match the expected schema.
            try:
                parsed = self._output_parser.parse(llm_output)
                return parsed
            except Exception as e:
                print(f"⚠️ Parsing attempt {parse_try+1} failed:", e)
                # Optional: retry with a fix
                format_instructions = self._output_parser.get_format_instructions()
                fixed_prompt = f"""Your previous output was invalid: \n{llm_output} \n\n {format_instructions}\n
                Fix this into valid JSON only, nothing else.
                
                If you dont know the answer to one of the fields, just return "Invalid output" for that field."""
                llm_output = self._llm.invoke(fixed_prompt).content.strip()
                
        # Only raise after all retries failed
        raise ValueError("Failed to parse output after multiple attempts.") 