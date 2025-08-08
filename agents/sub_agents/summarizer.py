from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .base import BaseSubAgent

class SummarySubAgent(BaseSubAgent):
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.5)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template("Summarize the following content: {query}")
        )

    def run(self, query: str) -> str:
        return self.chain.run(query=query)

    def describe(self) -> str:
        return "Summarizes articles or paragraphs."
