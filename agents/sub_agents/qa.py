from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .base import BaseSubAgent

class QASubAgent(BaseSubAgent):
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template("Answer the following question: {query}")
        )

    def run(self, query: str) -> str:
        return self.chain.run(query=query)

    def describe(self) -> str:
        return "Answers factual questions."
