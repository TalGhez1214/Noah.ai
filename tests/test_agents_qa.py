# from agents.sub_agents.qa import QASubAgent

# def test_qa_agent_uses_retriever_context(fake_retriever, stub_chain_run):
#     agent = QASubAgent(deps={"retriever": fake_retriever})
#     out = agent.run("Who is involved in the scandal?")
#     # Assert our stub LLM saw the question
#     assert "[LLMStub] q='Who is involved in the scandal?'" in out
#     # Ensure the context was formatted from retriever hits
#     assert "Item A" in out or "https://a" in out
#     assert "Item B" in out or "https://b" in out
