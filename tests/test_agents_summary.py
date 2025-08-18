# from agents.sub_agents.summarizer import SummarySubAgent

# def test_summary_agent_title_mode_for_short_query(fake_retriever, stub_chain_run):
#     agent = SummarySubAgent(deps={"retriever": fake_retriever})
#     # Short query ⇒ title mode (per heuristic in code)
#     out = agent.run("Ceasefire talks")
#     assert "[LLMStub]" in out
#     assert fake_retriever.last_kwargs is not None
#     assert fake_retriever.last_kwargs["mode"] == "title"

# def test_summary_agent_article_mode_for_long_query(fake_retriever, stub_chain_run):
#     agent = SummarySubAgent(deps={"retriever": fake_retriever})
#     q = "Please provide a detailed summary of the latest developments around the ceasefire negotiations including mediator statements."
#     out = agent.run(q)
#     assert "[LLMStub]" in out
#     assert fake_retriever.last_kwargs["mode"] == "article"
