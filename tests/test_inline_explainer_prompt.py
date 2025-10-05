import os
from datetime import date

import pytest
from langchain_core.messages import SystemMessage

# Import your prompt builder + agent
from agents.inline_agents.prompts import explainer_prompt
from agents.inline_agents.explainer import ExplainerAgent


def test_explainer_system_prompt_prints(capsys):
    """
    Unit test: build the exact SystemMessage the Explainer will receive
    and print it so you can debug the full prompt content.
    This test does NOT call OpenAI and will pass without API keys.
    """
    highlighted = "Warfarin inhibits vitamin K epoxide reductase."
    page_content = (
        "This article explains anticoagulation mechanisms. "
        "Warfarin reduces activation of clotting factors II, VII, IX, and X."
    )
    today = date.today().isoformat()

    # Rebuild initial messages exactly as the agent will compose them
    init_msgs = explainer_prompt(
        {"messages": []},
        {"configurable": {
            "current_page_content": page_content,
            "highlighted_text": highlighted,
            "today": today,
        }},
    )

    # Extract + print the SystemMessage (so you can see the full prompt)
    sys_msgs = [m for m in init_msgs if isinstance(m, SystemMessage)]
    assert sys_msgs, "Expected at least one SystemMessage from explainer_prompt()."
    sys_msg = sys_msgs[0]

    # Helpful sanity checks
    assert highlighted in sys_msg.content
    assert page_content in sys_msg.content
    assert today in sys_msg.content

    # Print for debugging (shown with `-s`)
    print("\n===== SYSTEM PROMPT (Explainer) =====\n")
    print(sys_msg.content)
    print("\n===== END SYSTEM PROMPT =====\n")

    # Ensure printed output is there (pytest captures unless -s is used)
    captured = capsys.readouterr()
    assert "SYSTEM PROMPT (Explainer)" in captured.out


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key.")
def test_explainer_call_smoke():
    """
    Integration smoke test (optional): actually invoke the ExplainerAgent.call()
    to ensure the wiring works. Skips if OPENAI_API_KEY is not set.
    """
    agent = ExplainerAgent(model="gpt-4o-mini", temperature=0.2)
    msgs = agent.call(
        highlighted_text="Warfarin inhibits vitamin K epoxide reductase.",
        current_page_content="Warfarin reduces activation of clotting factors II, VII, IX, and X.",
        thread_id="test-inline-explainer",
    )

    # Should produce at least one AI message
    from langchain_core.messages import AIMessage
    assert any(isinstance(m, AIMessage) for m in msgs), "Explainer did not return an AIMessage."
