import sys
import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.manager_agent.manager_agent import ManagerAgent

# Create the manager
m = ManagerAgent()

# Render the Mermaid diagram to PNG bytes
png_bytes = m.app.get_graph().draw_mermaid_png()

# Save to file
output_path = os.path.join(os.path.dirname(__file__), "..", "manager_graph.png")
with open(output_path, "wb") as f:
    f.write(png_bytes)

print(f"✅ Graph saved to {output_path}")