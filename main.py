from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import the new LangGraph-based manager
from agents.manager_agent.manager_agent import ManagerAgent

load_dotenv()

app = FastAPI(title="Noah AI News Agent", version="1.0")

# Allow CORS (adjust in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init the LangGraph manager
manager_agent = ManagerAgent()

# Request schema
class AskRequest(BaseModel):
    query: str

# Response schema
class AskResponse(BaseModel):
    result: str
    agent_used: str

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    query = request.query

    # Get the formatted result from LangGraph manager
    raw_response = manager_agent.route(query)

    # Extract agent_used from the formatted response
    agent_used = "unknown"
    if "🔁 Routed to `" in raw_response:
        try:
            agent_used = raw_response.split("`")[1]
        except Exception:
            pass

    # Everything after the "\n\n" is the final answer
    if ":\n\n" in raw_response:
        result = raw_response.split(":\n\n", 1)[1]
    else:
        result = raw_response

    return AskResponse(result=result.strip(), agent_used=agent_used)