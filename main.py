from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from agents.manager_agent.manager_agent import ManagerAgent

load_dotenv()

app = FastAPI(title="Noah AI News Agent", version="1.0")

# Allow CORS (adjust allow_origins in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["http://localhost:3000"] for your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    # Call manager agent
    raw_response = manager_agent.route(query)

    # Try to extract agent used (based on your response format)
    if "🔁 Routed to `" in raw_response:
        try:
            agent_used = raw_response.split("`")[1]
            result = raw_response.split(":\n\n")[-1]
        except Exception:
            agent_used = "unknown"
            result = raw_response
    else:
        agent_used = "unknown"
        result = raw_response

    return AskResponse(result=result, agent_used=agent_used)
