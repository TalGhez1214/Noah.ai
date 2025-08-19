from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from agents.manager_agent.manager_agent import ManagerAgent

load_dotenv()

app = FastAPI(title="Noah AI News Agent", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    result: str

@app.post("/ask", response_model=AskResponse)
def ask_user(request: AskRequest):
    
    user_id = "123" # TODO: Replace with actual user ID if needed using MongoDB request
    manager = ManagerAgent(user_query=request.query, user_id=user_id)
    messages = manager.chat()

    last_ai = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), "...")
    return AskResponse(result=last_ai)
