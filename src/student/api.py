from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .session_manager import create_session, load_session, save_session, run_session

app = FastAPI()

class AgentRequest(BaseModel):
    input: str
    session_id: str = None
    agent_type: str = None

class AgentResponse(BaseModel):
    session_id: str
    response: dict

@app.post("/agent", response_model=AgentResponse)
def agent_endpoint(req: AgentRequest):
    # 1. Load or create session
    if req.session_id:
        session = load_session(req.session_id)
        if session is None:
            # Session ID provided but not found: create new session with the given session_id
            session_id = create_session(req.session_id, req.agent_type)
            session = load_session(session_id)
        else:
            session_id = req.session_id
    else:
        # No session_id provided: create new session
        session_id = create_session(session_id = req.session_id, agent_type=req.agent_type)
        session = load_session(session_id)
    
    # 2. Interact with agent
    response, updated_session = run_session(session, req.input)
    
    # 3. Save session state
    save_session(session_id, updated_session)
    
    # 4. Return response
    return AgentResponse(session_id=session_id, response=response)
