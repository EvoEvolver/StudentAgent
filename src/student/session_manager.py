import json, os, uuid
from .agent import RaspaAgent, StudentAgent, Agent

from pydantic import BaseModel
from typing import Any, List, Dict

SESSION_DIR = "./sessions"
DEFAULT_TYPE = "raspa"
DEFAULT_PROVIDER = "anthropic"
CSD_PATH = None
OUTPUT_PATH = "raspa_output"
CHECKPOINT_DIR = "checkpoints"

class SessionState(BaseModel):
    agent_type: str
    session_id: str
    provider: str
    path: str
    active_learning: bool
    output_path: str


def create_session(session_id=None, agent_type="default", provider="default") -> str:
    if session_id is None:
        session_id = str(uuid.uuid4())

    if provider == "default":
        provider = DEFAULT_PROVIDER
    
    if agent_type == "default":
        agent_type = DEFAULT_TYPE

    session_path = os.path.join(SESSION_DIR, session_id)
    state = SessionState(
        agent_type=agent_type, 
        session_id=session_id,
        provider=provider,
        path=session_path,
        active_learning=False,
        output_path=OUTPUT_PATH
    )
    
    session_path = os.path.join(SESSION_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)
    
    with open(os.path.join(session_path, "state.json"), "w") as f:
        json.dump(state.dict(), f)
    
    return session_id


def load_session(session_id):
    session_path = os.path.join(SESSION_DIR, session_id)
    if not os.path.isdir(session_path):
        return None
    try:
        with open(os.path.join(session_path, "state.json"), "r") as f:
            state = json.load(f)
        return SessionState(**state)
    except Exception as e:
        print(f"Failed to load session {session_id}: {e}")
        return None

def save_session(session_id, state=None):
    session_path = os.path.join(SESSION_DIR, session_id)
    if not os.path.isdir(session_path):
        raise FileNotFoundError(f"Session {session_id} directory does not exist!")
    
    if state is not None:
        with open(os.path.join(session_path, "state.json"), "w") as f:
            json.dump(state.dict(), f)


def load_agent(session):
    if session.agent_type == "RASPA":
        agent = RaspaAgent(
            provider=session.provider, 
            path=os.path.join(session.path, OUTPUT_PATH), 
            csd_path=CSD_PATH, 
            active_learning=session.active_learning
        )

    elif session.agent_type == "Student":
        agent = StudentAgent(provider=session.provider)
    else:
        agent = Agent(provider=session.provider)

    loading_error = agent.load(session.path+CHECKPOINT_DIR)
    if loading_error is not None:
        print(loading_error)
    return agent

def save_agent(session, agent):
    agent.save(session.path+CHECKPOINT_DIR)    # overwrite!


def run_session(session, agent, input):
    agent = load_agent(session)
    
    response = agent.run(input)
    if not isinstance(response, dict):
        response = {"result": response}

    save_agent(session, agent)
    
    if type(Agent) in [StudentAgent, RaspaAgent]:
        session.active_learning = agent.active_learning
    session.provider = agent.provider
    
    return response, session