import asyncio
import os
from dotenv import load_dotenv

from student.agent import RaspaAgent
from student.agent.agent_todolist import TodoListAgent

load_dotenv()
MEMORY_PATH = os.path.join(os.path.dirname(__file__), "agent", "memory", "raspa_memory", "mc5_5", "memory.txt")


# Updated to use RaspaAgent with todo list functionality
agent = RaspaAgent(model_name="openai:gpt-5-mini", path = "output/manager/")

# Use run_with_todo_list() for todo list-based execution
reply = asyncio.run(agent.run("""
Use RASPA2 to calculate the Henry’s coefficient of Methane in IRMOF-1 at 298 K using the Widom insertion method.
Given:
                                 

Framework: IRMOF-1
Adsorbate: Methane (CH4)
Temperature: 298 K
Unit cells: 2 × 2 × 2
Number of steps: 5,000
Force field: GenericMOFs

Tasks:
                                 
Write a RASPA input file to perform a Henry coefficient calculation (use the monte carlo move WidomProbability 1.0!)
Report the calculated Henry’s coefficient value.
"""))
print(reply)
