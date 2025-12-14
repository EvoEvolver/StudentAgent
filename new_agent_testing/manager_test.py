import asyncio
import datetime
import os
from typing import Set

from dotenv import load_dotenv

from student.agent import RaspaAgent

load_dotenv()
MEMORY_PATH = os.path.join(os.path.dirname(__file__), "agent", "memory", "raspa_memory", "mc5_5", "memory.txt")

# Updated to use RaspaAgent with todo list functionality
working_path = "output/manager/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
agent = RaspaAgent(model_name="openai:gpt-5-mini", path=working_path)

# Use run_with_todo_list() for todo list-based execution
reply: Set[str] = asyncio.run(agent.run("""
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
# write to file
with open(os.path.join(working_path, "manager_output.txt"), "w") as f:
    f.write("\n".join(reply))
