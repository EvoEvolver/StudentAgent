import os
from dotenv import load_dotenv

from student.agent import RaspaAgent

load_dotenv()
MEMORY_PATH = os.path.join(os.path.dirname(__file__), "agent", "memory", "raspa_memory", "mc5_5", "memory.txt")


# Updated to use RaspaAgent with todo list functionality
agent = RaspaAgent(provider="openai", path = "output/manager/")
agent.set_auto(True)
agent.memory_agent.load_memory(MEMORY_PATH)

# Use run_with_todo_list() for todo list-based execution
reply = agent.run_with_todo_list("""
Use RASPA2 to calculate the Henry’s coefficient of ethane in IRMOF-1 at 298 K using the Widom insertion method.
Use RASPA2 to calculate the Henry’s coefficient of Methane in IRMOF-1 at 298 K using the Widom insertion method.
Given:
                                 

Framework: IRMOF-1
Adsorbate: ethane (TraPPE model)
Adsorbate: Methane (CH₄)
Temperature: 298 K
Unit cells: 2 × 2 × 2
Number of Widom insertions: 50,000
Force field: GenericMOFs

Tasks:
                                 

Write a RASPA input file to perform a Henry coefficient calculation.
Run the simulation and locate the output file HenryCoefficientAverage.dat.
Run the simulation and locate the output file
Report the Henry’s coefficient in mol/kg/Pa and convert it to mol/kg/bar.
Briefly explain how temperature would affect the Henry’s coefficient.
""")
print(reply)
