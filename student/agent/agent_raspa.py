from .agent_student import StudentAgent
from .agent_memory import Memory

from .tools.tools_raspa import CoreMofLoader, TrappeLoader, ExecuteRaspa, ReadFile, WriteFile
from .tools.tools import Tool
from .tools.tools_memory import AddMemory, ModifyMemory, RecallMemory

from .tools.input_gen import init_molecule_name_memory, init_framework_memory

from mllm import Chat
from typing import List, Dict, Union
import json
import os


class RaspaAgent(StudentAgent):

    memory : Memory
    tools : Dict[str, Tool]
    system_prompt : str
    chat : Chat
    id : int

    molecule_memory : Memory
    framework_memory: Memory
    path : str


    def __init__(self, path="output"):
        molecule_memory = init_molecule_name_memory()
        framework_memory = init_framework_memory()

        raspa_tools = {
            "coremof": CoreMofLoader(framework_memory, path),
            "trappe": TrappeLoader(molecule_memory, path),
            "raspa": ExecuteRaspa(),
            "read": ReadFile(),
            "write": WriteFile(),
        }

        super().__init__(tools=raspa_tools)

        self.setup_path(path)
    
    
    def setup_path(self, path : str) -> None:
        os.makedirs(path, exist_ok=True)
        self.path = path
        return 


    def run(self):
        super().run()
        # Setup
        # Get input
        # Evaluate input (generate input files)
        #
        # Ask for feedback
        # Run and return output
    
        pass



