from .agent_student import StudentAgent
from .agent_memory import Memory

from .tools.tools_raspa import CoreMofLoader, TrappeLoader, ExecuteRaspa, ReadFile, WriteFile, InputFile, InspectFiles, OutputParser
from .tools.tools import Tool
from .tools.tools_memory import AddMemory, ModifyMemory, RecallMemory

from mllm import Chat
from typing import List, Dict, Union
import json
import os
from .utils import *


class RaspaAgent(StudentAgent):

    memory : Memory
    tools : Dict[str, Tool]
    system_prompt : str
    chat : Chat
    id : int

    molecule_memory : Memory
    framework_memory: Memory
    path : str
    path_add : str
    auto_run : bool


    def __init__(self, path="output"):
        # molecule_memory = init_molecule_name_memory()
        #framework_memory = init_framework_memory()

        raspa_tools = {
            "coremof": CoreMofLoader(path),
            "trappe": TrappeLoader(path),
            "raspa": ExecuteRaspa(),
            "input": InputFile(),
            "read": ReadFile(),
            "write": WriteFile(),
            "files": InspectFiles(),
            "output": OutputParser(),
        }

        super().__init__(tools=raspa_tools)

        self.reset(path)        # base path
        self.path_add = ""      # add onto path for simulations
        self.auto_run = False
    
    def setup_path(self, path : str) -> None:
        os.makedirs(path, exist_ok=True)
        self.path = path
        for tool in self.tools.values():
            if hasattr(tool, "path"):
                tool.path = path
        return 
    
    def set_path_add(self, path_add):
        self.path_add = path_add
        full_path = self.get_full_path()
        os.makedirs(full_path, exist_ok=True)
        for tool in self.tools.values():
            if hasattr(tool, "path_add"):
                tool.path_add = path_add
        return full_path
        
    def get_full_path(self):
        return os.path.join(self.path, self.path_add)


    def reset(self, path=None):
        '''
        # TODO: fully reset everything except for memory
        '''
        if path is not None:
            self.setup_path(path)
        self.tools['coremof'].has_file = False 
        self.tools['trappe'].has_file = False
        self.tools['input'].has_file = False
        return


    def check_files(self):
        if self.tools['coremof'].has_file and self.tools['trappe'].has_file and self.tools['write'].has_file:
            return True
        return False


    def run(self, prompt):
        self.setup()
        remove_tools = self.get_tool_mask()
        
        # Evaluate input (generate input files)
        res = super().run(prompt, remove_tools=remove_tools)
        
        # Ask for feedback
        # Run and return output
        output = res
        return output


    def get_tool_mask(self):
        mask = []

        # self.auto_run controls visibility of the raspa tool:
        if self.auto_run is False:
            mask.append("raspa")

        # all required files need to be present:
        elif not self.check_files():
            return ["raspa"]

        return mask


    def setup(self):
        #self.init_special_memories()
        return
    
    def set_auto(self, auto):
        self.auto_run = auto
        return


    '''
    def init_special_memories(self):
        for tool in self.tools.values():
            if hasattr(tool, "init_memory_prompt"):
                prompt = tool.init_memory_prompt()
                self.add_explicit_knowledge(prompt)
        return
    '''


