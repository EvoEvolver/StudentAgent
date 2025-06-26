from .agent_student import StudentAgent
from .memory import Memory

from .tools.tools_raspa import CoreMofLoader, TrappeLoader, ExecuteRaspa, ReadFile, WriteFile, InputFile, InspectFiles, OutputParser, FrameworkLoader, MoleculeLoader
from .tools.tools import Tool

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


    def __init__(self, path="output", version="v1", provider="anthropic", csd_path=None, verbose=False):
        if csd_path is not None:
            framework_loader = FrameworkLoader(path, coremof=False, csd_path=csd_path)
        else:
            print("A CSD path is required to access the coremof files.")
            framework_loader = FrameworkLoader(path, coremof=False)

        raspa_tools = [
            #"coremof": CoreMofLoader(path),
            framework_loader,
            # TrappeLoader(path),
            MoleculeLoader(path),
            ExecuteRaspa(),
            InputFile(),
            ReadFile(),
            WriteFile(),
            InspectFiles(),
            OutputParser()
        ]
        tools = {
            tool.name : tool
            for tool in raspa_tools
        }

        super().__init__(tools=tools, version=version, provider=provider, verbose=verbose)

        self.reset(path)        # base path
        self.path_add = ""      # add onto path for simulations
        self.auto_run = False
        self.add_raspa_prompt()

    def add_raspa_prompt(self):
        prompt = self._build_prompt("raspa", "v1")
        self.reset_system_prompt(prompt, append=True)
    
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
        for tool in self.tools:
            if hasattr(tool, "has_file"):
                tool.has_file =False
        return


    def check_files(self):
        if all([tool.has_file for tool in self.tools if hasattr(tool, "has_file")]):
            return True
        return False


    def run(self, prompt, max_iter=15):
        self.setup()
        remove_tools = self.get_tool_mask()
        
        # Evaluate input (generate input files)
        res = super().run(prompt, remove_tools=remove_tools, max_iter=max_iter)
        
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
        #elif not self.check_files():
        #    return ["raspa"]

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


