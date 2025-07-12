from .agent_student import StudentAgent
from .memory import Memory

from .tools.tools_raspa import CoreMofLoader, ExecuteRaspa, ReadFile, WriteFile, InputFile, InspectFiles, OutputParser, FrameworkLoader, MoleculeLoader
from .tools.tools import Tool

from mllm import Chat
from typing import List, Dict, Union
import re
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
            # CoreMofLoader(path),
            framework_loader,
            # TrappeLoader(path),
            MoleculeLoader(path),
            ExecuteRaspa(agent=self),
            InputFile(),
            ReadFile(),
            WriteFile(),
            # InspectFiles(),
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
        self._advance_to_next_folder()
        self.reset(path)

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

    def _file_overview(self):
        current_directory = self.path_add
        files_all = [i for i in all_files(current_directory) if i not in ['trappe_molecule_list.json']]
        file_list = "\n".join(f"- {f}" for f in files_all) if files_all else "Empty"
        file_overview = f"\n\n<file_overview>\nCurrent directory: {current_directory}\nFiles:\n{file_list}\n</file_overview>"
        return file_overview

    def remove_file_history(self, message):
        if isinstance(message, dict) and 'content' in message and isinstance(message['content'], dict):
            text = message['content'].get('text', '')
            # Remove <file_overview>...</file_overview> block
            cleaned_text = re.sub(r'<file_overview>.*?</file_overview>', '', text, flags=re.DOTALL)
            message['content']['text'] = cleaned_text.strip()
        return message

    def chat_length(self):
        return len(self.chat.messages)

    def run(self, prompt, max_iter=15):
        remove_tools = self.get_tool_mask()
        file_overview = self._file_overview()
        prompt += file_overview
        
        l0 = self.chat_length()
        res = super().run(prompt, remove_tools=remove_tools, max_iter=max_iter)
        l1 = self.chat_length()
        self.remove_file_history(self.chat.messages[l0-l1])
        return res


    def _advance_to_next_folder(self):
        """
        Find the next available folder (simulation_N) in self.path, create it, and set as path_add.
        """
        base_path = self.path
        os.makedirs(base_path, exist_ok=True)
        prefix = "simulation_"
        existing_folders = [
            d for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d)) and d.startswith(prefix) and d[len(prefix):].isdigit()
        ]
        if existing_folders:
            nums = [int(d[len(prefix):]) for d in existing_folders]
            max_num = max(nums)
            # If the highest-numbered folder is empty, reuse it
            highest_folder = f"{prefix}{max_num}"
            if not os.listdir(os.path.join(base_path, highest_folder)):
                next_num = max_num
            else:
                next_num = max_num + 1
        else:
            next_num = 1
        new_folder = f"{prefix}{next_num}"
        new_path = os.path.join(base_path, new_folder)
        os.makedirs(new_path, exist_ok=True)
        self.set_path_add(new_folder)
        return new_path



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


