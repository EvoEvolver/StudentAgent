from .agent_memory import Memory
from .tools.tools import Tool
from .tools.tools_memory import AddMemory, ModifyMemory, RecallMemory
from .agent import Agent

from mllm import Chat
from typing import List, Dict, Union
import json
from .utils import *


class StudentAgent(Agent):
    memory : Memory

    def __init__(self, tools: Dict[str, Tool] = {}, cache=None, expensive=None, version="v1.xml", provider="openai"):
        super().__init__(tools, cache, expensive, version, provider)
        
        self.memory = Memory()
        self.add_memory_tools()

        self.special_keywords = {"explicit knowledge" : keyword("explicit knowledge"),}

    def add_memory_tools(self):
        add = AddMemory(self.memory)
        modify = ModifyMemory(self.memory)
        recall = RecallMemory(self.memory)
        
        self.tools[add.name] = add
        self.tools[modify.name] = modify
        self.tools[recall.name] = recall

    

    def add_explicit_knowledge(self, prompt):
        prompt += instructions(f"""
            You are required to store this information in your memory as one block.
            Extract different relevant keywords and add the special keyword {self.get_special_keywords("explicit knowledge")} to it.
            Try to recall it and modify the keywords, if required!
        """)
        remove_tools = self.get_memory_tool_mask(memory_only=True)
        res = self.run(prompt, remove_tools=remove_tools)
        return res
    

    def get_special_keywords(self, key):
        return self.special_keywords.get(key, "")


    def get_memory_tool_mask(self):
        return [name for name in self.tools.keys() if name not in ["add", "recall", "modify"]]


    def load_memory(self, file:str):
        if os.path.exists(file):
            try:
                self.memory.load(file)
                return True
            except Exception as e:
                print("Error loading memory: ", e)
                return None
        return False
    

    def save_memory(self, file: str):
        try:
            self.memory.save(file)
            return True
        except Exception as e:
            print("Error saving memory: ", e)
            return False
        