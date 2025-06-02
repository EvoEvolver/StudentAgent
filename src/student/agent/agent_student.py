from .agent import Agent
from .agent_memory import MemoryAgent
from .agent_memory import memory_agent_tools

from typing import List, Dict
from .tools.tools import Tool
from .utils import *
from .utils import context as c


class StudentAgent(Agent):
    def __init__(self, tools: Dict[str, Tool] = {}, cache=None, expensive=None, version=None, provider="openai"):
        super().__init__(tools=tools, cache=cache, expensive=expensive, version=version, provider=provider)

        self.reset_system_prompt(self.student_prompt())
        self.get_memory_tools()


    def get_memory_tools(self):
        memory = memory_agent_tools(provider=self.provider)
        
        self.memory_agent : MemoryAgent = memory['agent']
        for tool in memory['tools']:
            self.tools[tool.name] = tool
        

    def get_memory_agent(self):
        return self.memory_agent


    def student_prompt(self):
        general = self._build_prompt("student/general", "v1")
        learning_instructions = self._build_prompt("student/learning", "v1")
        retrieval_instructions = self._build_prompt("student/retrieval", "v1")

        full = general.format(retrieval_instructions=retrieval_instructions, learning_instructions=learning_instructions)
        full += "\n"
        full += self._build_prompt("output", "v1")
        return full
    
    def run(self, prompt: str, max_iter: int=10, remove_tools=[]):
        # 1. Decompose into learning and asking
        # 2. Ask -> Knowledge
        # 3. ReAct using knowledge
        # 4. Update learning
        # 5. Summarize for response
        
        super().run(prompt, max_iter=max_iter, remove_tools=remove_tools)
    

    def decompose(self, input):
        prompt = self.get_prompt("decompose")
        prompt += f"Decompose this context: {c(input)}"
        return self.single_run(prompt)