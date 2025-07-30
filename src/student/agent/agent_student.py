from .agent import Agent
from .agent_memory import MemoryAgent
from .agent_memory import memory_agent_tools

from typing import List, Dict
from .tools.tools import Tool
from .utils import *
from .utils import context as c


class StudentAgent(Agent):
    def __init__(self, tools: Dict[str, Tool] = {}, cache=None, expensive=None, version=None, provider="openai", verbose=False, active_learning=True):
        super().__init__(tools=tools, cache=cache, expensive=expensive, version=version, provider=provider, verbose=verbose)

        self.reset_system_prompt(self.student_prompt())
        self.get_memory_tools()
        self.active_learning = active_learning

    def get_memory_tools(self):
        memory = memory_agent_tools(provider=self.provider)
        
        self.memory_agent : MemoryAgent = memory['agent']
        for tool in memory['tools']:
            self.tools[tool.name] = tool
        

    def get_memory_agent(self):
        return self.memory_agent

    def student_prompt(self):
        general = self._build_prompt("student/general", "v2")
        learning_instructions = self._build_prompt("student/learning", "v2")
        retrieval_instructions = self._build_prompt("student/retrieval", "v2")

        full = general.format(retrieval_instructions=retrieval_instructions, learning_instructions=learning_instructions)
        full += "\n"
        full += self._build_prompt("output", "v3")
        return full
    
    def run(self, prompt: str, max_iter: int=10, remove_tools=[]):
        # 1. Decompose into learning and asking
        # 2. Ask -> Knowledge
        # 3. ReAct using knowledge
        # 4. Update learning
        # 5. Summarize for response
        remove_tools.append(self.get_tool_mask())
        return super().run(prompt, max_iter=max_iter, remove_tools=remove_tools)
    

    def get_tool_mask(self):
        mask = []
        if self.active_learning is False:
            mask.append("learn")
        return mask

    def decompose(self, input):
        prompt = self.get_prompt("decompose")
        prompt += f"Decompose this context: {c(input)}"
        return self.single_run(prompt)
    
    def save(self, folder_name):
        os.makedirs(folder_name, exist_ok=True)
        
        super().save(folder_name)
        self.get_memory_agent().save_memory(os.path.join(folder_name,"memory.parquet"))
        self.get_memory_agent().save_conversation(os.path.join(folder_name,"conversation_memory.txt"))

    def load(self, folder_name):
        if len(self.conversation) == 0:
            return
        super().load(folder_name)
        
        mem_path = os.path.join(folder_name,"memory.parquet")
        mem_conv_path = os.path.join(folder_name,"conversation_memory.txt")
        if os.path.exists(mem_path) and os.path.exists(mem_conv_path):
            try:
                self.get_memory_agent().load_memory(mem_path)
                self.get_memory_agent().load_conversation(mem_conv_path)
            except Exception as e:
                return e
        

    def render_memory(self):
        return self.get_memory_agent().render_memory()
    
    def memory_size(self):
        return self.get_memory_agent().memory_size()
    