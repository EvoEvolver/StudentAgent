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
        self.chat_config_memory(cache=cache, expensive=expensive)
        self.active_learning = active_learning

    def get_memory_tools(self):
        memory = memory_agent_tools(provider=self.provider)
        
        self.memory_agent : MemoryAgent = memory['agent']
        for tool in memory['tools']:
            self.tools[tool.name] = tool
        
    
    def run(self, prompt: str, max_iter: int=10, remove_tools=[]):
        # 1. Decompose into learning and asking
        # 2. Ask -> Knowledge
        # 3. ReAct using knowledge
        # 4. Update learning
        # 5. Summarize for response
        remove_tools.extend(self.get_tool_mask())
        return super().run(prompt, max_iter=max_iter, remove_tools=remove_tools)
    
    def chat_config_memory(self, cache=None, expensive=None):
        mem_agent = self.get_memory_agent()
        mem_agent.chat_config(cache=cache, expensive=expensive)

    def get_memory_agent(self):
        if not hasattr(self,"memory_agent"):
            return None
        return self.memory_agent

    def student_prompt(self):
        general = self._build_prompt("student/general", "v2")
        learning_instructions = self._build_prompt("student/learning", "v2")
        retrieval_instructions = self._build_prompt("student/retrieval", "v2")

        full = general.format(retrieval_instructions=retrieval_instructions, learning_instructions=learning_instructions)
        full += "\n"
        full += self._build_prompt("output", "v3")
        return full
    
    def quiz_prompt(self):
        general = self._build_prompt("student/quiz", "v1")
        retrieval_instructions = self._build_prompt("student/retrieval", "v2")

        full = general.format(retrieval_instructions=retrieval_instructions)
        full += "\n"
        full += self._build_prompt("output", "v3")
        return full

    def setup_quiz(self):
        self.active_learning = False
        self.reset_system_prompt(self.quiz_prompt())
        self.reset_chat()


    def get_tool_mask(self, no_memory=False):
        mask = []
        if self.active_learning is False:
            mask.append("learn")
        if no_memory is True:
            mask.append("ask memory")
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
    

    def reset_token_count(self):
        sum_tokens = self.sum_token_count()
        
        student_tokens = super().reset_token_count()
        memory_agent = self.get_memory_agent()
        if memory_agent is not None:
            memory_tokens = memory_agent.reset_token_count()
        return sum_tokens

    def sum_token_count(self):
        student_tokens = self._sum_token_count()
        memory_agent = self.get_memory_agent()
        if memory_agent is None:
            return student_tokens

        else:
            memory_tokens = memory_agent._sum_token_count()
            return {
                "input_tokens": student_tokens["input_tokens"] + memory_tokens["input_tokens"],
                "output_tokens": student_tokens["output_tokens"] + memory_tokens["output_tokens"],
            }