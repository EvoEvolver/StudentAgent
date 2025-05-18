from .memory import Memory
from .tools.tools import Tool
from .tools.tools_memory import AddMemory, ModifyMemory, RecallMemory
from .agent import Agent

from mllm import Chat
from typing import List, Dict, Union
import json
from .utils import *
from .utils import question as q
from .utils import context as c


def memory_agent_tools(provider):
    agent = MemoryAgent(provider=provider)
    return {'agent': agent, 'tools' : [Ask(agent), Learn(agent)]}
    

class Ask(Tool):
    def __init__(self, agent:Agent):
        name="ask"
        description="Use to retrive related knowledge to a question from your memory."
        super().__init__(name, description)
        self.agent = agent

    def run(self, question:str):
        res = self.agent.ask(question)
        return tool_response(self.name, res)


class Learn(Tool):
    def __init__(self, agent:Agent):
        name="learn"
        description="Use to learn the knowledge from a given context."
        super().__init__(name, description)
        self.agent = agent
    
    def run(self, context:str):
        res = self.agent.learn(context)
        return tool_response(self.name, res)




class MemoryAgent(Agent):
    memory : Memory

    def __init__(self, tools: Dict[str, Tool] = {}, cache=None, expensive=None, version="v1.xml", provider="openai"):
        super().__init__(tools=tools, cache=cache, expensive=expensive, version=version, provider=provider)
        
        self.memory = Memory()
        self.add_memory_tools()
        self.add_system_prompt(dir="memory/general", version=version)

        #self.special_keywords = {"explicit knowledge" : keyword("explicit knowledge"),}

    ####### Setup #######

    def add_memory_tools(self):
        add = AddMemory(self.memory)
        modify = ModifyMemory(self.memory)
        recall = RecallMemory(self.memory)
        
        self.tools[add.name] = add
        self.tools[modify.name] = modify
        self.tools[recall.name] = recall

    def get_memory_tool_mask(self):
        return [name for name in self.tools.keys() if name not in ["add", "recall", "modify"]]
    
    def get_learning_mask(self):
        return ["add", "modify"]

    def get_question_mask(self):
        return ["recall"]

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
    
    ####### Calls #######

    def ask(self, question:str):
        # self.reset_chat()
        question = q(question)
        self.set_prompt(type="retrieval", version="v1")
        return self.run(question)


    def learn(self, context:str):
        # self.reset_chat()

        # TODO:
        #   decompose into abstraction levels
        #   iterate by summarizing:
        #       ask for context
        #       store (summary, extracted keys)
        #
        #   iterate inversely (abstract -> detail):
        #       run learn ()


        # ask for context
        question = f"What knowledge is in my memory related to this context: {c(context)}"
        recall = self.ask(question)

        new = context # TODO: summarize at current level of abstraction <=> input last summary
        prompt = f" <new_knowledge>{new}</new_knowledge><knowledge>I have recalled this related knowledge from my memory: {recall}</knowledge>"

        # learn
        self.set_prompt(type="learning", version="v1")
        out = self.run(prompt)
        # TODO: add quality control tool for key selection - automatic recall of used keywords
        
        # TODO: after loop, summarize, reflect, ask for clarification, response
        return out


    def summarize(self, context):
        prompt = self.get_prompt("summarize", "v1")
        prompt += f"Summarize this context: {c(context)}"
        return self.single_run(prompt)

    def decompose(self, context):
        prompt = self.get_prompt("decompose", "v1")
        prompt += f"Decompose this context: {c(context)}"
        return self.single_run(prompt)

    def extract_keys(self, context):
        # TODO: add list of all current keywords
        prompt = self.get_prompt("extract_keys", "v1")
        prompt += f"Extract keys from this context: {c(context)}"

    ####### Prompts #######

    def get_prompt(self, type, version, json=True):
        super().get_prompt(type, dir="memory", version=version, version_general="v3", version_output="v1", json=json)
    

    def set_prompt(self, prompt=None, type=None, version="v1"):
        
        if prompt is not None:
            try:
                prompt = self.get_prompt(type, version=version)
            except Exception as e:
                raise e
        self.system_prompt = prompt

    ######### Potentially interesting additional features #########

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