from .memory import Memory
from .tools.tools import Tool
from .tools.tools_memory import AddMemory, ModifyMemory, RecallMemory, ExtendedModifyMemory
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

    def __init__(self, tools: Dict[str, Tool] = {}, cache=None, expensive=None, version="v1", provider="openai"):
        super().__init__(tools=tools, cache=cache, expensive=expensive, version=version, provider=provider)
        
        self.memory = Memory()
        self.add_memory_tools()
        
        prompt = self.get_prompt(type="general", version=version)
        self.reset_system_prompt(prompt, append=True)

        #self.special_keywords = {"explicit knowledge" : keyword("explicit knowledge"),}

    ####### Setup #######

    def add_memory_tools(self):
        add = AddMemory(self.memory)
        modify = ExtendedModifyMemory(self.memory, self.single_run)# ModifyMemory(self.memory)
        recall = RecallMemory(self.memory)
        
        self.tools[add.name] = add
        self.tools[modify.name] = modify
        self.tools[recall.name] = recall

    def get_memory_tool_mask(self):
        return [name for name in self.tools.keys() if name not in ["add", "recall", "modify"]]
    
    def get_learning_mask(self):
        return ["recall"]

    def get_question_mask(self):
        return ["add", "modify"]
    
    
    def get_all_mask(self):
        return [name for name in self.tools.keys()]


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
        self.set_prompt(type="retrieval", version="v3")
        
        recall = []
        extracted_keys = []
        rev_summaries = self.full_summary(question)     # decompose into abstraction level

        for summary in rev_summaries:                   # iterate by summarizing
            keys = self.extract_keys(summary)           # ask for context
            extracted_keys.append(keys)                 # store extracted keys
        
            input = f"Retrieve all knowledge related to this input: {q(question)}"    
            input += f"Use these or similar keys as stimuli: {keys}"

            recall.append(self.run(input, remove_tools=self.get_question_mask())) # TODO: see only last level
            # TODO: aggregate each time
            
        response = self.filter_information(q(question), recalled(';\n'.join(recall)))

        return response


    def learn(self, context:str):

        recall = []
        extracted_keys = []
        updates = []

        rev_summaries = self.full_summary(context)     # decompose into abstraction level
        
        for summary in rev_summaries:                   # iterate by summarizing
            self.set_prompt(type="learning", version="v4")
            
            # ask
            keys = self.extract_keys(summary)           # ask for context
            extracted_keys.append(keys)                 # store extracted keys

            input = f"Retrieve all knowledge related to this input: {q(context)}"    
            input += f"Use these or similar keys as stimuli: {keys}"
            
            prompt = self.get_prompt(type="retrieval", version="v3", json=False, general=False)
            prompt += input
            mem = self.run(prompt, remove_tools=self.get_question_mask())
            recall.append(mem)

            # learn
            prompt = self.get_prompt(type="update_mem", version="v1", json=False, general=False)
            prompt = prompt.format(new_information = summary, recalled=mem)
            update = self.run(prompt)
            updates.append(update)

            # TODO: refactor quality control tool for key selection - automatic recall of used keywords
            #     ALWAYS try to recall memory after adding to evaluate if the keys need to be modified.

        answer = self.learning_answer(updates, context)
        return answer

    def learning_answer(self, updates, new_information):
        prompt = self.get_prompt("learning_answer", "v1", json=False, general=False)
        prompt = prompt.format(updates=updates, new_information=new_information)
        return self.single_run(prompt)

    def filter_information(self, question, information):
        prompt = self.get_prompt("filter", "v2", json=False, general=False)
        prompt = prompt.format(question=question, knowledge=information)
        return self.single_run(prompt)

    def summarize(self, context):
        prompt = self.get_prompt("summarize", "v3", json=False, general=False)
        prompt = prompt.format(context=context)
        return self.single_run(prompt)

    def decompose(self, context):
        prompt = self.get_prompt("decompose", "v2", json=False)
        prompt = prompt.format(context=context)
        return self.single_run(prompt, parse="list")

    def extract_keys(self, context):
        keywords = self.memory.get_keywords(topk=50)

        prompt = self.get_prompt("extract_keys", "v2", json=False)
        prompt = prompt.format(context=context, keywords=keywords.__str__()[1:-1])
        return self.single_run(prompt)

    def full_summary(self, context, max_runs=5):
        summaries = [context]
        for i in range(max_runs):
            res = self.summarize(summaries[-1])
            if res == "<nothing/>":
                break
            summaries.append(res)
        summaries.reverse()
        return summaries


    ####### Prompts #######

    def get_prompt(self, type, version, json=True, general=True):
        return super().get_prompt(type, dir="memory", version=version, version_general="v3", version_output="v2", json=json, general=general)
    

    def set_prompt(self, prompt=None, type=None, version="v1"):
        
        if prompt is None:
            try:
                prompt = self.get_prompt(type, version=version)
            except Exception as e:
                raise e
        
        self.reset_system_prompt(prompt)

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