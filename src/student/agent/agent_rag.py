import os
from typing import List, Sequence, Dict

import numpy as np
import pandas as pd

from .agent import Agent
from .agent_memory import MemoryAgent
from .memory_rag import MemoryNodeRAG, MemoryRAG
from .memory import Memory
from .agent_memory import Ask

from .tools.tools_memory import RecallMemory
from .utils import question as q

class RAGRecallMemory(RecallMemory):
    def __init__(self, memory: MemoryRAG):
        super().__init__(memory)
        self.name = "recall"
        self.description = """
        Recall knowledge from your memory.
        ALWAYS choose a short description of the question as query.
        The retrieval is based on the semantic similarity of the query to the memory entries.
        """
    
    def run(self, query: str) -> Dict[str, str]:
        return super().run(query)


def memory_agent_tools(provider, memory_path):
    agent = AgenticRAGAgent(provider=provider, memory_path=memory_path)
    return {'agent': agent, 'tools' : [Ask(agent)]}


class AgenticRAGAgent(MemoryAgent):
    '''
    Memory Agent that allows agentic RAG for memory recall and question answering.
    '''
    
    def __init__(self, memory:Memory=None, memory_path=None, tools: dict = {}, cache=None, expensive=None, provider="openai"):

        self.memory_path = memory_path
        super().__init__(tools=tools, cache=cache, expensive=expensive, version="v1", provider=provider)

        self.add_memory(memory)

    def setup_general_prompt(self, version):
        prompt = self.get_prompt(type="general_rag", version=version)
        self.reset_system_prompt(prompt, append=True)

    def add_memory_tools(self):
        self.memory = MemoryRAG()
        recall = RAGRecallMemory(self.memory)
        self.tools[recall.name] = recall
        
        self.load_memory(self.memory_path)
    
        
    def load_memory(self, file=None):
        if file is None:
            self.add_memory(MemoryRAG())
            #print("Initializing empty RAG memory")
            return
        else:
            self.memory = Memory()
            super().load_memory(file)
            
            mem_rag = MemoryRAG()
            mem_rag.load_from_memory(self.memory)
            self.add_memory(mem_rag)
            #print("Loading memory")
    
    
    def add_memory(self,memory:Memory):
        if memory is None:
            return
        if type(memory) == MemoryRAG:
            self.memory = memory
        else: 
            self.memory = MemoryRAG()
            for id, node in memory.memory.items():
                if len(node.content) > 0:
                    new_node = MemoryNodeRAG(input=node.content)
                    new_node.id = id
                    self.memory.add(new_node)

        for tool in self.tools.values():
            if hasattr(tool, 'memory'):
                tool.memory = self.memory


    def ask(self, question: str) -> str:
        self.set_prompt(type="retrieval_rag", version="v1")
        
        prompt = f"Retrieve all knowledge related to this input: {q(question)}"
        res = self.run(prompt)
        return res
    
class NaiveRAGAgent(AgenticRAGAgent):
    '''
    Naive RAG Agent
    '''
    def setup_general_prompt(self, version):
        prompt = self.get_prompt(type="naive_rag", version=version)
        self.reset_system_prompt(prompt, append=True)

    def add_memory_tools(self):
        self.memory = MemoryRAG()
        self.load_memory(self.memory_path)


    def _retrieve(self, query, sensitivity=0.1):
        res = self.memory.recall(query, sensitivity=sensitivity)
        mem = ""
        for id, i in res.items():
            mem += i
        if mem == "":
            mem = "<no memory found/>"
        return mem
    

    def run(self, prompt: str, max_iter: int=15):
        recalled = self._retrieve(prompt)
        
        rag_prompt = f"""<query>{prompt}</query>
        <context>
        {recalled}
        </context>
        <answer/>
        """
        res = super().run(rag_prompt, max_iter=max_iter)
        return res


    def get_output_jsonschema(self, remove_tools=[]):

        schema = {
        "type": "object",
        "properties": {
            "react": {
                "type": "array",
                "description": "A sequence of reasoning steps as discrete thoughts",
                "items": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "A reasoning step or internal reflection."
                        }
                    },
                    "required": ["thought"],
                    "additionalProperties": False
                },
            },
            "response": {
                "type": "string",
                "description": "Final response to the user. IGNORED IF a function is included in the react scheme"
            }
        },
        "required": ["react", "response"],
        "additionalProperties": False
        }
        return schema