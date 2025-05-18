from mllm import Chat
from .tools_memory import *
from ..agent import Agent
from typing import List


class NewMemoryAgent(Agent):
    def __init__(self, memory):
        super().__init__(memory)
        self.chat = Chat()

    def run(self, new_knowledge:str, thoughts:List[str]):
        '''
        Input:
        - new_knowledge: "Kevin is the cousin of Pete and Carla. He has the same age as Pete."
        - thoughts: [
                    "Thought: I want to add the knowledge about the relation of kevin, pete and cala.",
                    "Thought: The age of pete and kevin is a info.
                    ]
        Returns:
        - out: The added memory items.
        '''

        new_knowledge, thoughts = self.filter(new_knowledge, thoughts)

        content = self.extract_content(new_knowledge, thoughts)
        stimuli = self.extract_stimuli(new_knowledge, thoughts)
        
        return super().run(stimuli=stimuli, content=content)

    def extract_content(self, new_knowledge, thoughts:List[str]):
        pass

    def extract_stimuli(self, new_knowledge, thoughts:List[str]):
        pass