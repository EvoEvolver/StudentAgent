from typing import List, Dict
from .tools import Tool
from ..memory import Memory, MemoryNode
from ..utils import *


class AddMemory(Tool):

    def __init__(self, memory: Memory):
        name = "add"
        description="""
        Store new knowledge in your memory that you can later recall.
        DO NOT use without recalling relevant memory first
        ALWAYS choose keywords as stimuli that facilitate the retrieval
        ALWAYS split knowledge into its building blocks by adding multiple memorys with suitable content/stimuli
        ALWAYS add abstract keywords NO NOT use details as keywords
        ONLY inject the relevant knowledge to the content and NOT simply copy
        NEVER use a memory id as key or content
        """
        description +="ALWAYS highlight words in the content that likely are keywords for other memory entries as xml: <keyword/>"

        super().__init__(name, description)
        self.memory = memory

    def run(self, stimuli: list[str], content: str):
        new_node = MemoryNode(content=content, keys=stimuli)
        self.memory.add(new_node)
        return self.get_output(new_node)

    def get_output(self, new_node):
        out = tool_response(self.name, f"Added:\n\t{new_node.__str__()}\n")
        return out


class RecallMemory(Tool):
    def __init__(self, memory: Memory):
        name = "recall"
        description = """
        Recall knowledge from your memory based on a list of stimuli to use your knowledge.
        ALWAYS choose abstract keywords as stimuli
        ALWAYS add more specific keywords to target more specific knowledge
        AFTER recalling, extract new keywords from the content, especially highlighted as xml: <keyword/>
        """
        #The sensitivity value [0,1] controls the memory search. A smaller value returns less strict matches and is therefore prefered such as 0.1
        old="""
            You must provide a list of search keywords and a sensitivity value (a float between 0 and 1) that controls how loosely related the results can be. 
            A higher sensitivity retrieves more results even if the match is weaker. 
            The tool returns up to 3 memory items that are most similar to the given stimuli. 
            The output is a dictionary mapping a memory ID (which can be used with the modify tool) to the memory content.
        """
        super().__init__(name, description)
        self.memory = memory
        self.sensitivity = 0.3

    def run(self, stimuli: list[str]) -> str:
        res = self.memory.recall(stimuli, max_recall=3, sensitivity=self.sensitivity)
        return self.get_output(stimuli, res)
    
    def get_output(self, stimuli, res: Dict[str, Dict[str, str]]) -> str:
        mem = ""
        for id, i in res.items():
            mem += i
        if mem == "":
            mem = "<no memory found/>"
        out = tool_response(self.name, f"Recalled: \n\t{mem}")
        return out


class ModifyMemory(Tool):
    def __init__(self, memory: Memory):
        name = "modify"
        description = """
        Modify a specific knowledge entry from the memory to correct, update or refine knowledge.
        DO NOT use without recalling relevant memory first
        ALWAYS use a memory id from a recalled memory entry
        update one or both of the memory stimuli and content.
        to delete a memory entry, choose new_content=None and new_stimuli=None
        """
        super().__init__(name, description)
        self.memory = memory

    def run(self, id: str, new_stimuli: List[str] = None, new_content: str = None) -> None:
        node, deleted = self.memory.modify(id, new_stimuli, new_content)
        return self.get_output(node, deleted=deleted)

    def get_output(self, node, deleted=False):
        if node is None:
            return error("No memory found to modify: Incorrect ID")
        if deleted:
            out = tool_response(self.name, "Memory deleted.\n")
            return out
        out = tool_response(self.name, f"Modified entry: \n\t{node.__str__()}\n")
        return out
