from typing import List, Dict
from .tools import Tool
from ..memory import Memory, MemoryNode
from ..utils import *
from mllm import Chat

class AddMemory(Tool):

    def __init__(self, memory: Memory):
        name = "add"
        description="""
        Store new knowledge in your memory that you can later recall.
        DO NOT use without recalling relevant memory first
        ALWAYS choose keywords as stimuli that facilitate the retrieval
        AVOID long content!
        ALWAYS split knowledge into its building blocks by adding multiple memorys with suitable content/stimuli
        ALWAYS add abstract keywords NO NOT use details as keywords
        ONLY inject the relevant knowledge to the content and NOT simply copy
        NEVER use a memory id as key or content
        """
        description +="ALWAYS highlight words in the content that likely are keywords for other memory entries as xml: <keyword/>"

        super().__init__(name, description)
        self.memory = memory

    def _run(self, stimuli: list[str], content: str):
        new_node = MemoryNode(content=content, keys=stimuli)
        self.memory.add(new_node)
        return new_node
    
    def run(self, stimuli: list[str], content: str):
        if len(stimuli) == 0:
            return self.get_output(None)
        new_node = self._run(stimuli, content)
        return self.get_output(new_node)

    def get_output(self, new_node):
        if new_node is None:
            return error("Error during creation of new memory node. Maybe you forgot to include stimuli!")
        out = tool_response(self.name, f"Added:\n\t{new_node.__str__()}\n")
        return out

'''
class ExtendedAddMemory(AddMemory):
    def __init__(self, memory:Memory, chat):
        super().__init__(memory)
        self.chat = chat

    def run(self, stimuli, new_content) -> None:
        
        new_node = super()._run(stimuli, new_content)

        return self.get_output()
'''

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
        self.sensitivity = 0.2

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

class ExtendedModifyMemory(ModifyMemory):
    def __init__(self, memory:Memory, chat):
        super().__init__(memory)
        self.chat = chat
        self.description = """
        Modify a specific memory entry to correct, update or refine knowledge.
        Input the memory ID you want to modify and the new_information.
        ALWAYS use a memory id from a recalled memory entry
        """

    def run(self, id: str, new_information) -> None:
        node = self.memory.get_node(id)
        if node is None:
            return self.get_output(None)
        
        old_stimuli = node.keys
        old_content = node.content

        extract_content = f"Based on this new information: {new_information}. \n Update this old information: {old_content}. YOU MUST ONLY output a the new information: 'new information'"
        new_content = self.chat(extract_content)

        extract_stimuli = f'Based on this new information: {new_information}. \n Update these old keywords by adding new ones or removing old ones: {old_stimuli}. YOU MUST ONLY output a list of the all keywords including the old keys you want to keep: ["key", "key"] (DO NOT leave the keys empty!)'
        new_stimuli = self.chat(extract_stimuli, parse="list")

        return super().run(id, new_stimuli, new_content)