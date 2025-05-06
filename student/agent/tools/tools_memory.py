from typing import List, Dict
from .tools import Tool
from ..agent_memory import Memory, MemoryNode


class AddMemory(Tool):

    def __init__(self, memory: Memory):
        name = "add"
        description = """
        Use this tool to generate and store a new memory item. 
        You must provide a list of keywords (called 'stimuli') that describe the context or meaning of the memory, and the content of the memory itself as a string. 
        This tool is useful when you want to save information for future retrieval. 
        For example, if the user provides a fact, insight, or important context, you can use this tool to remember it.    
        """

        super().__init__(name, description)
        self.memory = memory

    def run(self, stimuli: list[str], content: str):
        new_node = MemoryNode(content=content, keys=stimuli)
        self.memory.memory[new_node.id] = new_node
        return self.get_output(new_node)

    def get_output(self, new_node):
        out = f"""
        <tool>Successfully added this to memory:\n\t{new_node.__str__()}\n</tool>
        """
        return out


class RecallMemory(Tool):
    def __init__(self, memory: Memory):
        name = "recall"
        description = """
            Use this tool to search for and retrieve previously stored memory items based on their associated keywords ('stimuli'). 
            You must provide a list of search keywords and a sensitivity value (a float between 0 and 1) that controls how loosely related the results can be. 
            A higher sensitivity retrieves more results even if the match is weaker. 
            The tool returns up to 3 memory items that are most similar to the given stimuli. 
            This is useful when trying to remember related facts or previously stored context relevant to the current conversation or task.
            The output is a dictionary mapping a memory ID (which can be used with the modify tool) to the memory content.
        """
        super().__init__(name, description)
        self.memory = memory

    def run(self, stimuli: list[str], sensitivity: float = 0.01) -> str:
        res = self.memory.recall(stimuli, max_recall=3, sensitivity=sensitivity)
        return self.get_output(stimuli, res)
    
    def get_output(self, stimuli, res: Dict[str, Dict[str, str]]) -> str:
        mem = ""
        for id, i in res.items():
            mem += i
        if mem == "":
            mem = "<no memory found/>"
        out = f"<tool>Recalled from memory with the stimuli \n\t<stimuli>{stimuli}</stimuli>: \n\t{mem}\n</tool>"
        return out


class ModifyMemory(Tool):
    def __init__(self, memory: Memory):
        name = "modify"
        description = """
          Use this tool to update an existing memory item by providing its memory ID.
          You have access to the memory IDs via the output of the recall tool.
          You can change its associated keywords ('stimuli'), its content, or both. 
          This is useful when a memory item needs to be corrected, refined, or re-contextualized.
          Only one or both of the optional parameters — new_stimuli or new_content — need to be provided. 
          If you provide new_content=None and new_stimuli=None, then the memory node is deleted.
        """
        super().__init__(name, description)
        self.memory = memory

    def run(self, id: str, new_stimuli: List[str] = None, new_content: str = None) -> None:
        node : MemoryNode = self.memory.memory.get(id)
        deleted=False

        if node is None:
            return "<tool>No memory found to modify: Incorrect ID!</tool>"
        
        if new_stimuli is not None:
            node.remove_keys(node.keys)
            node.add_keys(new_stimuli)

        if new_content is not None:
            node.content = new_content
        
        if new_content is None and new_stimuli is None:
            del self.memory.memory[id]
            deleted=True

        return self.get_output(node, deleted=deleted)

    def get_output(self, node, deleted=False):
        if deleted:
            out = f"<tool>Memory deleted.\n</tool>"
            return out
        out = f"<tool>Modified memory with the updated memory: \n\t{node.__str__()}\n</tool>"
        return out
