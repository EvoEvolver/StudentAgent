from typing import Dict, List

from mllm import Chat

from ..memory import Memory, MemoryNode
from ..utils import *
from .tools import Tool


class AddMemory(Tool):

    def __init__(self, memory: Memory):
        name = "add"
        description = """
        Store new knowledge in your memory that you can later recall.
        IMPORTANT: DO NOT use without recalling relevant memory first.

        <detailed instructions>
        Split knowledge into its building blocks by adding multiple memory entries
        NEVER use a memory id as key or content
            <stimuli>
            ALWAYS choose stimuli that allow robust retrieval.
            Stimuli can be one or multiple words. You can retrieve entries with less keys easier.
            The stimuli should be associated with the content of the entry.
            ALWAYS add abstract keywords to the stimuli.
            </stimuli>
            <content>
            ALWAYS add  explainations and reflections to the information. DO NOT ONLY COPY
            If details are essential, you should copy the information AND add explainations or comments.
            You can link entries by adding stimuli of associated memories as xml: "<key>replace this with the stimuli</key>".
            and links to other memory entries here!
            </content>
        </detailed instructions>
        """

        super().__init__(name, description)
        self.memory = memory

    def _run(self, stimuli: list[str], content: str):
        new_node = MemoryNode(content=content, keys=stimuli)
        self.memory.add(new_node)
        return new_node

    def run(self, stimuli: list[str], content: str):
        if len(stimuli) == 0:
            return self.get_output(
                e="Error during creation of new memory node. Maybe you forgot to include stimuli!"
            )
        new_node = self._run(stimuli, content)
        return self.get_output(f"Added:\n\t{new_node.__str__()}\n")


class RecallMemory(Tool):
    def __init__(self, memory: Memory):
        name = "recall"
        description = """
        Recall knowledge from your memory based on a list of stimuli to use your knowledge.
        ALWAYS choose abstract keywords as stimuli
        ALWAYS add more specific keywords to target more specific knowledge
        AFTER recalling, extract new keywords from the content, especially highlighted as xml: <keyword/>
        """
        # The sensitivity value [0,1] controls the memory search. A smaller value returns less strict matches and is therefore prefered such as 0.1
        old = """
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
        mem = ""
        for id, i in res.items():
            mem += i
        if mem == "":
            mem = "<no memory found/>"
        return self.get_output(f"Recalled: \n\t{mem}")


class ModifyMemory(Tool):
    def __init__(self, memory: Memory):
        name = "modify"
        description = """
        Modify a specific memoy entry from the memory to correct, update or refine knowledge.        
        IMPORTANT: DO NOT use without recalling relevant memory first.

        <detailed instructions>
        Update one or both of the memory stimuli and content.
        Delete a memory entry by choosing: new_content=None and new_stimuli=None

        Split knowledge into its building blocks by adding multiple memory entries
        NEVER use a memory id as key or content
            <id>ALWAYS use a memory id from a recalled memory entry. If the id is invalid, nothing happens.</id>
            <delete>If true, the memory entry is completely deleted! Avoid to loose information!</delete>
            <new_stimuli>
                If None, nothing happens!
                Else the old stimuli will be removed and replaced with the new ones!
                ALWAYS choose stimuli that allow robust retrieval. 
                Stimuli can be one or multiple words. You can retrieve entries with less keys easier.
                The stimuli should be associated with the content of the entry.
                ALWAYS add abstract keywords to the stimuli.
            </new_stimuli>
            <new_content>
                If None, nothing happens!
                Else the old content will be erased and replacd with the new one!
                ALWAYS add  explainations and reflections to the information. DO NOT ONLY COPY
                If details are essential, you should copy the information AND add explainations or comments.
                You can link entries by adding stimuli of associated memories as xml: "<key>replace this with the stimuli</key>".
                and links to other memory entries here!
            </new_content>
        </detailed instructions>
        """
        super().__init__(name, description)
        self.memory = memory

    def run(
        self,
        id: str,
        new_stimuli: List[str] = None,
        new_content: str = None,
        delete=False,
    ) -> None:
        if delete is True:
            node = self.memory.delete_node(id)
            return self.get_output("Memory deleted.")

        node, deleted = self.memory.modify(id, new_stimuli, new_content)
        if node is None:
            return self.get_output(e="No memory found to modify: Incorrect ID")

        return self.get_output(f"Modified entry: \n\t{node.__str__()}")


class ExtendedModifyMemory(ModifyMemory):
    def __init__(self, memory: Memory, chat):
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
            return self.get_output(e="No memory found to modify: Incorrect ID")

        old_stimuli = node.keys
        old_content = node.content

        extract_content = f"Based on this new information: {new_information}. \n Update this old information: {old_content}. YOU MUST ONLY output a the new information: 'new information'"
        new_content = self.chat(extract_content)

        extract_stimuli = f'Based on this new information: {new_information}. \n Update these old keywords by adding new ones or removing old ones: {old_stimuli}. YOU MUST ONLY output a list of the all keywords including the old keys you want to keep: ["key", "key"] (DO NOT leave the keys empty!)'
        new_stimuli = self.chat(extract_stimuli, parse="list")

        return super().run(id, new_stimuli, new_content)
