from abc import ABC, abstractmethod
from mllm.chat import Chat
import inspect
from typing import get_origin, get_args, List, Dict
from student.agent_memory import Memory, MemoryNode


class Tool(ABC):
    name : str
    description : str

    def __init__(self, name, description):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self):
        pass

    def get_output(self):
        return self.name

    def parse(self) -> Dict:
        func = self.run

        sig = inspect.signature(func)
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue 

            param_schema = {}
            default_json_type = "string"
            item_schema = None

            if param.annotation is not inspect.Parameter.empty:
                origin = get_origin(param.annotation)
                args = get_args(param.annotation)
                if origin in (list, List):
                    # Parameter is annotated as a list type
                    param_schema["type"] = "array"
                    if args:
                        # Map the inner type of the list to a JSON Schema type
                        inner_type = args[0]
                        if inner_type == int:
                            item_type = "integer"
                        elif inner_type == float:
                            item_type = "number"
                        elif inner_type == bool:
                            item_type = "boolean"
                        elif inner_type == str:
                            item_type = "string"
                        else:
                            item_type = default_json_type
                        item_schema = {"type": item_type}
                    else:
                        item_schema = {}
                elif param.annotation == int:
                    param_schema["type"] = "integer"
                elif param.annotation == float:
                    param_schema["type"] = "number"
                elif param.annotation == bool:
                    param_schema["type"] = "boolean"
                elif param.annotation == str:
                    param_schema["type"] = "string"
                else:
                    param_schema["type"] = default_json_type
            else:
                param_schema["type"] = default_json_type

            if param_schema.get("type") == "array" and item_schema is not None:
                param_schema["items"] = item_schema

            param_schema["description"] = f"Parameter {param_name}"
            properties[param_name] = param_schema

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        tool_data = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                #"strict": True,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        return tool_data


class Thought(Tool):
    def __init__(self):
        name = "think"
        description = """
        Use this tool to generate a thought progress.
        Use this tool to store a 'thought' as text which you can access later to follow the reasoning steps.
        For example, if you want to use another tool, explain why you did this.
        """
        super().__init__(name, description)


    def run(self, thought_content: str):
        return self.get_output(thought_content)

    def get_output(self, out):
        return f"""
        <thought>\n\t{out}\n</thought>
        """



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
        for i in res:
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
          If you provide new_content="" and new_stimuli=[], then the memory node is deleted.
        """
        super().__init__(name, description)
        self.memory = memory

    def run(self, id: str, new_stimuli: str = None, new_content: str = None) -> None:
        node = self.memory.memory.get(id)
        if node is None:
            return "<tool>No memory found to modify: Incorrect ID!</tool>"
        
        if new_stimuli is not None:
            node.remove_keys(node.keys)
            node.add_keys(new_stimuli)

        if new_content is not None:
            node.content = new_content
        
        if new_content == "" and new_stimuli == []:
            del self.memory.memory['id']

        return self.get_output(node)

    def get_output(self, node):
        out = f"<tool>Modified memory with the updated memory: \n\t{node.__str__()}\n</tool>"
        return out