from abc import ABC, abstractmethod
from mllm import Chat
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
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        return tool_data


class AddMemory(Tool):

    def __init__(self, memory: Memory):
        name = "add"
        description = "This tool can be used to add a new memory item."
        super().__init__(name, description)
        self.memory = memory

    def run(self, stimuli: list[str], content: str):
        # self.memory.add_memory(stimuli, content)
        print("Add Memory: ", stimuli, content)



class RecallMemory(Tool):
    def __init__(self, memory: Memory):
        name = "recall"
        description = """
        This tool can be used to recall memory items related to a list of keys. 
        It needs the following parameters: 
            keys : list[str]  
            sensitivity : float # range [0, 1]
        """
        super().__init__(name, description)
        self.memory = memory

    def run(self):
        # TODO
        print("Recall memory")



class ModifyMemory(Tool):
    def __init__(self, memory: Memory):
        name = "modify"
        description = "This tool can be used to modify memory items. It needs the following parameters: ..."
        super().__init__(name, description)
        self.memory = memory

    def run(self):
        # TODO
        print("Modify memory")