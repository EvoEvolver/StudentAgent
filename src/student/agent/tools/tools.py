import os
from abc import ABC, abstractmethod
import inspect
from typing import get_origin, get_args, List, Dict


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

    def parse(self, name=None) -> Dict:
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
                    param_schema["type"] = "array"
                    if args:
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
                properties[param_name] = param_schema
        
        required = list(properties.keys())
        return {
            "type": "object",
            "properties": {
                "function": {
                    "type": "string",
                    "const": name if name is not None else self.name,
                    "description": self.description,
                },
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                }
            },
            "required": ["function", "parameters"],
            "additionalProperties": False
        }



class RaspaTool(Tool):
    def __init__(self, name, description, path=None, path_add=None):
        super().__init__(name, description)
        self.path = path
        self.path_add = path_add
    
    def get_path(self, full=False):
        if self.path is None:
            #raise RuntimeWarning(f"No path was set for {self.name}.")
            print(f"Warning: No path was set for {self.name}!")
            return "./"
        else:
            if full is True and self.path_add is not None:
                return os.path.join(self.path, self.path_add)
            else:
                return self.path
