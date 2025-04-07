from student.agent_memory import Memory
from student.tools import Tool, AddMemory, ModifyMemory, RecallMemory
from mllm import Chat
from typing import List, Dict
import json


class MemoryAgent:
    memory : Memory
    tools : Dict[str, Tool]
    system_prompt : str
    chat : Chat

    def __init__(self, tools: Dict[str, Tool] = {}):
        self.tools = tools
        self.memory = Memory() # adjust later
        self.add_memory_tools()

        self.system_prompt = """
        You are an agent with a dynamic, long-term memory. You can retrieve, store and modify the knowlege with your tools.
        """
        self.reset_chat()

    def add_memory_tools(self):
        add = AddMemory(self.memory)
        modify = ModifyMemory(self.memory)
        recall = RecallMemory(self.memory)

        self.tools = {
            add.name : add,
            modify.name : modify,
            recall.name : recall
        }

    def parse_tools(self):
        return [tool.parse() for tool in self.tools.values()]

    def reset_chat(self):
        self.chat = Chat(system_message=self.system_prompt, dedent=True)


    def run(self, prompt: str, parse: str = None, expensive=False, tool_choice: str ="auto"):
        if parse not in ["dict", "list", "obj", "quotes", "colon"] and parse is not None:
            raise ValueError("Invalid parse type")
        
        options = {"tools" : self.parse_tools(), "tool_choice": tool_choice}

        self.chat += prompt
        res = self.chat.complete(parse=parse, cache=True, expensive=expensive, options=options)
        self.use_tools()
        return res

    def use_tools(self):
        calls = self.chat.additional_res['full_message'].tool_calls
        outputs = []
        for call in calls:
            func = call.function
            name = func.name
            args = json.loads(func.arguments)
        
            tool = self.tools.get(name)
            if tool is None:
                continue
            outputs.append(tool.run(**args))
        
        return outputs
            