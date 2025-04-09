from student.agent_memory import Memory
from student.tools import Tool, AddMemory, ModifyMemory, RecallMemory
from mllm import Chat
from typing import List, Dict, Union
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
        You are an agent with a dynamic, long-term memory. You can retrieve, add and modify the knowlege with your tools.
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
        if res is None:
            # self.chat.messages[-1]["content"]["text"] = ""
            self.chat.messages.pop()
        
        self.use_tools()
        return res


    def add_message(self, message: Union[Dict[str, str], List[Dict[str, str]]]):
        if isinstance(message, list):
            for m in message:
                self.add_message(m)
            return
        if message is None:
            return
        assert "content" in message and "role" in message, "Message must contain 'content' and 'role'"
        self.chat.messages.append(message)


    def use_tools(self) -> None:
        '''
        Returns a list of messages or an empty list.
        '''
        msg = self.chat.additional_res['full_message']
        calls = msg.tool_calls

        if calls is None:
            return None
        
        for call in calls:
            func = call.function
            name = func.name
            args = json.loads(func.arguments)
            message = self._use_tool(func, name, args, call['id'])
            self.add_message(message)
            
        return None

    
    def _use_tool(self, func, name, args, tool_call_id=""):
        tool = self.tools.get(name)
        if tool is None:
            return None
        out = tool.run(**args)
    
        message = {
            "role": "assistant",
            "tool_call_id": tool_call_id,
            'content': {
                'type': 'text',
                'text': str(out)
            }
        }
        return message
        