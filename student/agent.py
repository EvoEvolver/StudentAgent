from student.agent_memory import Memory
from student.tools import Tool, AddMemory, ModifyMemory, RecallMemory
from mllm import Chat


class MemoryAgent:
    memory : Memory
    tools : list[Tool]
    system_prompt : str
    chat : Chat

    def __init__(self, tools: list[Tool] = []):
        self.tools = tools
        self.memory = Memory() # adjust later
        self.add_memory_tools()

        self.system_prompt = """
        You are an agent with a dynamic, long-term memory. You can retrieve, store and modify the knowlege with your tools.
        """
        self.reset_chat()

    def add_memory_tools(self):
        self.tools = [AddMemory(self.memory), ModifyMemory(self.memory), RecallMemory(self.memory)]

    def parse_tools(self):
        return [tool.parse() for tool in self.tools]

    def reset_chat(self):
        self.chat = Chat(system_message=self.system_prompt, dedent=True)


    def run(self, prompt: str, parse: str = None, expensive=False, tool_choice: str ="auto"):
        if parse not in ["dict", "list", "obj", "quotes", "colon"] and parse is not None:
            raise ValueError("Invalid parse type")
        
        options = {"tools" : self.parse_tools(), "tool_choice": tool_choice}

        self.chat += prompt
        res = self.chat.complete(parse=parse, cache=True, expensive=expensive, options=options)
        return res

    def use_tool(self, tool: Tool, args=None):
        out = tool.run(args)
        return out
