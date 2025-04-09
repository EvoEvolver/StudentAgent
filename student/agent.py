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
        You are an agent with a dynamic, long-term memory. 
        You can retrieve, add and modify the knowlege with your tools. 
        The tools will be automatically called and you will see the output as an assistant message to use it.
        """
        # self.system_ptompt += "\n If you have conflicting knowledge that you cannot resolve on your own, you can ask the user for clarifications. Use these to modify the memory.""
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
    

    def render_chat_html(self):
        chat = self.chat.messages
        
        from IPython.display import HTML
        import html

        html_parts = ["<div style='font-family:Arial, sans-serif; line-height:1.6;'>"]

        for turn in chat:
            role = turn["role"].capitalize()
            raw_message = turn["content"].get("text", "").strip()

            # Escape and format message
            message = html.escape(raw_message).replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")

            # If it's a tool response and starts with <tool>, prettify it more clearly
            if "<tool>" in raw_message:
                # Simple pretty formatting for visual indentation
                message = raw_message.strip().replace("        ", "    ")
                message = html.escape(message).replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")

            # Tool Call ID label
            tool_call_note = ""
            if "tool_call_id" in turn:
                tool_id = turn["tool_call_id"]
                tool_call_note = f" <span style='color:#666; font-size:0.85em;'>(Tool Call ID: <code>{tool_id}</code>)</span>"

            # Style
            bg_color = "#e0f7fa" if role == "Assistant" else "#f8f9fa"
            text_color = "#111"
            border = "1px solid #ccc"
            border_radius = "10px"
            padding = "10px"
            margin = "10px 0"

            html_parts.append(f"""
            <div style="background:{bg_color}; color:{text_color}; border:{border}; border-radius:{border_radius}; padding:{padding}; margin:{margin};">
                <strong>{role}{tool_call_note}:</strong><br>
                <div style="margin-top:5px; font-family:monospace;">{message}</div>
            </div>
            """)

        html_parts.append("</div>")
        return HTML("".join(html_parts))
