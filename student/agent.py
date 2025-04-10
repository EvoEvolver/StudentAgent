from student.agent_memory import Memory
from student.tools import Tool, AddMemory, ModifyMemory, RecallMemory, Thought
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
        You can autonomously retrieve, add and modify the knowlege with your tools.
        Always try to recall relevant memory to give a response. 
        Always analyze the user input if you should add something to your memory or modify a memory you recalled.
        If you have conflicting knowledge that you cannot resolve on your own, you can ask the user for clarifications.
        
        If you opt to use a tool, it will be automatically called from your response and you are reprompted with the output as a new assistant message.
        If you use tools, you are reprompted including the output of the tools.
        """
        #You are required to use the 'think' tool before every other tool call to store a reasoning path.
        # self.system_ptompt += "\n If you have conflicting knowledge that you cannot resolve on your own, you can ask the user for clarifications. Use these to modify the memory.""
        self.system_prompt += "To finish, you need to give a text answer."

        self.reset_chat()


    def add_memory_tools(self):
        add = AddMemory(self.memory)
        modify = ModifyMemory(self.memory)
        recall = RecallMemory(self.memory)
        thought = Thought()
        
        self.tools = {
            add.name : add,
            modify.name : modify,
            recall.name : recall,
            thought.name : thought
        }


    def parse_tools(self):
        return [tool.parse() for tool in self.tools.values()]

    def reset_chat(self):
        self.chat = Chat(system_message=self.system_prompt, dedent=True)


    def run(self, prompt: str, parse: str = None, expensive=False, tool_choice: str ="auto", max_iter: int=10):
        if parse not in ["dict", "list", "obj", "quotes", "colon"] and parse is not None:
            raise ValueError("Invalid parse type")

        options = {"tools" : self.parse_tools(), "tool_choice": tool_choice}
        self.chat += prompt

        for i in range(max_iter):
                        
            res = self.chat.complete(parse=parse, cache=True, expensive=expensive, options=options)
            if res is None:
                self.chat.messages.pop() # since mllm does not include the tool callings into the message
                self.use_tools()         # adds the assistant messages for tool calls and responses
            else:
                break                    # give output to the user when no tool is called

        return res


    def add_message(self, message: Union[Dict[str, str], List[Dict[str, str]]]):
        if isinstance(message, list):
            for m in message:
                self.add_message(m)
            return
        if message is None:
            return
        assert "content" in message and "role" in message, "Message must contain 'content' and 'role'"
        if message['content'] is None:
            calls = message['tool_calls']
            message['content'] = {'type': 'text', 'text' : f'Tool calls: {calls}'}
            del message['tool_calls']
            del message['function_call']
        self.chat.messages.append(message)


    def use_tools(self) -> None:
        '''
        Returns a list of messages or an empty list.
        '''
        msg = self.chat.additional_res['full_message']
        calls = msg.tool_calls
        self.add_message(msg.to_dict())

        if calls is None: # This should never happen, i think.
            raise RuntimeError("Should not end up here.")
            # return None
        
        tool_messages = []
        for call in calls:
            func = call.function
            name = func.name
            args = json.loads(func.arguments)
            message = self._use_tool(func, name, args, call['id'])
            tool_messages.append(message)

        self.add_message(tool_messages)
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
        from IPython.display import HTML
        import html

        chat = self.chat.messages
        html_parts = ["<div style='font-family:Arial, sans-serif; line-height:1.6;'>"]

        for turn in chat:
            role = turn["role"].capitalize()
            raw_message = turn["content"].get("text", "").strip()

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

            # Preprocess message
            if "<tool>" in raw_message or "<thought>" in raw_message:
                # Use <pre> for preserving indentation
                escaped = html.escape(raw_message)
                message = f"<pre style='margin:0;'>{escaped}</pre>"
            else:
                # Use <div> for normal text
                escaped = html.escape(raw_message).replace("\n", "<br>")
                message = f"<div style='margin-top:5px; font-family:monospace;'>{escaped}</div>"

            # Final block
            html_parts.append(
                f"<div style='background:{bg_color}; color:{text_color}; border:{border}; border-radius:{border_radius}; padding:{padding}; margin:{margin};'>"
                f"<strong>{role}{tool_call_note}:</strong><br>{message}</div>"
            )

        html_parts.append("</div>")
        return HTML("".join(html_parts))
