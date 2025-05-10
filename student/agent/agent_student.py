from .agent_memory import Memory
from .tools.tools import Tool
from .tools.tools_memory import AddMemory, ModifyMemory, RecallMemory

from mllm import Chat
from typing import List, Dict, Union
import json
from .utils import *
import html

class StudentAgent:
    memory : Memory
    tools : Dict[str, Tool]
    system_prompt : str
    chat : Chat
    id : int

    def __init__(self, tools: Dict[str, Tool] = {}, cache=None, expensive=None):
        self.tools = tools
        self.memory = Memory()
        self.add_memory_tools()
        self.id = 0

        self.system_prompt = """
        You are an agent with a dynamic, long-term memory. 
        You can autonomously retrieve, add and modify the knowlege with your tools.
        When using tools, leave <response/> empty. You will be automaitcally reprompted with the output of the tools and you can think or use tools again. 
        If you did not use any tool, put your response in the <response/> field to the user.
        You must follow these guidelines:
        <tool use>
        - Always try to recall relevant memory before adding or modifying memory or answering a question by the user. 
        - Always analyze the user input for knowledge that you can add to your memory or use to modify the memory you recalled.
        - If you have any conflicting knowledge in your memory that you cannot resolve on your own, you can ask the user for clarifications.
        </tool use>
        <recall>
        - To recall general memory, choose few abstract keywords related to the information you are looking for.
        - To recall detailed knowledge, use several less abstract keywords to reach more specific knowledge.
        - You can extract new keywords from recalled knowledge by looking for xml elements containing keywords.
        </recall>
        <add>
        - To add memory general knowledge, choose abstract keywords as stimuli and try to only deposite abstract knowledge into the memory content.
        - In the memory content you shoud highlight all major keywords that may be associated with other memory entries as empty xml elements with the keyword as element name: <keyword/>.
        - To add detailed knowledge (like examples), you must select both abstract and specific keywords.
        - If you have new knowledge you want to put into your memory, prefer <add/> over <modify/>.
        - After adding memory, you get a response that reflects if you successfully added the memory.
        </add>
        <modify>
        - You can modify memory based on the memory id you obtain in the response to <recall/>.
        - To update knowledge of an existing memory entry, modify the content such that the old content is updated and no relevant knowledge is lost.
        - If you have new information that requires to update existing memory, prefer <modify/> over <add/>.
        - To change the stimuli of a memory entry, follow the guidelines for <add/>.
        - To delete a memory entry, provide None as stimuli and new content.
        </modify>
        """
        self.chat_config(cache, expensive)
        self.reset_chat()
        self.special_keywords = {
            "explicit knowledge" : keyword("explicit knowledge"),
        }

    def chat_config(self, cache=None, expensive=None):
        self.cache = cache if cache is not None else True
        self.expensive = expensive if expensive is not None else True


    def add_memory_tools(self):
        add = AddMemory(self.memory)
        modify = ModifyMemory(self.memory)
        recall = RecallMemory(self.memory)
        
        self.tools[add.name] = add
        self.tools[modify.name] = modify
        self.tools[recall.name] = recall
    
    
    def reset_chat(self):
        self.chat = Chat(system_message=self.system_prompt, dedent=False)
        self.id = 0


    def run(self, prompt: str, max_iter: int=10, schema:str=None, remove_tools:List[str]=[]):
        if schema is None:
            schema = self.get_output_jsonschema(remove_tools=remove_tools)
        options = {"response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                        "schema": schema,
                        "strict": True
                    },
            }
        }
        self.chat += prompt

        for i in range(max_iter):
            response, done = self._run(options)     
            if done:
                break            
        return self.response(response)
    
    def _run(self, options):
        res = self.chat.complete(parse=None, cache=self.cache, expensive=self.expensive, options=options)
        res = json.loads(res)
        done = self.use_tools(res)    
        return res, done
    
    def response(self, message: Dict):
        response = message.get("response", '')
        return response


    def add_message(self, message: Union[Dict[str, str], List[Dict[str, str]]]):
        if isinstance(message, list):
            for m in message:
                self.add_message(m)
            return
        if message is None:
            return
        assert "content" in message and "role" in message, "Message must contain 'content' and 'role'"
        self.chat.messages.append(message)

    def get_next_id(self):
        self.id += 1
        return self.id


    def use_tools(self, message: Dict) -> bool:
        '''
        Return a boolean indicating, >=1 tool call is present.
        '''
        done = True
        tool_messages = []
        for call in message['react']:
            if "function" in call:
                message, success = self._use_tool(call)
                tool_messages.append(message)
                if success is False:
                    done = False
        self.add_message(tool_messages)
        return done

    
    def _use_tool(self, call):
        success = False
        name = call['function']
        args = call['parameters']['parameters']
        tool = self.tools.get(name)
        id = self.get_next_id()
        call['tool_call_id'] = id

        try:
            out = tool.run(**args)
        except Exception as e:
            success = False
            out = e
    
        message = {
            "role": "assistant",
            'content': {
                'type': 'text',
                'text': json.dumps({
                    "tool_call_id": id,
                    "tool_response": str(out)
                })
            }
        }
        return message, success


    def add_explicit_knowledge(self, prompt):
        prompt += instructions(f"""
            You are required to store this information in your memory as one block.
            Extract different relevant keywords and add the special keyword {self.get_special_keywords("explicit knowledge")} to it.
            Try to recall it and modify the keywords, if required!
        """)
        remove_tools = self.get_memory_tool_mask(memory_only=True)
        res = self.run(prompt, remove_tools=remove_tools)
        return res
    

    def get_special_keywords(self, key):
        return self.special_keywords.get(key, "")


    def get_memory_tool_mask(self):
        return [name for name in self.tools.keys() if name not in ["add", "recall", "modify"]]

    def render_content(self, message):
        parsed = json.dumps(message)
        parsed = json.loads(parsed)["content"]['text']
        return self.render_message_content(parsed)
    

    def render_message_content(self, parsed):
        inner_parts = []

        if type(parsed) == str and parsed[0] != "{":
            text = parsed
            escaped_text = html.escape(text).replace("\n", "<br>")
            inner_parts.append(f"<div style='margin-top:5px;'>{escaped_text}</div>")
        else:
            if type(parsed) == str:
                parsed = json.loads(parsed)

            if "react" in parsed:
                react_trace = parsed["react"]
                for i, item in enumerate(react_trace):
                    
                    if "thought" in item:
                        inner_parts.append(f"💭 <strong>Thought:</strong> {html.escape(item['thought'])}")
                    elif "function" in item:
                        function = html.escape(item.get("function", "unknown"))
                        params = item.get("parameters", {}).get("parameters", {})

                        lines = [f"⚙️ <strong>Action:</strong> {function}"]

                        if params:
                            param_lines = []
                            for key, value in params.items():
                                label = key.replace("_", " ").capitalize()

                                if isinstance(value, list):
                                    formatted = ", ".join(
                                        f"<code style='padding:2px 4px; border-radius:4px; background:none; color:inherit'>{html.escape(str(v))}</code>"
                                        for v in value
                                    )
                                    value_str = f"[{formatted}]"
                                elif isinstance(value, (int, float)):
                                    value_str = str(value)
                                else:
                                    value_str = html.escape(str(value))

                                param_lines.append(f"<li><strong>{label}:</strong> {value_str}</li>")

                            param_block = "<ul style='margin-left: 1.5em; margin-top: 0.3em'>" + "".join(param_lines) + "</ul>"
                            lines.append(param_block)

                        inner_parts.append("<br>".join(lines))
                
            if "response" in parsed:
                text = parsed["response"].strip()
                if text:
                    escaped_text = html.escape(text).replace("\n", "<br>")
                    inner_parts.append(f"<div style='margin-top:5px;'><strong>Response:</strong>: {escaped_text}</div>")

            if "tool_response" in parsed:
                text = parsed["tool_response"].strip()
                tool_id = str(parsed['tool_call_id']).strip()
                tool_call_note = f" <span style='color:#666; font-size:0.85em;'>(Tool Call ID: <code>{tool_id}</code>)</span>"

                if text:
                    if text.strip().startswith("<"):
                        formatted_text = f"<pre style='background:#f8f8f8; padding:8px; border-radius:6px; overflow:auto; font-family:monospace; font-size:0.9em'><code style='background:none; color:inherit'>{html.escape(text)}</code></pre>"
                    else:
                        formatted_text = html.escape(text).replace("\n", "<br>")
                    inner_parts.append(f"<div style='margin-top:5px;'><strong>Response:</strong><br>{formatted_text}</div>")

        return "<br>".join(inner_parts)



    def render_message(self, message, st=False):
        try:
            parsed = json.dumps(message)
            parsed = json.loads(parsed)["content"]['text']
        except (KeyError, json.JSONDecodeError):
            return
        
        
        role = message.get("role", "Unknown").capitalize()
        tool_call_note = ""
        '''
        if "tool_call_id" in message:
            tool_id = message["tool_call_id"]
            tool_call_note = f" <span style='color:#666; font-size:0.85em;'>(Tool Call ID: <code>{tool_id}</code>)</span>"
        '''
        bg_color = "#e0f7fa" if role == "Assistant" else "#f8f9fa"
        text_color = "#111"
        border = "1px solid #ccc"
        border_radius = "10px"
        padding = "10px"
        margin = "10px 0"

        html_parts = []

        content_block = self.render_message_content(parsed)

        html_parts.append(
            f"<div style='background:{bg_color}; color:{text_color}; border:{border}; border-radius:{border_radius}; padding:{padding}; margin:{margin};'>"
            f"<strong>{role}{tool_call_note}:</strong><br>{content_block}</div>"
        )
        return "".join(html_parts)
        

    def render_chat_html(self, messages=None):
        from IPython.display import HTML

        if messages is None:
            messages = self.chat.messages

        html_parts = ["<div style='font-family:Arial, sans-serif; line-height:1.6;'>"]

        for message in messages:
            html_parts.append(self.render_message(message))

        html_parts.append("</div>")
        return HTML("".join(html_parts))
        

    def get_output_jsonschema(self, remove_tools=[]):
        function_branches = []
        tools = self.tools
        
        for name, tool in tools.items():
            tool_name = name 
            if name in remove_tools:
                continue
            tool_schema = tool.parse()

            function_branches.append({
                "type": "object",
                "properties": {
                    "function": {
                        "type": "string",
                        "const": tool_name,
                        "description": f"Calls the {tool_name} function"
                    },
                    "parameters": tool_schema
                },
                "required": ["function", "parameters"],
                "additionalProperties": False
            })

        schema = {
        "type": "object",
        "properties": {
            "react": {
                "type": "array",
                "description": "A sequence of reasoning steps, including thoughts and actions.",
                "items": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "thought": {
                                    "type": "string",
                                    "description": "A reasoning step or internal reflection."
                                }
                            },
                            "required": ["thought"],
                            "additionalProperties": False
                        },
                        *function_branches
                    ]
                }
            },
            "response": {
                "type": "string",
                "description": "Final response to the user. Empty string if more actions/thoughts are expected."
            }
        },
        "required": ["react", "response"],
        "additionalProperties": False
        }

        return schema
