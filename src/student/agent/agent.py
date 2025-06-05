from .tools.tools import Tool

from mllm import Chat
import mllm.provider_switch
from typing import List, Dict, Union
import json
from .utils import *
import html


class Agent:
    tools : Dict[str, Tool]
    system_prompt : str
    chat : Chat
    id : int
    conversation : List

    def __init__(self, tools: Dict[str, Tool] = {}, cache=None, expensive=None, dir=None, version=None, provider="openai"):
        self.tools = tools
        self.id = 0
        self.conversation = [] # list of conversations. new list starts at each reset
        self.system_prompt = ""

        if dir is not None and version is not None:
            self.reset_system_prompt(self.get_prompt(dir=dir, version=version))
            
        self.chat_config(cache, expensive)
        self.reset_chat()
        self.reset_id()
        self.setup_provider(provider)
        
    ############ General Setup ############
    def setup_provider(self, provider="openai"):
        self.provider = provider
        if provider== "anthropic":
            mllm.provider_switch.set_default_to_anthropic()
            mllm.config.default_models.expensive = "claude-sonnet-4-20250514"
            mllm.config.default_models.normal = "claude-3-5-haiku-20241022"
        

    def _build_prompt(self, dir, version) -> str:
        # Reads the prompt file and returns it as a string.
        here = os.path.dirname(__file__)
        base_dir = os.path.join(here, "prompts", "system")
        
        path = os.path.join(base_dir, dir)
        path = os.path.join(path, f"{version}.xml")

        if not os.path.isfile(path):
            raise RuntimeError(f"Required prompt file missing: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read().strip()

        return text
    
    def get_prompt(self, type, dir=None, version="v1", version_general="v3", version_output="v3", json=True, general=True):
        
        full = self._build_prompt(f"{dir}/general", version_general) if general is True else ""

        if type != "general":
            p = os.path.join(dir, type)
            add = self._build_prompt(p, version)   
            full += add
        
        if json is True:
            full += "\n"
            full += self._build_prompt("output", version_output)
        return full
    

    def chat_config(self, cache=None, expensive=None):
        self.cache = cache if cache is not None else True
        self.expensive = expensive if expensive is not None else True

    def reset_system_prompt(self, sys_prompt, append=False):
        if append is True:
            self.system_prompt += sys_prompt
        else:
            self.system_prompt = sys_prompt
        self.reset_chat()


    def reset_chat(self):
        self.chat = Chat(system_message=self.system_prompt, dedent=False)
        if len(self.conversation) > 0:
            if len(self.conversation[-1]) != 0:
                self.conversation.append([])
        else:
            self.conversation.append([])
            
    
    def reset_id(self):
        self.id = 0

 
    ############ Running ############

    def single_run(self, prompt, expensive=False, parse=None):
        chat = Chat(dedent=True)
        chat += prompt
        res = chat.complete(cache=True, expensive=expensive, parse=parse) # TODO: check cache
        return res


    def run(self, prompt: str, max_iter: int=15, schema:str=None, remove_tools:List[str]=[]):
        if schema is None:
            schema = self.get_output_jsonschema(remove_tools=remove_tools)
        options = self.get_options(schema)
        
        self.chat += prompt
        n_tool_responses = 0
        for i in range(max_iter):
            response, done, n = self._run(options)     
            n_tool_responses += n
            if done:
                break         
        
        n = i + 2 + n_tool_responses # number of new messages = (i+1) responses + 1 user message
        self.update_conversation(n)   
        return self.response(response)
    
    def _run(self, options):
        res = self.chat.complete(parse=None, cache=self.cache, expensive=self.expensive, options=options)
        res = json.loads(res)
        done, n_tool_responses = self.use_tools(res)    
        return res, done, n_tool_responses
    
    def response(self, message: Dict):
        response = message.get("response", '')
        return response

    def get_options(self, schema):
        options = {"response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                        "schema": schema,
                        "strict": True
                    },
            }
        }
        return options

    def add_message(self, message: Union[Dict[str, str], List[Dict[str, str]]]):
        if isinstance(message, list):
            for m in message:
                self.add_message(m)
            return
        if message is None:
            return
        assert "content" in message and "role" in message, "Message must contain 'content' and 'role'"
        self.chat.messages.append(message)


    def update_conversation(self, n_messages):
        #for n in range(n_messages):
        #    message = self.chat.messages[-(n_messages-n)]
        #    self.conversation[-1].append(message)
        new_messages = self.chat.messages[-n_messages:] if n_messages > 0 else []
        self.conversation[-1].extend(new_messages)


    def get_conversation(self):
        conv = []    
        for conversation in self.conversation:
            for message in conversation:
                conv.append(message)
            conv.append("reset")
        return conv[:-1]

    def get_next_id(self):
        self.id += 1
        return self.id


    def use_tools(self, message: Dict) -> bool:
        '''
        Return a boolean indicating, >=1 tool call is present.
        '''
        n = 0
        done = True
        tool_messages = []
        for call in message['react']:
            if "function" in call:
                message, success = self._use_tool(call)
                tool_messages.append(message)
                if success is False:
                    done = False
                n += 1
        self.add_message(tool_messages)
        return done, n

    
    def _use_tool(self, call):
        success = False
        name = call['function']
        args = call['parameters']
        if "parameters" in args.keys():
            args = args['parameters']
        tool = self.tools.get(name, None)
        id = self.get_next_id()
        call['tool_call_id'] = id

        try:
            out = tool.run(**args)
        except Exception as e:
            success = False
            out = e

        if tool is None:
            name = "INVALID TOOL NAME"
        else:
            name = tool.name
            
        message = {
            "role": "user",
            'content': {
                'type': 'text',
                'text': json.dumps({
                    "tool_call_id": id,
                    "tool_name": name,
                    "tool_response": str(out),
                })
            }
        }
        return message, success
    
    def get_memory_tool_mask(self):
        return []


    ############ Load/Save ############

    def save_conversation(self, filename, note=""):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "note" : note,
                    "messages": self.conversation,
                    'id': self.id,
                },
                f,
                ensure_ascii=False,
                indent=2
            )
        return


    def load_conversation(self, filename, reset=True):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        messages = data.get("messages", [])
        id = data.get("id", 0)

        # Only loads the messages if no previous conversation happened.
        if reset is True:
            self.conversation = messages
            self.chat.messages = messages[-1]
            self.id = id

        return messages


    ############ Render/Parsing ############


    def render_content(self, message, no_background=False):
        parsed = json.dumps(message)
        parsed = json.loads(parsed)["content"]['text']
        return self.render_message_content(parsed, no_background=no_background)
    

    def render_message_content(self, parsed, no_background=False):
        inner_parts = []

        if type(parsed) == str and len(parsed) > 0 and parsed[0] != "{":
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
                        params = item.get("parameters", {})
                        params = params.get("parameters", params)

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
                #tool_call_note = f" <span style='color:#666; font-size:0.85em;'>(Tool Call ID: <code>{tool_id}</code>)</span>"
                tool_name = str(parsed['tool_name']).strip()

                if text:
                    if text.strip().startswith("<"):
                        if no_background is True:
                            formatted_text = f"<pre style='background:#2d2d2d; color:#e0e0e0; padding:8px; border-radius:6px; overflow:auto; font-family:monospace; font-size:0.9em'><code style='background:none; color:inherit'>{html.escape(text)}</code></pre>"
                        else:
                            formatted_text = f"<pre style='background:#f8f8f8; padding:8px; border-radius:6px; overflow:auto; font-family:monospace; font-size:0.9em'><code style='background:none; color:inherit'>{html.escape(text)}</code></pre>"
                    else:
                        formatted_text = html.escape(text).replace("\n", "<br>")
                    #inner_parts.append(f"<div style='margin-top:5px;'><strong>Tool:</strong><br>{formatted_text}</div>")
                    inner_parts.append(f"<div style='margin-top:5px;'><strong>Tool: {tool_name}</strong><br>{formatted_text}</div>")


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
        
    def render_conversation(self):
        from IPython.display import HTML

        html_parts = ["<div style='font-family:Arial, sans-serif; line-height:1.6;'>"]        
        
        for messages in self.conversation:

            for message in messages:
                html_parts.append(self.render_message(message))

            html_parts.append("</div>")
            html_parts.append("<hr>")
        html_parts.pop()
        
        return HTML("".join(html_parts))


    def get_output_jsonschema(self, remove_tools=[]):
        function_branches = []
        tools = self.tools
        
        for name, tool in tools.items():
            tool_name = name 
            if name in remove_tools:
                continue
            tool_schema = tool.parse(tool_name)

            function_branches.append(tool_schema)

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
                "description": "Final response to the user. IGNORED IF a function is included in the react scheme"
            }
        },
        "required": ["react", "response"],
        "additionalProperties": False
        }

        return schema


