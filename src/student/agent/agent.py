import html
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Union

import mllm
import mllm.provider_switch
from json_repair import repair_json
from mllm import Chat

from .logger import Logger
from .tools.tools import Tool


class Agent:
    name = "DefaultAgent"
    tools: Dict[str, Tool]
    system_prompt: str
    chat: Chat
    id: int
    conversation: List
    logger: Optional[Logger]
    agent_id: str
    model_name: str

    def __init__(
        self,
        tools: Dict[str, Tool] = {},
        cache=None,
        expensive=None,
        dir=None,
        version=None,
        provider="openai",
        verbose=False,
        logger: Optional[Logger] = None,
    ):
        self.tools = tools
        self.id = 0
        self.conversation = []  # list of conversations. new list starts at each reset
        self.system_prompt = ""
        self.token_counter = []
        self.logger = logger
        self.agent_id = str(uuid.uuid4())[:8]  # Short unique ID for this agent instance

        self.model_name = "unknown"  # Will be set by setup_provider

        if dir is not None and version is not None:
            self.reset_system_prompt(self.get_prompt(dir=dir, version=version))

        self.setup_logger()
        self.chat_config(cache, expensive)
        self.reset_chat()
        self.reset_id()
        self.setup_provider(provider)
        self.verbose = verbose
        if self.verbose:
            print(f"Created agent with agent_id: {self.agent_id}")

        self.max_message_content = 4000
        self.reset_token_count()

    ############ General Setup ############
    def setup_provider(self, provider="openai"):
        self.provider = provider
        if provider == "anthropic":
            mllm.provider_switch.set_default_to_anthropic()
            mllm.config.default_models.expensive = "claude-sonnet-4-5"
            mllm.config.default_models.normal = "claude-haiku-4-5"
        if provider == "openai":
            mllm.provider_switch.set_default_to_openai()
            mllm.config.default_models.expensive = "gpt-5.1"
            mllm.config.default_models.normal = "gpt-5-nano"
            mllm.config.default_options.temperature = 1

        self._update_model_name()

    def setup_logger(self):
        """Initialize logger if not provided."""
        if self.logger is None:
            # Create default log directory and file
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"agent_{self.agent_id}.json")
            self.logger = Logger(file=log_file, format="json", auto_load=False)

    def _update_model_name(self):
        """Update the model name based on current config."""
        if self.expensive:
            self.model_name = mllm.config.default_models.expensive
        else:
            self.model_name = mllm.config.default_models.normal

    def _get_agent_type(self) -> str:
        """Get the agent type name."""
        return self.__class__.__name__

    def _log_llm_call(
        self,
        input_messages: List[Dict],
        output_message: str,
        metadata: Optional[Dict] = None,
    ):
        """Log an LLM call if logger is available."""
        if self.logger:
            self.logger.log_llm_call(
                agent_id=self.agent_id,
                agent_type=self._get_agent_type(),
                model=self.model_name,
                input_messages=input_messages,
                output_message={"content": output_message},
                metadata=metadata,
            )

    def _log_tool_call(
        self,
        tool_name: str,
        tool_input: Dict,
        tool_output: Any,
        metadata: Optional[Dict] = None,
    ):
        """Log a tool call if logger is available."""
        if self.logger:
            self.logger.log_tool_call(
                agent_id=self.agent_id,
                agent_type=self._get_agent_type(),
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                metadata=metadata,
            )

    def _log_error(
        self, error_message: str, error_type: str, metadata: Optional[Dict] = None
    ):
        """Log an error if logger is available."""
        if self.logger:
            self.logger.log_error(
                agent_id=self.agent_id,
                agent_type=self._get_agent_type(),
                error_message=error_message,
                error_type=error_type,
                metadata=metadata,
            )

    def _log_info(self, message: str, metadata: Optional[Dict] = None):
        """Log general information if logger is available."""
        if self.logger:
            self.logger.log_info(
                agent_id=self.agent_id,
                agent_type=self._get_agent_type(),
                message=message,
                metadata=metadata,
            )

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

    def chat_length(self):
        return len(self.chat.messages)

    def get_prompt(
        self,
        type,
        dir=None,
        version="v1",
        version_general="v3",
        version_output="v3",
        json=True,
        general=True,
    ):

        full = (
            self._build_prompt(f"{dir}/general", version_general)
            if general is True
            else ""
        )

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

    ########### Checkpointing #######

    def save(self, folder_name):
        os.makedirs(folder_name, exist_ok=True)
        self.save_conversation(os.path.join(folder_name, "conversation.txt"))

    def load(self, folder_name):
        if len(self.conversation) == 0:
            return

        file_path = os.path.join(folder_name, "conversation.txt")
        if os.path.exists(file_path):
            try:
                self.load_conversation(file_path)
            except Exception as e:
                return e

    ############ Running ############

    def single_run(self, prompt, expensive=False, parse=None, info=""):
        """Run a single prompt without conversation context. Logs if logger is available."""
        chat = Chat(dedent=True)
        chat += prompt

        # Log input
        input_messages = [{"role": "user", "content": prompt}]

        res = chat.complete(cache=False, expensive=expensive, parse=parse)
        token_count = self.token_count(chat)

        # Log output
        self._log_llm_call(
            input_messages=input_messages,
            output_message=res,
            metadata={
                "expensive": expensive,
                "method": "single_run: " + str(info),
                "token_count": token_count,
            },
        )

        return res

    def run(
        self,
        prompt: str,
        max_iter: int = 15,
        schema: str = None,
        remove_tools: List[str] = [],
    ):
        """Run agent with ReAct loop. Logs all LLM calls and tool executions."""
        if schema is None:
            schema = self.get_output_jsonschema(remove_tools=remove_tools)
        options = self.get_options(schema)

        # Log the start of the run
        self._log_info(
            "Starting run with prompt",
            metadata={"max_iter": max_iter, "prompt_preview": prompt[:100]},
        )

        self.chat += prompt
        n_tool_responses = 0
        for i in range(max_iter):
            response, done, n = self._run(options, iteration=i)
            n_tool_responses += n
            if done:
                break

        n = (
            i + 2 + n_tool_responses
        )  # number of new messages = (i+1) responses + 1 user message
        self.update_conversation(n)

        final_response = self.response(response)
        # Log completion
        self._log_info(
            "Run completed",
            metadata={
                "iterations": i + 1,
                "tool_calls": n_tool_responses,
                "response_preview": final_response[:100] if final_response else "",
            },
        )

        return final_response

    def _run(self, options, iteration=0):
        """Single iteration of the ReAct loop. Logs LLM call."""
        # Extract input messages for logging
        input_messages = [
            {"role": msg.get("role", "unknown"), "content": str(msg.get("content", ""))}
            for msg in self.chat.messages[-5:]
        ]  # Last 5 messages for context

        res = self.chat.complete(
            parse=None, cache=self.cache, expensive=self.expensive, options=options
        )
        token_count = self.token_count()
        res = json.loads(repair_json(res))

        # Log the LLM call
        self._log_llm_call(
            input_messages=input_messages,
            output_message=json.dumps(res),
            metadata={
                "iteration": iteration,
                "cache": self.cache,
                "expensive": self.expensive,
                "method": "_run",
                "token_count": token_count,
            },
        )

        done, n_tool_responses, tool_messages = self.use_tools(res)

        if self.verbose is True:
            print(self.response(res))

        return res, done, n_tool_responses

    def run_single_tool(self, prompt: str):
        schema = self.get_output_jsonschema(remove_tools=[])
        options = self.get_options(schema)
        chat = Chat(system_message=self.system_prompt)
        chat += prompt
        res = chat.complete(
            parse=None, cache=self.cache, expensive=self.expensive, options=options
        )
        res = json.loads(repair_json(res))
        done, n_tool_responses, tool_messages = self.use_tools(res)

        # Concatenate all tool messages into a single string
        concatenated = "\n\n".join(
            msg["content"]["text"]
            for msg in tool_messages
            if msg.get("content", {}).get("text")
        )
        return concatenated

    def response(self, message: Dict):
        response = message.get("response", "")
        return response

    def get_options(self, schema):
        options = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": schema, "strict": True},
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
        assert "role" in message, "Message must contain 'role'"
        assert (
            "content" in message or message["role"] == "tool"
        ), "Message must contain 'content' (or be a tool message)"
        self.chat.messages.append(message)

    def update_conversation(self, n_messages):
        # for n in range(n_messages):
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

    def use_tools(self, message: Dict):
        """
        Return a boolean indicating, >=1 tool call is present.
        """
        n = 0
        done = True
        tool_messages = []

        react = message["actions"]
        if not isinstance(react, list):
            try:
                react = json.loads(repair_json(message["actions"]))
            except Exception as e:
                raise e
        for call in react:
            if "function" in call:
                message, success = self._use_tool(call)
                tool_messages.append(message)
                if success is False:
                    done = False
                n += 1
        print("tool calls used:", n)
        print("tool messages:", tool_messages)
        self.add_message(tool_messages)
        return done, n, tool_messages

    def _use_tool(self, call):
        success = False
        name = call["function"]
        args = call["parameters"]
        if "parameters" in args.keys():
            args = args["parameters"]
        tool = self.tools.get(name, None)
        id = self.get_next_id()
        call["tool_call_id"] = id

        # Print tool call information
        if tool is None:
            print(f"\n[TOOL ERROR] Invalid tool name: {name}")
            error_msg = f"Invalid tool name: {name}"
            self._log_error(
                error_msg,
                "InvalidToolError",
                metadata={"requested_tool": name, "tool_call_id": id},
            )
            name = "INVALID TOOL NAME"
        else:
            print(f"\n[TOOL CALL] {tool.name}")
            if args:
                print(f"[ARGUMENTS] {json.dumps(args, indent=2)}")
            name = tool.name

        import time

        start_time = time.time()

        try:
            out = tool.run(**args)
            success = True
            execution_time = time.time() - start_time

            # Print tool result
            print(
                f"[TOOL RESULT] {str(out)[:500]}{'...' if len(str(out)) > 500 else ''}"
            )

            # Log successful tool execution
            self._log_tool_call(
                tool_name=name,
                tool_input=args,
                tool_output=str(out)[:1000],  # Limit output size
                metadata={
                    "success": True,
                    "execution_time": execution_time,
                    "tool_call_id": id,
                },
            )

        except Exception as e:
            success = False
            out = e
            execution_time = time.time() - start_time
            print(f"[TOOL ERROR] {str(e)}")

            # Log tool error
            self._log_error(
                error_message=str(e),
                error_type=type(e).__name__,
                metadata={
                    "tool_name": name,
                    "tool_input": args,
                    "execution_time": execution_time,
                    "tool_call_id": id,
                },
            )

        message = {
            "role": "tool",
            "tool_call_id": id,
            "name": name,
            "content": {"type": "tool_result", "result": str(out), "success": success},
        }
        return message, success

    def get_memory_tool_mask(self):
        return []

    ############ Token counting ############

    def reset_token_count(self):
        """Reset the token counter to an empty list."""
        token_sum_old = self._sum_token_count()
        self.token_counter = []
        return token_sum_old

    def _sum_token_count(self) -> tuple[int, int]:
        """
        Get the total number of input and output tokens used.

        Returns:
            tuple[int, int]: Total (input_tokens, output_tokens) used across all interactions
        """
        in_tokens = 0
        out_tokens = 0
        for count in self.token_counter:
            in_tokens += count.get("input_tokens", 0)
            out_tokens += count.get("output_tokens", 0)
        return {"input_tokens": in_tokens, "output_tokens": out_tokens}

    def token_count(self, chat: Chat = None) -> None:
        """
        Count tokens for a chat interaction and add to the token counter.

        Args:
            chat: The chat to count tokens for. If None, uses self.chat
        """
        if chat is None:
            chat = self.chat
        count = self._token_count(chat)
        self.token_counter.append(count)
        return count

    def _token_count(self, chat: Chat) -> Dict[str, int]:
        """
        Get the token counts for a specific chat interaction.

        Args:
            chat: The chat to count tokens for

        Returns:
            Dict with input_tokens and output_tokens counts
        """
        input_tokens = chat.additional_res["prompt_tokens"]
        output_tokens = chat.additional_res["completion_tokens"]

        return {"input_tokens": input_tokens, "output_tokens": output_tokens}

    ############ Load/Save ############

    def save_conversation(self, filename, note=""):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "note": note,
                    "messages": self.conversation,
                    "id": self.id,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return

    def load_conversation(self, filename, reset=True):
        with open(filename, "r", encoding="utf-8") as f:
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
        parsed = json.loads(repair_json(parsed))["content"]["text"]
        return self.render_message_content(parsed, no_background=no_background)

    def render_message_content(self, parsed, no_background=False):
        inner_parts = []

        if isinstance(parsed, str) and len(parsed) > 0 and parsed[0] != "{":
            text = parsed
            escaped_text = html.escape(text).replace("\n", "<br>")
            inner_parts.append(f"<div style='margin-top:5px;'>{escaped_text}</div>")
        else:
            if isinstance(parsed, str):
                parsed = json.loads(repair_json(parsed))
            if "react" in parsed:
                actions_trace = parsed["actions"]
                if not isinstance(actions_trace, list):
                    try:
                        actions_trace = json.loads(repair_json(parsed["actions"]))
                    except Exception:
                        return
                for i, item in enumerate(actions_trace):

                    if "thought" in item:
                        inner_parts.append(
                            f"💭 <strong>Thought:</strong> {html.escape(item['thought'])}"
                        )
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

                                param_lines.append(
                                    f"<li><strong>{label}:</strong> {value_str}</li>"
                                )

                            param_block = (
                                "<ul style='margin-left: 1.5em; margin-top: 0.3em'>"
                                + "".join(param_lines)
                                + "</ul>"
                            )
                            lines.append(param_block)

                        inner_parts.append("<br>".join(lines))

            if "response" in parsed:
                text = parsed["response"].strip()
                if text:
                    escaped_text = html.escape(text).replace("\n", "<br>")
                    inner_parts.append(
                        f"<div style='margin-top:5px;'><strong>Response:</strong>: {escaped_text}</div>"
                    )

            if "tool_response" in parsed:
                text = parsed["tool_response"].strip()
                # tool_id = str(parsed["tool_call_id"]).strip()
                # tool_call_note = f" <span style='color:#666; font-size:0.85em;'>(Tool Call ID: <code>{tool_id}</code>)</span>"
                tool_name = str(parsed["tool_name"]).strip()

                if text:
                    if text.strip().startswith("<"):
                        if no_background is True:
                            formatted_text = f"<pre style='background:#2d2d2d; color:#e0e0e0; padding:8px; border-radius:6px; overflow:auto; font-family:monospace; font-size:0.9em'><code style='background:none; color:inherit'>{html.escape(text)}</code></pre>"
                        else:
                            formatted_text = f"<pre style='background:#f8f8f8; padding:8px; border-radius:6px; overflow:auto; font-family:monospace; font-size:0.9em'><code style='background:none; color:inherit'>{html.escape(text)}</code></pre>"
                    else:
                        formatted_text = html.escape(text).replace("\n", "<br>")
                    # inner_parts.append(f"<div style='margin-top:5px;'><strong>Tool:</strong><br>{formatted_text}</div>")
                    inner_parts.append(
                        f"<div style='margin-top:5px;'><strong>Tool: {tool_name}</strong><br>{formatted_text}</div>"
                    )

        return "<br>".join(inner_parts)

    def render_message(self, message, st=False):
        role = message.get("role", "Unknown").capitalize()

        # Handle new tool message format
        if role == "Tool":
            tool_name = message.get("name", "unknown")
            content = message.get("content", {})
            if isinstance(content, dict) and content.get("type") == "tool_result":
                result = html.escape(str(content.get("result", "")))
                success = content.get("success", True)
                status_icon = "✓" if success else "✗"
                status_color = "#4CAF50" if success else "#f44336"

                bg_color = "#f1f8e9" if success else "#ffebee"
                content_block = f"<div style='margin-top:5px;'><span style='color:{status_color}; font-weight:bold;'>{status_icon}</span> <strong>Tool: {html.escape(tool_name)}</strong><br>{result.replace(chr(10), '<br>')}</div>"
            else:
                content_block = str(content)
        else:
            # Handle regular messages
            try:
                parsed = json.dumps(message)
                parsed = json.loads(parsed)["content"]["text"]
                content_block = self.render_message_content(parsed)
            except (KeyError, json.JSONDecodeError):
                return

            bg_color = "#e0f7fa" if role == "Assistant" else "#f8f9fa"

        text_color = "#111"
        border = "1px solid #ccc"
        border_radius = "10px"
        padding = "10px"
        margin = "10px 0"

        html_parts = []
        html_parts.append(
            f"<div style='background:{bg_color}; color:{text_color}; border:{border}; border-radius:{border_radius}; padding:{padding}; margin:{margin};'>"
            f"<strong>{role}:</strong><br>{content_block}</div>"
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

    def get_tool_schema(self, remove_tools=[], json=True):
        """Generate tool descriptions dynamically."""
        tool_schema = []

        for name, tool in self.tools.items():
            tool_name = name
            if name in remove_tools:
                continue
            tool_schema.append(tool.parse(tool_name, json=json))

        return tool_schema

    def get_output_jsonschema(self, remove_tools=[]):
        function_branches = self.get_tool_schema(remove_tools=remove_tools, json=True)

        schema = {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": "A sequence of reasoning steps, including thoughts and actions.",
                    "items": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "thought": {
                                        "type": "string",
                                        "description": "A reasoning step or internal reflection.",
                                    }
                                },
                                "required": ["thought"],
                                "additionalProperties": False,
                            },
                            *function_branches,
                        ]
                    },
                }
            },
            "required": ["actions"],
            "additionalProperties": False,
        }

        return schema
