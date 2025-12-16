"""
Base Agent class using pydantic-ai for LLM interactions.

This replaces the old mllm-based implementation with pydantic-ai's
structured approach to agent development.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import (
    AgentRunResultEvent,
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPartDelta,
    ThinkingPartDelta,
    Tool,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)

from .logger import Logger


class Agent:
    """
    Base agent class that wraps pydantic-ai Agent functionality.

    Provides:
    - Structured tool calling
    - Conversation management
    - Logging integration
    - Checkpointing
    - Multiple model provider support
    """

    name = "DefaultAgent"

    def __init__(
        self,
        tools: Union[Dict[str, Any], List[Callable]] = None,
        system_prompt: str = "",
        model: str = None,
        provider: str = "openai",
        expensive: bool = False,
        cache: bool = True,
        verbose: bool = False,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize Agent.

        Args:
            tools: Dictionary of tool name -> Tool object, or list of callable functions
            system_prompt: System prompt for the agent
            model: Model identifier (e.g., "openai:gpt-4", "anthropic:claude-3-5-sonnet")
            provider: Model provider ("openai" or "anthropic")
            expensive: Whether to use expensive models by default
            cache: Whether to use caching (where available)
            verbose: Whether to print verbose output
            logger: Optional Logger instance for tracking
        """
        self.system_prompt = system_prompt
        self.provider = provider
        self.expensive = expensive
        self.cache = cache
        self.verbose = verbose

        # Agent identification
        self.agent_id = (
            datetime.now().strftime("%Y-%m-%d %H-%M-%S-") + str(uuid.uuid4())[:4]
        )

        # Setup model
        self.model_name = self._get_model_name(model, provider, expensive)

        # Setup logger
        self.logger = logger
        if self.logger is None:
            self.setup_logger()

        # Convert tools to pydantic-ai format
        self.tools = self._prepare_tools(tools)

        # Create pydantic-ai agent
        self._agent = None
        self._create_pydantic_agent()

        # Conversation tracking
        self.conversation_history: List[ModelMessage] = []
        self.token_usage = {"input_tokens": 0, "output_tokens": 0}

    def _get_model_name(
        self, model: Optional[str], provider: str, expensive: bool
    ) -> str:
        """Determine the model name based on provider and expense settings."""
        if model:
            return model

        # Default models by provider
        if provider == "anthropic":
            return "claude-sonnet-4-5" if expensive else "claude-haiku-4-5"
        else:  # openai
            return "gpt-5.1" if expensive else "gpt-5-mini"

    def _prepare_tools(self, tools: Union[Dict, List, None]) -> List[Tool]:
        """
        Convert tools to pydantic-ai Tool format.

        Args:
            tools: Dictionary of tool objects or list of callables

        Returns:
            List of pydantic-ai Tool instances
        """
        if tools is None:
            return []

        pydantic_tools = []

        if isinstance(tools, dict):
            # Convert old-style Tool objects to pydantic-ai Tools
            for name, tool_obj in tools.items():
                if hasattr(tool_obj, "run"):
                    # Wrap the run method
                    pydantic_tools.append(Tool(tool_obj.run, takes_ctx=False))
        elif isinstance(tools, list):
            # Assume list of callables
            for func in tools:
                pydantic_tools.append(Tool(func, takes_ctx=False))

        return pydantic_tools

    def _create_pydantic_agent(self):
        """Create the underlying pydantic-ai agent."""
        model_id = (
            f"{self.provider}:{self.model_name}"
            if ":" not in self.model_name
            else self.model_name
        )

        self._agent = PydanticAgent(
            model=model_id,
            system_prompt=self.system_prompt,
            tools=self.tools,
        )

    def setup_logger(self):
        """Initialize logger if not provided."""
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"agent_{self.agent_id}.json")
        self.logger = Logger(file=log_file, format="json", auto_load=False)

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

    async def run(
        self,
        prompt: str,
        deps: Optional[Dict[str, Any]] = None,
        message_history: Optional[List[ModelMessage]] = None,
    ) -> str:
        """
        Run the agent with a prompt.

        Args:
            prompt: User prompt
            deps: Optional dependencies to pass to tools via RunContext
            message_history: Optional message history to use

        Returns:
            Agent's response as string
        """
        self._log_info("Starting run", metadata={"prompt_preview": prompt[:100]})

        try:
            result = await self._agent.run(
                prompt,
                deps=deps,
                message_history=message_history or self.conversation_history,
            )

            # Update conversation history
            self.conversation_history = result.all_messages()

            # Update token usage
            if hasattr(result, "usage"):
                self.token_usage["input_tokens"] += result.usage().request_tokens
                self.token_usage["output_tokens"] += result.usage().response_tokens

            # Log the interaction
            self._log_llm_call(
                input_messages=[{"role": "user", "content": prompt}],
                output_message=str(result.output),
                metadata={
                    "token_usage": self.token_usage,
                    "method": "run",
                },
            )

            if self.verbose:
                print(f"[Response] {result.output}")

            return str(result.output)

        except Exception as e:
            self._log_error(str(e), type(e).__name__, metadata={"prompt": prompt[:100]})
            raise

    async def run_stream(
        self,
        prompt: str,
        deps: Optional[Dict[str, Any]] = None,
        message_history: Optional[List[ModelMessage]] = None,
    ):
        """
        Run the agent with streaming output.

        Args:
            prompt: User prompt
            deps: Optional dependencies to pass to tools
            message_history: Optional message history

        Yields:
            Stream events from the agent
        """
        self._log_info(
            "Starting streaming run", metadata={"prompt_preview": prompt[:100]}
        )

        try:
            async for event in self._agent.run_stream(
                prompt,
                deps=deps,
                message_history=message_history or self.conversation_history,
            ):
                # Log tool calls
                if isinstance(event, FunctionToolCallEvent):
                    if self.verbose:
                        print(f"\n[Tool Call] {event.part.tool_name}")
                        print(f"[Arguments] {event.part.args}")

                # Log tool results
                elif isinstance(event, FunctionToolResultEvent):
                    self._log_tool_call(
                        tool_name=event.tool_name or "unknown",
                        tool_input={},
                        tool_output=str(event.result.content)[:1000],
                        metadata={"tool_call_id": event.tool_call_id},
                    )

                    if self.verbose:
                        print(f"[Tool Result] {str(event.result.content)[:500]}")

                # Log text deltas
                elif isinstance(event, PartDeltaEvent) and isinstance(
                    event.delta, TextPartDelta
                ):
                    if self.verbose:
                        print(event.delta.content_delta, end="", flush=True)

                yield event

                # Update conversation history on final result
                if isinstance(event, AgentRunResultEvent):
                    self.conversation_history = event.result.all_messages()

                    if hasattr(event.result, "usage"):
                        self.token_usage[
                            "input_tokens"
                        ] += event.result.usage().request_tokens
                        self.token_usage[
                            "output_tokens"
                        ] += event.result.usage().response_tokens

                    self._log_llm_call(
                        input_messages=[{"role": "user", "content": prompt}],
                        output_message=str(event.result.output),
                        metadata={
                            "token_usage": self.token_usage,
                            "method": "run_stream",
                        },
                    )

        except Exception as e:
            self._log_error(str(e), type(e).__name__, metadata={"prompt": prompt[:100]})
            raise

    def single_run(self, prompt: str, expensive: Optional[bool] = None) -> str:
        """
        Run a single prompt without conversation context (synchronous wrapper).

        Args:
            prompt: The prompt to run
            expensive: Whether to use expensive model (overrides default)

        Returns:
            The response string
        """

        # Temporarily adjust model if needed
        original_model = self.model_name
        if expensive is not None and expensive != self.expensive:
            self.model_name = self._get_model_name(None, self.provider, expensive)
            self._create_pydantic_agent()

        try:
            # Run with empty history for single run
            result = asyncio.run(self.run(prompt, message_history=[]))
            return result
        finally:
            # Restore original model
            if expensive is not None and expensive != self.expensive:
                self.model_name = original_model
                self._create_pydantic_agent()

    def reset_system_prompt(self, sys_prompt: str, append: bool = False):
        """
        Reset or append to the system prompt.

        Args:
            sys_prompt: New system prompt
            append: Whether to append to existing prompt
        """
        if append:
            self.system_prompt += sys_prompt
        else:
            self.system_prompt = sys_prompt

        # Recreate agent with new prompt
        self._create_pydantic_agent()

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self._log_info("Conversation reset")

    def reset_token_count(self):
        """Reset token usage counters."""
        old_usage = self.token_usage.copy()
        self.token_usage = {"input_tokens": 0, "output_tokens": 0}
        return old_usage

    def get_token_usage(self) -> Dict[str, int]:
        """Get current token usage."""
        return self.token_usage.copy()

    # Checkpointing methods
    def save(self, folder_name: str):
        """
        Save agent state to folder.

        Args:
            folder_name: Directory to save state
        """
        os.makedirs(folder_name, exist_ok=True)

        # Save conversation history
        conversation_file = os.path.join(folder_name, "conversation.json")
        self.save_conversation(conversation_file)

        # Save configuration
        config_file = os.path.join(folder_name, "config.json")
        with open(config_file, "w") as f:
            json.dump(
                {
                    "agent_id": self.agent_id,
                    "model_name": self.model_name,
                    "provider": self.provider,
                    "expensive": self.expensive,
                    "system_prompt": self.system_prompt,
                    "token_usage": self.token_usage,
                },
                f,
                indent=2,
            )

    def load(self, folder_name: str):
        """
        Load agent state from folder.

        Args:
            folder_name: Directory to load state from
        """
        # Load conversation history
        conversation_file = os.path.join(folder_name, "conversation.json")
        if os.path.exists(conversation_file):
            try:
                self.load_conversation(conversation_file)
            except Exception as e:
                self._log_error(f"Failed to load conversation: {e}", "LoadError")

        # Load configuration
        config_file = os.path.join(folder_name, "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                    self.agent_id = config.get("agent_id", self.agent_id)
                    self.token_usage = config.get("token_usage", self.token_usage)
            except Exception as e:
                self._log_error(f"Failed to load config: {e}", "LoadError")

    def save_conversation(self, filename: str):
        """Save conversation history to file."""
        # Convert ModelMessage objects to serializable format
        serializable_history = []
        for msg in self.conversation_history:
            serializable_history.append(
                {
                    "kind": msg.kind,
                    "parts": [self._serialize_part(part) for part in msg.parts],
                }
            )

        with open(filename, "w") as f:
            json.dump(
                {
                    "agent_id": self.agent_id,
                    "conversation": serializable_history,
                },
                f,
                indent=2,
            )

    def _serialize_part(self, part) -> Dict:
        """Serialize a message part to dictionary."""
        if isinstance(part, UserPromptPart):
            return {"type": "user_prompt", "content": part.content}
        elif isinstance(part, ToolCallPart):
            return {
                "type": "tool_call",
                "tool_name": part.tool_name,
                "args": part.args,
                "tool_call_id": part.tool_call_id,
            }
        elif isinstance(part, ToolReturnPart):
            return {
                "type": "tool_return",
                "tool_name": part.tool_name,
                "content": str(part.content),
                "tool_call_id": part.tool_call_id,
            }
        else:
            return {"type": "unknown", "content": str(part)}

    def load_conversation(self, filename: str):
        """Load conversation history from file."""
        with open(filename, "r") as f:
            data = json.load(f)

        # Note: Full reconstruction of ModelMessage objects from JSON is complex
        # For now, we'll store the serialized version and log it
        self._log_info(
            "Loaded conversation from file",
            metadata={
                "filename": filename,
                "message_count": len(data.get("conversation", [])),
            },
        )

    def _build_prompt(self, dir: str, version: str) -> str:
        """
        Read a prompt file and return it as a string.

        Args:
            dir: Directory name under prompts/system/
            version: Version file name (e.g., "v1")

        Returns:
            Prompt text
        """
        here = os.path.dirname(__file__)
        base_dir = os.path.join(here, "prompts", "system")
        path = os.path.join(base_dir, dir, f"{version}.xml")

        if not os.path.isfile(path):
            raise RuntimeError(f"Required prompt file missing: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read().strip()

        return text

    def get_prompt(
        self,
        type: str,
        dir: str = None,
        version: str = "v1",
        version_general: str = "v3",
        version_output: str = "v3",
        json: bool = True,
        general: bool = True,
    ) -> str:
        """
        Build a composite prompt from multiple files.

        Args:
            type: Prompt type ("general" or specific type)
            dir: Directory under prompts/system/
            version: Version of specific prompt
            version_general: Version of general prompt
            version_output: Version of output format prompt
            json: Whether to include output format instructions
            general: Whether to include general prompt

        Returns:
            Composite prompt string
        """
        full = ""

        if general and dir:
            full = self._build_prompt(f"{dir}/general", version_general)

        if type != "general" and dir:
            p = os.path.join(dir, type)
            add = self._build_prompt(p, version)
            full += add

        if json:
            full += "\n"
            full += self._build_prompt("output", version_output)

        return full
