"""
Event logger for pydantic-ai agent events.

Provides utilities to log pydantic-ai streaming events to the Logger system.
"""

from pydantic_ai import (
    AgentRunResultEvent,
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)

from .logger import Logger


class EventLogger:
    """
    Wrapper to handle pydantic-ai streaming events and log them appropriately.

    Usage:
        event_logger = EventLogger(logger, agent_id, agent_type)
        async for event in agent.run_stream(...):
            event_logger.log_event(event)
            # ... handle event ...
    """

    def __init__(
        self,
        logger: Logger,
        agent_id: str,
        agent_type: str,
        verbose: bool = False,
        model_name: str = None,
    ):
        """
        Initialize event logger.

        Args:
            logger: Logger instance to use
            agent_id: Unique agent identifier
            agent_type: Type of agent (class name)
            verbose: Whether to print events to console
            model_name: Model name for logging
        """
        self.logger = logger
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.verbose = verbose
        self.model_name = model_name

        # Track events for aggregation
        self.current_text_parts = []
        self.tool_calls = {}
        self.input_prompt = None  # Track the input prompt

    def set_input_prompt(self, prompt: str):
        """Set the input prompt for logging."""
        self.input_prompt = prompt

    def log_event(self, event: AgentStreamEvent):
        """
        Log a pydantic-ai stream event.

        Args:
            event: The stream event to log
        """
        if isinstance(event, PartStartEvent):
            self._log_part_start(event)

        elif isinstance(event, PartDeltaEvent):
            self._log_part_delta(event)

        elif isinstance(event, FunctionToolCallEvent):
            self._log_tool_call(event)

        elif isinstance(event, FunctionToolResultEvent):
            self._log_tool_result(event)

        elif isinstance(event, FinalResultEvent):
            self._log_final_result(event)

        elif isinstance(event, AgentRunResultEvent):
            self._log_run_result(event)

    def _log_part_start(self, event: PartStartEvent):
        """Log the start of a new part."""
        if self.verbose:
            print(f"\n[Part Start] Index {event.index}: {type(event.part).__name__}")

        self.logger.log_info(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            message=f"Part started: {type(event.part).__name__}",
            metadata={
                "event_type": "part_start",
                "part_index": event.index,
                "part_type": type(event.part).__name__,
            },
        )

    def _log_part_delta(self, event: PartDeltaEvent):
        """Log incremental part updates."""
        if isinstance(event.delta, TextPartDelta):
            # Accumulate text deltas
            self.current_text_parts.append(event.delta.content_delta)

            if self.verbose:
                print(event.delta.content_delta, end="", flush=True)

        elif isinstance(event.delta, ThinkingPartDelta):
            if self.verbose:
                print(f"[Thinking] {event.delta.content_delta}", end="", flush=True)

        elif isinstance(event.delta, ToolCallPartDelta):
            # Accumulate tool call args
            if event.index not in self.tool_calls:
                self.tool_calls[event.index] = {"args_parts": []}

            self.tool_calls[event.index]["args_parts"].append(event.delta.args_delta)

            if self.verbose:
                print(event.delta.args_delta, end="", flush=True)

    def _log_tool_call(self, event: FunctionToolCallEvent):
        """Log a complete tool call."""
        if self.verbose:
            print(f"\n[Tool Call] {event.part.tool_name}")
            print(f"[Arguments] {event.part.args}")

        self.logger.log_tool_call(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            tool_name=event.part.tool_name,
            tool_input=event.part.args,
            tool_output="<pending>",
            metadata={
                "event_type": "tool_call_start",
                "tool_call_id": event.part.tool_call_id,
            },
        )

        # Store for later matching with result
        self.tool_calls[event.part.tool_call_id] = {
            "tool_name": event.part.tool_name,
            "args": event.part.args,
        }

    def _log_tool_result(self, event: FunctionToolResultEvent):
        """Log a tool execution result."""
        if self.verbose:
            result_preview = str(event.result.content)[:500]
            print(f"\n[Tool Result] {result_preview}")
            if len(str(event.result.content)) > 500:
                print("...")

        self.logger.log_tool_call(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            tool_name=self.tool_calls.get(event.tool_call_id, {}).get(
                "tool_name", "unknown"
            ),
            tool_input=self.tool_calls.get(event.tool_call_id, {}).get("args", {}),
            tool_output=str(event.result.content)[:1000],  # Limit output size
            metadata={
                "event_type": "tool_result",
                "tool_call_id": event.tool_call_id,
                "success": True,
            },
        )

    def _log_final_result(self, event: FinalResultEvent):
        """Log when final result generation starts."""
        if self.verbose:
            print(f"\n[Final Result] Starting (tool_name={event.tool_name})")

        self.logger.log_info(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            message="Final result generation started",
            metadata={
                "event_type": "final_result_start",
                "tool_name": event.tool_name,
            },
        )

    def _log_run_result(self, event: AgentRunResultEvent):
        """Log the complete run result."""
        # Combine accumulated text parts
        full_text = "".join(self.current_text_parts)

        if self.verbose:
            print("\n[Run Complete]")
            if hasattr(event.result, "usage"):
                usage = event.result.usage()
                print(
                    f"[Token Usage] Request: {usage.input_tokens}, Response: {usage.output_tokens}"
                )

        # Log as LLM call
        metadata = {
            "event_type": "run_complete",
        }

        if hasattr(event.result, "usage"):
            usage = event.result.usage()
            metadata["token_usage"] = {
                "request_tokens": usage.input_tokens,
                "response_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }

        # Build input messages from tracked prompt
        input_messages = []
        if self.input_prompt:
            input_messages.append({"role": "user", "content": self.input_prompt})

        self.logger.log_llm_call(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            model=self.model_name or "unknown",
            input_messages=input_messages,
            output_message={
                "content": str(event.result.output),
                "full_text": full_text,
            },
            metadata=metadata,
        )

        # Reset accumulators
        self.current_text_parts = []
        self.tool_calls = {}
        self.input_prompt = None


async def log_stream_events(
    stream,
    logger: Logger,
    agent_id: str,
    agent_type: str,
    verbose: bool = False,
    model_name: str = None,
    input_prompt: str = None,
):
    """
    Async generator that logs all events from a pydantic-ai stream.

    Args:
        stream: Async iterator of AgentStreamEvent
        logger: Logger instance
        agent_id: Agent identifier
        agent_type: Agent type name
        verbose: Whether to print to console
        model_name: Model name for logging
        input_prompt: The input prompt to log

    Yields:
        Original stream events
    """
    event_logger = EventLogger(logger, agent_id, agent_type, verbose, model_name)
    if input_prompt:
        event_logger.set_input_prompt(input_prompt)

    async for event in stream:
        event_logger.log_event(event)
        yield event
