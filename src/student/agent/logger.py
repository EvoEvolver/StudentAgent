"""
Keep track of the full agentic process across multiple agents.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class LogEntry:
    """Structured log entry for agent interactions."""
    timestamp: str
    agent_id: str
    agent_type: str
    model: str
    event_type: str  # "llm_call", "tool_call", "error", "info"
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper handling of complex types."""
        return asdict(self)


class Logger:
    """
    Multi-agent chronological logger that tracks all agent interactions.

    Features:
    - Chronological logging across multiple agents
    - Agent identification and model tracking
    - Input/output capture for LLM calls and tool executions
    - Metadata support for custom information
    - JSON and human-readable formats
    - Token usage tracking
    - Error logging
    """

    def __init__(self, file: str, format: str = "json", auto_load: bool = True):
        """
        Initialize logger.

        Args:
            file: Path to log file
            format: "json" for JSON lines, "text" for human-readable
            auto_load: Automatically load existing entries from file on init
        """
        self.file = file
        self.format = format
        self.entries: List[LogEntry] = []
        self._setup_file()

        # Auto-load existing entries if file exists
        if auto_load:
            self.load_from_file()

    def _setup_file(self):
        """Create log file and directory if they don't exist."""
        file_dir = os.path.dirname(self.file)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        if not os.path.exists(self.file):
            with open(self.file, "w") as f:
                if self.format == "json":
                    pass  # JSON lines format - no header needed
                else:
                    f.write(f"=== Agent Log Started: {datetime.now().isoformat()} ===\n\n")

    def set_file(self, file: str, auto_load: bool = True):
        """
        Change the log file path.

        Args:
            file: New log file path
            auto_load: Automatically load existing entries from new file
        """
        self.file = file
        self._setup_file()

        if auto_load:
            self.load_from_file()

    def log_llm_call(
        self,
        agent_id: str,
        agent_type: str,
        model: str,
        input_messages: List[Dict[str, Any]],
        output_message: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an LLM API call.

        Args:
            agent_id: Unique identifier for the agent instance
            agent_type: Type of agent (e.g., "StudentAgent", "RaspaAgent")
            model: Model name (e.g., "gpt-4", "claude-sonnet-4")
            input_messages: List of input messages sent to LLM
            output_message: Response from LLM
            metadata: Additional info (token counts, temperature, etc.)
        """
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            agent_type=agent_type,
            model=model,
            event_type="llm_call",
            input_data=input_messages,
            output_data=output_message,
            metadata=metadata or {}
        )
        self._write_entry(entry)

    def log_tool_call(
        self,
        agent_id: str,
        agent_type: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a tool execution.

        Args:
            agent_id: Unique identifier for the agent instance
            agent_type: Type of agent
            tool_name: Name of the tool being called
            tool_input: Parameters passed to tool
            tool_output: Result from tool execution
            metadata: Additional info (execution time, etc.)
        """
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            agent_type=agent_type,
            model="N/A",
            event_type="tool_call",
            input_data={"tool": tool_name, "parameters": tool_input},
            output_data=tool_output,
            metadata=metadata or {}
        )
        self._write_entry(entry)

    def log_error(
        self,
        agent_id: str,
        agent_type: str,
        error_message: str,
        error_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an error occurrence.

        Args:
            agent_id: Unique identifier for the agent instance
            agent_type: Type of agent
            error_message: Error description
            error_type: Type/class of error
            metadata: Additional context
        """
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            agent_type=agent_type,
            model="N/A",
            event_type="error",
            input_data={"error_type": error_type},
            output_data=error_message,
            metadata=metadata or {}
        )
        self._write_entry(entry)

    def log_info(
        self,
        agent_id: str,
        agent_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log general information or events.

        Args:
            agent_id: Unique identifier for the agent instance
            agent_type: Type of agent
            message: Information message
            metadata: Additional context
        """
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            agent_id=agent_id,
            agent_type=agent_type,
            model="N/A",
            event_type="info",
            input_data=None,
            output_data=message,
            metadata=metadata or {}
        )
        self._write_entry(entry)

    def _write_entry(self, entry: LogEntry):
        """Write a log entry to file."""
        self.entries.append(entry)

        with open(self.file, "a") as f:
            if self.format == "json":
                f.write(json.dumps(entry.to_dict()) + "\n")
            else:
                f.write(self._format_text_entry(entry) + "\n")

    def _format_text_entry(self, entry: LogEntry) -> str:
        """Format entry as human-readable text."""
        lines = [
            f"{'='*80}",
            f"[{entry.timestamp}] {entry.agent_type} ({entry.agent_id})",
            f"Event: {entry.event_type.upper()} | Model: {entry.model}",
            ""
        ]

        if entry.input_data is not None:
            lines.append("INPUT:")
            lines.append(self._format_data(entry.input_data, indent=2))
            lines.append("")

        if entry.output_data is not None:
            lines.append("OUTPUT:")
            lines.append(self._format_data(entry.output_data, indent=2))
            lines.append("")

        if entry.metadata:
            lines.append("METADATA:")
            lines.append(self._format_data(entry.metadata, indent=2))
            lines.append("")

        return "\n".join(lines)

    def _format_data(self, data: Any, indent: int = 0) -> str:
        """Format data for text output."""
        prefix = " " * indent
        if isinstance(data, (dict, list)):
            return prefix + json.dumps(data, indent=2)
        return prefix + str(data)

    def get_entries(
        self,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> List[LogEntry]:
        """
        Retrieve log entries with optional filtering.

        Args:
            agent_id: Filter by specific agent instance
            agent_type: Filter by agent type
            event_type: Filter by event type

        Returns:
            List of matching log entries
        """
        filtered = self.entries

        if agent_id:
            filtered = [e for e in filtered if e.agent_id == agent_id]
        if agent_type:
            filtered = [e for e in filtered if e.agent_type == agent_type]
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]

        return filtered

    def get_agent_summary(self, agent_id: str) -> Dict[str, Any]:
        """
        Generate summary statistics for a specific agent.

        Args:
            agent_id: Agent instance identifier

        Returns:
            Dictionary with counts and statistics
        """
        entries = self.get_entries(agent_id=agent_id)

        summary = {
            "agent_id": agent_id,
            "total_events": len(entries),
            "event_counts": {},
            "models_used": set(),
            "total_tokens": 0,
            "errors": 0
        }

        for entry in entries:
            # Count events by type
            summary["event_counts"][entry.event_type] = \
                summary["event_counts"].get(entry.event_type, 0) + 1

            # Track models
            if entry.model != "N/A":
                summary["models_used"].add(entry.model)

            # Sum tokens if available
            if entry.metadata and "tokens" in entry.metadata:
                summary["total_tokens"] += entry.metadata["tokens"]

            # Count errors
            if entry.event_type == "error":
                summary["errors"] += 1

        summary["models_used"] = list(summary["models_used"])
        return summary

    def export_to_json(self, output_file: str):
        """Export all entries to a JSON file."""
        with open(output_file, "w") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    def clear(self):
        """Clear in-memory entries (does not delete log file)."""
        self.entries = []

    def load_from_file(self, replace: bool = False):
        """
        Load entries from existing log file (JSON format only).

        Args:
            replace: If True, replace in-memory entries. If False, append new entries.
        """
        if self.format != "json" or not os.path.exists(self.file):
            return

        if replace:
            self.entries = []

        # Track existing timestamps to avoid duplicates
        existing_timestamps = {(e.timestamp, e.agent_id, e.event_type) for e in self.entries}

        loaded_count = 0
        with open(self.file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        entry = LogEntry(**data)

                        # Avoid duplicates
                        entry_key = (entry.timestamp, entry.agent_id, entry.event_type)
                        if entry_key not in existing_timestamps:
                            self.entries.append(entry)
                            existing_timestamps.add(entry_key)
                            loaded_count += 1
                    except (json.JSONDecodeError, TypeError):
                        # Skip malformed lines silently
                        continue

        return loaded_count
