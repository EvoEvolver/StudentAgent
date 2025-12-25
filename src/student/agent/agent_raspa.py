import os
import re
from typing import List

from pydantic_ai import (
    Agent,
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

from .agent_memory_helper import MemoryHelperAgent
from .event_logger import EventLogger
from .logger import Logger
from .memory import Memory
from .tools.execute import execute_raspa, run_command, execute_python_script
from .tools.file_overview import get_file_message
from .tools.framework_loader import FrameworkLoader
from student.agent.input_agent.make_input_file import call_input_file_agent
from .tools.molecule_loader import molecule_loader
from .tools.output_extractor import output_extractor


class RaspaAgent:
    path: str
    todo_list: str
    memory: Memory
    memory_agent: MemoryHelperAgent

    def make_tools(self):
        if self.csd_path is None:
            print("A CSD path is required to access the coremof files.")

        def framework_loader(
            ctx: RunContext, simulation_name: str, framework_name: str
        ):
            """Load a framework file as framework.cif in the agent's working directory."""
            path = os.path.join(ctx.deps["cwd"], simulation_name)
            return FrameworkLoader(
                path=path, coremof=False, csd_path=self.csd_path
            ).run(framework_name)

        def update_todo_list(ctx: RunContext, todo_list: str):
            """Update the agent's todo list."""
            self.todo_list = todo_list
            print("====== Current todo list in messages ======")
            print(self.todo_list)
            print("============================================")
            return "Todo list updated."

        def ask_human(ctx: RunContext, question: str):
            """Ask a human for help with a specific question."""
            print("====== Human Intervention Required ======")
            print(question)
            print("hint: you can type 'no' or 'skip' to skip this question.")
            print("=========================================")
            answer = input("Please provide your input: ")
            if answer.strip().lower() in ["no", "skip"]:
                return "No input provided by human."
            context = (
                "The agent is at the current todo list:\n" + self.todo_list + "\n---"
            )
            self.memory._create_memory_from_user_feedback(context, question, answer)
            return answer

        tool_list = [
            framework_loader,
            molecule_loader,
            execute_raspa,
            call_input_file_agent,
            execute_python_script,
            run_command,
            output_extractor,
            update_todo_list,
        ]

        if self.ask_human:
            tool_list.append(ask_human)

        pydantic_tools = [Tool(t, takes_ctx=True) for t in tool_list]
        return pydantic_tools

    def __init__(
        self,
        path="output",
        model_name="openai:gpt-5-mini",
        memory_path=None,
        csd_path=None,
        ask_human=False,
        retrieve_memory=False,
        logger: Logger = None,
        verbose: bool = True,
    ):
        self.csd_path = csd_path
        self.ask_human = ask_human
        self.retrieve_memory = retrieve_memory
        self.path = path
        self.model_name = model_name
        self.verbose = verbose

        # Setup logger
        self.logger = logger
        if self.logger is None:
            import os
            from datetime import datetime

            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(
                log_dir, f"raspa_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            self.logger = Logger(file=log_file, format="json", auto_load=False)

        self.initialize_memory(memory_path)

    def initialize_memory(self, memory_path):
        self.memory_agent = MemoryHelperAgent(
            provider="openai",
            verbose=self.verbose,
            expensive=False,
            cache=True,
            logger=self.logger,
        )
        memory = Memory._load_from_file(memory_path) if memory_path else Memory("root")
        memory.set_helper_agent(self.memory_agent)
        self.memory = memory
        self.memory_path = memory_path

    async def run(self, query: str):
        current_file_overview = get_file_message(self.path, 3)
        self.todo_list = ""

        def add_todo_list_to_message(
            ctx: RunContext[None], messages: list[ModelMessage]
        ) -> list[ModelMessage]:
            new_messages = messages.copy()
            if len(new_messages) < 3:
                return new_messages
            # find and delete existing todo list message
            if new_messages:
                for i, message in enumerate(new_messages):
                    if (
                        message.metadata
                        and message.metadata.get("type") == "memory_retrieval"
                    ):
                        del new_messages[i]
                        break
            # find all the tool calls that update the todo list and delete them except the last one
            if new_messages:
                indices_to_delete = []
                for i, message in enumerate(new_messages):
                    if message.kind == "response":
                        for part in message.parts:
                            if isinstance(part, ToolCallPart):
                                if part.tool_name == "update_todo_list":
                                    indices_to_delete.append(i)
                for index in reversed(indices_to_delete[:-1]):
                    del new_messages[index]

                indices_to_delete = []
                for i, message in enumerate(new_messages):
                    if message.kind == "request":
                        for part in message.parts:
                            if isinstance(part, ToolReturnPart):
                                if part.tool_name == "update_todo_list":
                                    indices_to_delete.append(i)
                for index in reversed(indices_to_delete[:-1]):
                    del new_messages[index]

            next_todo = ""
            if self.todo_list == "":
                next_todo = query
            else:
                todos = extract_tasks_from_markdown(self.todo_list)
                if todos:
                    next_todo = todos[0]

            if self.retrieve_memory and next_todo:
                memory_in_prompt = self.retrieve(next_todo)
                if memory_in_prompt:
                    memory_message = ModelRequest(
                        parts=[UserPromptPart(memory_in_prompt)]
                    )
                    memory_message.metadata = {"type": "memory_retrieval"}
                    new_messages.append(memory_message)

            current_tokens = ctx.usage.total_tokens
            if current_tokens > 10000 and len(new_messages) > 25:
                # keep the first message (system prompt) and the last 20 messages
                # new_messages = new_messages[:1] + new_messages[-20:]
                ...

            for message in new_messages:
                # print(message.kind)
                for p in message.parts:
                    # print(p)
                    ...

            return new_messages

        agent = Agent(
            tools=self.make_tools(),
            system_prompt=system_prompt_v2 + current_file_overview,
            model=self.model_name,
            history_processors=[add_todo_list_to_message],
        )

        # Create event logger for better logging integration
        event_logger = EventLogger(
            logger=self.logger,
            agent_id=f"raspa_agent_{id(self)}",
            agent_type="RaspaAgent",
            model_name=self.model_name,
            verbose=False,  # Don't print here, handle_event does the printing
        )
        event_logger.set_input_prompt(query)

        async for event in agent.run_stream_events(query, deps={"cwd": self.path}):
            # Log the event
            event_logger.log_event(event)

            if isinstance(event, AgentRunResultEvent):
                return {event.result.output}
            else:
                await handle_event(event)

    def retrieve(self, query: str) -> str:
        """Retrieve relevant information from memory based on the query."""
        # Placeholder for memory retrieval logic
        # In a real implementation, this would query a memory database or knowledge base
        print("Retrieving " + query)
        items = self.memory.retrieve(query, top_k=3)
        if len(items) == 0:
            return ""
        prompt = (
            "Potentially relevant information from your memory:"
            + "".join(["- " + item.content + "\\n" for item in items])
            + "\n----\n"
        )
        print(prompt)
        return prompt


def extract_tasks_from_markdown(markdown_todo: str) -> List[str]:
    """Extract task descriptions from markdown todo list."""
    tasks = []
    lines = markdown_todo.split("\n")

    for line in lines:
        line = line.strip()
        # Match markdown checkbox format: - [ ] not - [x]
        # match = re.match(r"^-\s*\[\s*[x ]?\s*\]\s*(.+)$", line)
        match = re.match(r"^-\s*\[\s*\]\s*(.+)$", line)
        if match:
            tasks.append(match.group(1).strip())

    return tasks


system_prompt_v2 = """
You specialize to assist with RASPA simulations.
You are equipped with tools to handle RASPA's input and output.
You should actively update your todo list as you progress through the task.
The todo list should be in markdown format with each line start with - [] or - [x].
The input generation tools will generate files in the folder with the simulation name in the working directory.
For a typical RASPA simulation, you need to create a framework file, a molecule file, and an input file.
"""
output_messages: list[str] = []


async def handle_event(event: AgentStreamEvent):
    if isinstance(event, PartStartEvent):
        print(f"[Request] Starting part {event.index}: {event.part!r}")
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta):
            print(f"{event.delta.content_delta}", end="", flush=True)
        elif isinstance(event.delta, ThinkingPartDelta):
            print(f"{event.delta.content_delta}", end="", flush=True)
        elif isinstance(event.delta, ToolCallPartDelta):
            print(f"{event.delta.args_delta}", end="", flush=True)
    elif isinstance(event, FunctionToolCallEvent):
        print(
            f"[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})"
        )
    elif isinstance(event, FunctionToolResultEvent):
        print(
            f"[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}"
        )
    elif isinstance(event, FinalResultEvent):
        print(
            f"[Result] The model starting producing a final result (tool_name={event.tool_name})"
        )
