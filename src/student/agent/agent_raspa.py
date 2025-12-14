import os

from pydantic_ai import Tool, AgentRunResultEvent, ModelMessage, ModelRequest, SystemPromptPart, UserPromptPart

from .tools.file_overview import get_file_message
from .tools.framework_loader import FrameworkLoader
from .tools.molecule_loader import molecule_loader
from .tools.output_extractor import output_extractor
from .tools.execute import execute_raspa, run_command
from .tools.make_input_file import call_input_file_agent
from .tools.tools_system import ask_human


class RaspaAgent:
    path: str
    todo_list: str

    def make_tools(self):
        if self.csd_path is None:
            print("A CSD path is required to access the coremof files.")

        def framework_loader(ctx: RunContext, simulation_name: str, framework_name: str):
            """Load a framework file as framework.cif in the agent's working directory."""
            path = os.path.join(ctx.deps["cwd"], simulation_name)
            return FrameworkLoader(path=path, coremof=False, csd_path=self.csd_path).run(framework_name)

        def update_todo_list(ctx: RunContext, todo_list: str):
            """Update the agent's todo list."""
            self.todo_list = todo_list
            print("====== Current todo list in messages ======")
            print(self.todo_list)
            print("============================================")
            return f"Todo list updated to: {self.todo_list}"

        tool_list = [
            framework_loader,
            molecule_loader,
            execute_raspa,
            call_input_file_agent,
            run_command,
            output_extractor,
            update_todo_list
        ]

        if self.ask_human:
            tool_list.append(ask_human)

        pydantic_tools = [Tool(t, takes_ctx=True) for t in tool_list]
        return pydantic_tools

    def __init__(
            self,
            path="output",
            model_name="openai:gpt-5-mini",
            csd_path=None,
            ask_human=False
    ):
        self.csd_path = csd_path
        self.ask_human = ask_human
        self.path = path
        self.model_name = model_name



    async def run(self, query: str):
        current_file_overview = get_file_message(self.path, 3)
        self.todo_list = ""

        def add_todo_list_to_message(messages: list[ModelMessage]) -> list[ModelMessage]:
            new_messages = messages.copy()
            # find and delete existing todo list message
            if new_messages:
                for i, message in enumerate(new_messages):
                    if message.metadata and message.metadata.get("type") == "todo_list":
                        del new_messages[i]
                        break
            # find all the tool calls that update the todo list and delete them except the last one
            if new_messages:
                indices_to_delete = []
                for i, message in enumerate(new_messages):
                    if message.metadata and message.metadata.get("type") == "tool_call":
                        tool_name = message.metadata.get("tool_name")
                        if tool_name == "update_todo_list":
                            indices_to_delete.append(i)
                for index in reversed(indices_to_delete[:-1]):
                    del new_messages[index]
            # add updated todo list message
            if self.todo_list != "":
                todolist_message = ModelRequest(parts=[UserPromptPart(f"Current todo list:\n{self.todo_list}")])
                todolist_message.metadata = {"type": "todo_list"}
                #new_messages.append(todolist_message)
            return new_messages

        agent = Agent(
            tools=self.make_tools(),
            system_prompt=system_prompt_v2 + current_file_overview,
            model=self.model_name,
            history_processors=[add_todo_list_to_message]
        )
        async for event in agent.run_stream_events(query, deps={"cwd": self.path}):
            if isinstance(event, AgentRunResultEvent):
                return {event.result.output}
            else:
                await handle_event(event)



system_prompt_v2 = """
You specialize to assist with RASPA simulations.
You are equipped with tools to handle RASPA's input and output.
You should actively update your todo list as you progress through the task.
The todo list should be in markdown [] [x] format.
The input generation tools will generate files in the folder with the simulation name in the working directory.
"""
output_messages: list[str] = []
from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)
async def handle_event(event: AgentStreamEvent):
    if isinstance(event, PartStartEvent):
        print(f'[Request] Starting part {event.index}: {event.part!r}')
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta):
            print(f'{event.delta.content_delta}', end='', flush=True)
        elif isinstance(event.delta, ThinkingPartDelta):
            print(f'{event.delta.content_delta}', end='', flush=True)
        elif isinstance(event.delta, ToolCallPartDelta):
            print(f'{event.delta.args_delta}', end='', flush=True)
    elif isinstance(event, FunctionToolCallEvent):
        print(
            f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})'
        )
    elif isinstance(event, FunctionToolResultEvent):
        print(f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}')
    elif isinstance(event, FinalResultEvent):
        print(f'[Result] The model starting producing a final result (tool_name={event.tool_name})')
