import os

from pydantic_ai import Tool, AgentRunResultEvent

from .tools.file_overview import get_file_message
from .tools.framework_loader import FrameworkLoader
from .tools.molecule_loader import molecule_loader
from .tools.output_extractor import output_extractor
from .tools.execute import execute_raspa, run_command
from .tools.make_input_file import call_input_file_agent
from .tools.tools_system import report_to_human, ask_human


class RaspaAgent:
    path: str

    def make_tools(self):
        if self.csd_path is None:
            print("A CSD path is required to access the coremof files.")

        def framework_loader(ctx: RunContext, simulation_name: str, framework_name: str):
            """Load a framework file as framework.cif in the agent's working directory."""
            path = os.path.join(ctx.deps["cwd"], simulation_name)
            return FrameworkLoader(path=path, coremof=False, csd_path=self.csd_path).run(framework_name)

        tool_list = [
            framework_loader,
            molecule_loader,
            execute_raspa,
            call_input_file_agent,
            run_command,
            output_extractor,
            report_to_human
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
        agent = Agent(
            tools=self.make_tools(),
            system_prompt=system_prompt_v2 + current_file_overview,
            model=self.model_name
        )
        async for event in agent.run_stream_events(query, deps={"cwd": self.path}):
            if isinstance(event, AgentRunResultEvent):
                return {event.result.output}
            else:
                await handle_event(event)



system_prompt_v2 = """
You specialize to assist with RASPA simulations.
You are equipped with tools to handle RASPA's input and output.
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
