import os

from pydantic_ai import RunContext, Agent, Tool

from student.agent.tools.file_overview import get_file_message
from student.agent.tools.output_parser import OutputParser
from student.agent.tools.execute import run_command


def parse_output(ctx: RunContext, file_path):
    """Use this tool to read the raspa output files since they are too long to read directly."""
    path = os.path.join(ctx.deps["cwd"], file_path)
    with open(path) as in_file:
        data = in_file.read()
    parser = OutputParser(path)
    out = parser.parse(data)
    out = parser.filter(out)
    out = parser.strip_block_fields(out)
    return out


class OutputExtractor(OutputParser):
    def __init__(self, path=None):
        super().__init__(path=path)

    def run(self, query: str):
        agent = Agent(
            model="openai:gpt-5-mini",
            tools=[
                Tool(run_command, takes_ctx=True),
                Tool(parse_output, takes_ctx=True)
            ],
            system_prompt=f"""
You task is to extract specific information from a RASPA simulation output file based on the provided query.
You should find the correct output file first.
Then you should call the output_parser tool to parse the output file and extract the relevant information.
File tree of the working directory:
{get_file_message(self.path, 3)}
"""
        )

        res = agent.run_sync(query, deps={"cwd": self.path})
        return res.output


def output_extractor(ctx: RunContext, simulation_name: str, query: str):
    """Use this tool to parse the raspa output files since they are too long to read directly.
    Provide the path of the output file you want to read based on the root directory (ALWAYS include the active subdirectory). Example: path=simulation_3/Output/System_0/output_Box_1.1.1_300.000000_100000.data"""
    path = os.path.join(ctx.deps["cwd"], simulation_name+"/Output")
    return OutputExtractor(path=path).run(query)
