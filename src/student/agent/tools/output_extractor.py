import os

from pydantic_ai import RunContext, Agent, Tool

from student.agent.tools.file_overview import get_file_message
from student.agent.tools.output_parser import OutputParser
from student.agent.tools.execute import run_command


def parse_output(ctx: RunContext, file_path: str):
    """Use this tool to read the raspa output files since they are too long to read directly."""
    path = os.path.join(ctx.deps["cwd"], file_path)
    with open(path) as in_file:
        data = in_file.read()
    parser = OutputParser(os.path.dirname(path))
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
output_parser tool can be used parse the output file and extract the relevant information.
If output_parser is not enough, you can use the run_command tool to read .data files with `grep` and `sed` because the file is too large to parse entirely.
File tree of the working directory:
{get_file_message(self.path, 3)}
"""
        )

        res = agent.run_sync(query, deps={"cwd": self.path})
        print("====== Output Extractor Agent Response ======")
        print(res)
        print("=============================================")
        return str(res.output)


def output_extractor(ctx: RunContext, simulation_name: str, query: str)->str:
    """Use this tool to extraction information the raspa output files ending with .data. The query should be a natural language question about the output."""
    path = os.path.join(ctx.deps["cwd"], simulation_name+"/Output")
    return OutputExtractor(path=path).run(query)
