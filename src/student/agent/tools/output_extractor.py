import asyncio
import os

from pydantic_ai import Agent, RunContext, Tool

from student.agent.reader_agent.agentic_reader import AgenticReaderOptions, agentic_reader_with_events
from student.agent.tools.execute import run_command, execute_python_script
from student.agent.tools.file_overview import get_file_message
from student.agent.tools.output_parser import OutputParser

def data_to_preview(text: str)->str:
    # find all the pos starting with \n===============================
    preview = text
    positions = [i for i in range(len(preview)) if preview.startswith("\n===============================", i)]
    # make the preview the texts -100 to +100 around each position
    snippets = []
    for pos in positions:
        start = max(0, pos - 100)
        end = min(len(preview), pos + 100)
        snippets.append("(Position:" + str(pos) + ")\n")
        snippets.append(preview[start:end])
        snippets.append("")
    res = "\n".join(snippets)
    print("Data to preview:", res)
    return res

def read_output_file(ctx: RunContext, output_data_path: str, question: str) -> str:
    output_path = os.path.join(ctx.deps["cwd"], output_data_path)
    try:
        # Read the output file
        with open(output_path, "r") as f:
            data_content = f.read()

        # Use agentic reader to answer the question
        print("Agentic started solveing documentation question...")
        print("Question:", question)
        options = AgenticReaderOptions(max_iterations=5, model="openai:gpt-5-mini")
        result = asyncio.run(agentic_reader_with_events(question, data_content, emit_event=lambda _, data: print(data), text_to_preview=data_to_preview, options=options))
        print("Agentic reader result:", result)
        return result
    except FileNotFoundError:
        return f"Documentation file not found at {output_path}"
    except Exception as e:
        return f"Error reading documentation: {str(e)}"


class OutputExtractor(OutputParser):
    def __init__(self, path=None):
        super().__init__(path=path)

    def run(self, query: str):
        agent = Agent(
            model="openai:gpt-5-mini",
            tools=[
                Tool(read_output_file, takes_ctx=True),
                Tool(run_command, takes_ctx=True),
                Tool(execute_python_script, takes_ctx=True)
            ],
            system_prompt=f"""
You task is to extract specific information from a RASPA simulation output file based on the provided query.
You should find the correct output file first.
output_parser tool can be used parse the output file and extract the relevant information.
Use the read_output_file tool to read and query the output files ending with .data. The query should be a natural language question about the output.
If read_output_file is not enough, you can use the run_command tool to calculate or extract further information from the output data.
File tree of the working directory:
{get_file_message(self.path, 3)}

""",
        )

        res = agent.run_sync(query, deps={"cwd": self.path})
        print("====== Output Extractor Agent Response ======")
        print(res)
        print("=============================================")
        return str(res.output)


def output_extractor(ctx: RunContext, simulation_name: str, query: str) -> str:
    """Use this tool to extraction information the raspa output files ending with .data. The query should be a natural language question about the output."""
    path = os.path.join(ctx.deps["cwd"], simulation_name + "/Output")
    return OutputExtractor(path=path).run(query)
