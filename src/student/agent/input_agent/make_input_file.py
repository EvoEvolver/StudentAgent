import os
import asyncio

from pydantic_ai import Agent, RunContext, Tool

from student.agent.tools.execute import run_command
from student.agent.tools.file_overview import get_file_message
from student.agent.reader_agent.agentic_reader import agentic_reader, AgenticReaderOptions, agentic_reader_with_events


def read_documentation(ctx: RunContext, question: str) -> str:
    """Read and query the RASPA input file documentation using an agentic reader.

    Args:
        ctx: The run context containing dependencies
        question: The question to ask about the documentation

    Returns:
        The answer from the documentation
    """
    # Get the path to input_files.tex (in the same directory as this file)
    doc_path = os.path.join(os.path.dirname(__file__), "input_files.tex")

    try:
        # Read the documentation file
        with open(doc_path, "r") as f:
            tex_content = f.read()

        # Use agentic reader to answer the question
        print("Agentic started solveing documentation question...")
        print("Question:", question)
        options = AgenticReaderOptions(max_iterations=5, model="openai:gpt-5-mini")
        result = asyncio.run(agentic_reader_with_events(question, tex_content, emit_event=lambda _, data: print(data), options=options))
        print("Agentic reader result:", result)
        return result
    except FileNotFoundError:
        return f"Documentation file not found at {doc_path}"
    except Exception as e:
        return f"Error reading documentation: {str(e)}"


class MakeInputFile:
    def __init__(self, path=None):
        self.path = path

    def run(self, simulation_description: str):
        # files prompt contains all the existing files in the working directory

        agent = Agent(
            model="openai:gpt-5-mini",
            tools=[
                Tool(run_command, takes_ctx=True),
                Tool(read_documentation, takes_ctx=True)
            ],
            system_prompt=f"""
You task is to create a RASPA simulation input file named 'simulation.input' based on the provided simulation description using the run_command tool.
The input file must strictly adhere to the RASPA input file format.
Use the following template as a reference for the structure and required parameters of the input file:
<template>
{template}
</template>

If you need additional information about RASPA input file format, parameters, or examples, use the read_documentation tool to query the documentation.

Here is the current files in the working directory:
{get_file_message(self.path, 1)}
""",
        )

        return agent.run_sync(
            "Please generate 'simulation.input' based on the simulation description: "
            + simulation_description,
            deps={"cwd": self.path},
        )


template = """
SimulationType                MonteCarlo
NumberOfCycles                [int] # The number of cycles for the production run.
NumberOfInitializationCycles  [int] # The number of cycles used to initialize the system to equilibrate the positions of the atoms in the system.
PrintEvery                    [int]

Forcefield                    local
CutOff                        14.0
RemoveAtomNumberCodeFromLabel yes

...                                                # Optional: Add here if properties should be computed
# The square brackets respresent that this is optional! They have to be removed in a real setting!
[                                                  # Optional: box specifications (NEVER use a .cif file for a box!)
Box [int]                                          # IMPORTANT: System numbering is 0-based (always use 0 for first system)
BoxLengths [real] [real] [real]                    # in Angström (MUST BE twice the CutOff)
ExternalTemperature [real]                         # in Kelvin
ExternalPressure [list of real]                    # in pascal e.g. 1e4 1e5 1e6
]

[                                                  # Optional: framework specifications
Framework [int]                                    # IMPORTANT: System numbering is 0-based (always use 0 for first system)
FrameworkName framework                            # same as framework.cif file
UnitCells [int] [int] [int]                        # number of unit cells in each dimension (must be twice the CutOff), calcuated automatically when generating a framework.cif with the tool
HeliumVoidFraction [real]                          # has to be obtained from a separate simulation (essential to compute excess-adsorption)
ExternalTemperature [real]                         # in Kelvin
ExternalPressure [list of real]                    # in pascal e.g. 1e4 1e5 1e6
...
]

Component 0 MoleculeName       [molecule name]     # IMPORTANT: same as the file name of [molecule name].def
            MoleculeDefinition  local
            ...                                    # Monte Carlo move probabilities or properties

[
Component 1 MoleculeName        [molecule name]    # Optional: second component (use 1 for second component)
            MolFraction [real]                     # For multiple components, specify mol fraction (0 to 1) for all components
]
# always empty line in the end!"""


def call_input_file_agent(ctx: RunContext, simulation_name: str, message: str):
    """Call an input file agent to generate a RASPA simulation.input file. The message should contain the essential information for the simulation."""
    path = os.path.join(ctx.deps["cwd"], simulation_name)
    print(f"making input file: {path}")
    # create the folder if it doesn't exist
    os.makedirs(path, exist_ok=True)
    return MakeInputFile(path=path).run(message)
