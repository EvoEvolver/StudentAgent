import os
import subprocess

from dotenv import load_dotenv

from .file_overview import get_file_message
from ..utils import file
from .tools import RaspaTool
from pydantic_ai import Tool, Agent, RunContext

class MakeInputFile:
    def __init__(self, path=None):
        self.path = path

    def run(self, simulation_description: str):
        # files prompt contains all the existing files in the working directory

        agent = Agent(
            model="openai:gpt-5-mini",
            tools=[
                Tool(run_command, takes_ctx=True)
            ],
            system_prompt=f"""
You task is to create a RASPA simulation input file named 'simulation.input' based on the provided simulation description using the run_command tool.
The input file must strictly adhere to the RASPA input file format.
Use the following template as a reference for the structure and required parameters of the input file:
<template>
{template}
</template>
Here is the current files in the working directory:
{get_file_message(self.path, 1)}
"""
        )

        return agent.run_sync("Please generate 'simulation.input' based on the simulation description: "+simulation_description, deps={"cwd": self.path})

def call_input_file_agent(ctx: RunContext, simulation_name: str, message: str):
    """Call a input file agent to generate a RASPA simulation input file. The message should contain the essential information for the simulation."""
    path = os.path.join(ctx.deps["cwd"], simulation_name)
    print(f"making input file: {path}")
    # create the folder if it doesn't exist
    os.makedirs(path, exist_ok=True)
    return MakeInputFile(path=path).run(message)


class ExecuteRaspa(RaspaTool):
    def __init__(self, path=None):
        name = "execute_raspa"
        description = """Use this to start a RASPA simulation. The output indicates the success of the simulation."""
        super().__init__(name, description, path)

    def run(self):
        self.get_run_file()
        out = self.run_raspa()
        if out and isinstance(out, tuple):
            stdout, stderr = out
            return self.get_output(
                content=f"<terminal_output>{out.__str__()}</terminal_output>\\n (IMPORTANT: new, empty working directory created! To rerun, you must create all input files again!)"
            )
        return self.get_output(e=out)

    def check_success(self):
        if os.path.exists(os.path.join(self.path, "Output/")):
            return True
        else:
            return False

    def get_run_file(self):
        load_dotenv()
        raspa_dir = os.getenv("RASPA_DIR")
        if not raspa_dir:
            raise EnvironmentError("RASPA_DIR not found")

        content = (
            f"#! /bin/sh -f\nexport RASPA_DIR={raspa_dir}\n$RASPA_DIR/bin/simulate"
        )
        file_path = os.path.join(self.path, "run.sh")
        with open(file_path, "w") as f:
            f.write(content)
        os.chmod(file_path, 0o755)
        return

    def run_raspa(self):
        process = subprocess.Popen(
            ["bash", "run.sh"],
            cwd=self.get_path(full=True),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out = process.communicate()
        return out

def execute_raspa(ctx: RunContext, simulation_name:str):
    """Execute a RASPA simulation for a simulation."""
    path = os.path.join(ctx.deps["cwd"], simulation_name)
    return ExecuteRaspa(path=path).run()


def run_command(ctx: RunContext, command: str):
    """Run an arbitrary shell command in the working directory (path) of the agent.
    Provide a full command line string; it will be executed with the tool's path as the current working directory (cwd)."""

    timeout = 1200 # 20 minutes
    work_dir = ctx.deps["cwd"]
    process = subprocess.Popen(
        command,
        cwd=work_dir,
        text=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        raise RuntimeError(f"Command timed out after {timeout} seconds")

    if process.returncode != 0:
        err = f"Command failed with return code {process.returncode}"
        if stderr:
            err += f"\nError: {stderr}"
        if stdout:
            err += f"\nStdout: {stdout}"
        raise RuntimeError(err)

    return stdout if stdout else "(No output)"

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