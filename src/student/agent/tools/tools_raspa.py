import os
import subprocess

from dotenv import load_dotenv
from ..utils import file
from .tools import RaspaTool
from pydantic_ai import Agent, RunContext

class MakeInputFile(RaspaTool):
    def __init__(self, path=None, template_filename=None, advanced_template=False):
        super().__init__(path, "")

        self.name = "input_file"
        self.description = """Use this tool to write the simulation input file. The filename is always simulation.input.
ALWAYS use the template and modify it based on examples from your memory!
"""
        self.has_file = False
        self.set_template(template_filename, advanced_template)

    def set_template(self, template_filename=None, advanced_template=False):
        if template_filename is None:
            if advanced_template:
                template_filename = os.path.join(
                    os.path.dirname(__file__),
                    "templates/full_template_simulation.input",
                )
            else:
                template_filename = os.path.join(
                    os.path.dirname(__file__), "templates/template_simulation.input"
                )
        self.add_template(template_filename)

    def add_template(self, template_filename):
        if template_filename is None or not os.path.exists(template_filename):
            return False

        self.template_filename = template_filename
        with open(self.template_filename, "r") as file:
            template = file.read()
        self.description += f"\n<template>{template}</template>"
        return True

    def _run(self, file_content, file_name, path):
        try:
            new_path = os.path.join(path, file_name)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            with open(new_path, "w") as f:
                f.write(file_content)
            return self.get_output(content=f"Successfully generated: {file(new_path)}")
        except Exception as e:
            return self.get_output(e=e)

    def run(self, file_content):
        file_name = "simulation.input"
        out = self._run(file_content, file_name, self.get_path(full=True))
        if not (isinstance(out, str) and out.startswith("<error>")):
            self.has_file = True
        return out

def make_input_file(ctx: RunContext, folder_name: str, file_content: str):
    """Create a RASPA simulation input file named 'simulation.input' in the specified folder within the agent's working directory."""
    path = os.path.join(ctx.deps["cwd"], folder_name)
    return MakeInputFile(path=path).run(file_content)


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
        path = self.get_path(full=True)
        if os.path.exists(os.path.join(path, "Output/")):
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
        path = self.get_path(full=True)
        file_path = os.path.join(path, "run.sh")
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

def execute_raspa(ctx: RunContext, path_to_input_file:str):
    """Execute a RASPA simulation in the working directory (path) of the agent."""
    input_file_path = os.path.join(ctx.deps["cwd"], path_to_input_file)
    dir_name = os.path.dirname(input_file_path)
    return ExecuteRaspa(path=dir_name)


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


