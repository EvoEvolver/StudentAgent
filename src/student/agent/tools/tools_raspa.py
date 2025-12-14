import os
import subprocess

from dotenv import load_dotenv

from ..utils import file
from .tools import RaspaTool


class ReadFile(RaspaTool):
    def __init__(self, path=None):
        name = "read_file"
        description = """Use this tool to read the content of a text file (not directory!).
You must provide the path to the file as file name (based on the root directory NOT the current working directory).
For long documents, this tool only reads the beginning.
The tool does not work on RASPA output files directly, use the output_parser tool for that.
"""
        super().__init__(name, description, path)

        self.blacklist = ["output/", "Output/"]

    def run(self, file_name):
        path = self.get_path(full=False)
        content = None
        file_path = os.path.join(path, file_name)

        for x in self.blacklist:
            if file_path in x:
                return self.get_output(
                    e="Access to this file path is not possible with this tool."
                )

        try:
            if os.path.exists(file_path) and os.path.isfile(file_path):
                with open(file_path, "r") as f:
                    content = f.read()
            elif os.path.exists(file_path) and not os.path.isfile(file_path):
                content = "The is a directory, not a file!"
            else:
                content = "This path does not exist!"
            return self.get_output(content=f"{file(file_path)}:\n{content}")
        except Exception as e:
            return self.get_output(
                e="You must provide the path to the file based on the root directory NOT the current working directory)."
                + e
            )


class WriteFile(RaspaTool):
    def __init__(self, path=None):
        name = "write_file"
        description = """Use this tool to write text into a new file.
IMPORTANT: You must provide a file name based on the root directory NOT the current working directory.
IMPORTANT: To edit a (small) file, you must first read a file with another tool and then write it completely new with this tool. Dont do this to copy files!
IMPORTANT: This will overwrite any existing file with the same name!
"""
        super().__init__(name, description, path)

    def run(self, file_content, file_name):
        path = self.get_path(full=False)
        return self._run(file_content, file_name, path)

    def _run(self, file_content, file_name, path):
        try:
            new_path = os.path.join(path, file_name)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            with open(new_path, "w") as f:
                f.write(file_content)
            return self.get_output(content=f"Successfully generated: {file(new_path)}")
        except Exception as e:
            return self.get_output(e=e)


class MakeInputFile(WriteFile):
    def __init__(self, path=None, template_filename=None, advanced_template=False):
        super().__init__(path)

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

    def run(self, file_content):
        file_name = "simulation.input"
        out = super()._run(file_content, file_name, self.get_path(full=True))
        if not (isinstance(out, str) and out.startswith("<error>")):
            self.has_file = True
        return out


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


class RunCommand(RaspaTool):
    def __init__(self, path=None):
        name = "RunCommand"
        description = """Run an arbitrary shell command in the working directory (path) of the agent.
Provide a full command line string; it will be executed with the tool's path as the current working directory (cwd)."""
        super().__init__(name, description, path)

    def run(self, command: str):
        try:
            timeout = 1200 # 20 minutes
            work_dir = self.get_path(full=True)
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
                stdout, stderr = process.communicate()
                return self.get_output(
                    e=f"Command timed out after {timeout} seconds.\nPartial stdout: {stdout[:500]}\nPartial stderr: {stderr[:500]}"
                )

            if process.returncode != 0:
                err = f"Command failed with return code {process.returncode}"
                if stderr:
                    err += f"\nError: {stderr}"
                if stdout:
                    err += f"\nStdout: {stdout}"
                return self.get_output(e=err)

            return self.get_output(content=stdout if stdout else "(No output)")

        except Exception as e:
            return self.get_output(e=f"Error executing command: {str(e)}")


