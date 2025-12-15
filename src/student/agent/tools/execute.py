import os
import subprocess

from dotenv import load_dotenv
from pydantic_ai import RunContext

from .tools import RaspaTool


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
                content=f"<terminal_output>{out.__str__()}</terminal_output>"
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


def execute_raspa(ctx: RunContext, simulation_name: str):
    """Execute a RASPA simulation for a simulation."""
    path = os.path.join(ctx.deps["cwd"], simulation_name)
    return ExecuteRaspa(path=path).run()


def run_command(ctx: RunContext, command: str):
    """Run an arbitrary shell command in the working directory (path) of the agent.
    Provide a full command line string; it will be executed with the tool's path as the current working directory (cwd).
    """
    try:
        timeout = 1200  # 20 minutes
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
    except Exception as e:
        return f"Error executing command: {str(e)}"
