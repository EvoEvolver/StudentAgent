import json
import os

from pydantic_ai import Agent, RunContext, Tool

from .tools.framework_loader import FrameworkLoader
from .tools.molecule_loader import molecule_loader
from .tools.output_parser import output_extractor
from .tools.tools_raspa import execute_raspa, make_input_file, run_command
from .tools.tools_system import report_to_human, ask_human


class RaspaAgent:
    path: str

    def make_tools(self):
        if self.csd_path is None:
            print("A CSD path is required to access the coremof files.")

        def framework_loader(ctx: RunContext, framework_name: str):
            """Load a framework file as framework.cif in the agent's working directory."""
            path = ctx.deps["cwd"]
            return FrameworkLoader(path=path, coremof=False, csd_path=self.csd_path).run(framework_name)

        tool_list = [
            framework_loader,
            molecule_loader,
            execute_raspa,
            make_input_file,
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

    def get_file_message(self):
        """Return a formatted overview of files/folders up to depth 3.

        Produces a readable tree (directories first) excluding ignored paths.
        """
        root = self.path
        max_depth = 3

        if not os.path.exists(root):
            return f"\n\n<file_overview>\nTree:\n(NOT FOUND)\n</file_overview>\n"

        def list_children(base_path: str, base_root: str, current_depth: int):
            lines = []
            try:
                entries = sorted(os.listdir(base_path))
            except Exception:
                return lines

            # separate dirs and files; skip ignored
            dirs = []
            files = []
            for name in entries:
                full = os.path.join(base_path, name)
                rel = os.path.relpath(full, start=base_root)
                if check_ignore(rel):
                    continue
                if os.path.isdir(full):
                    dirs.append((name, full, rel))
                else:
                    files.append((name, full, rel))

            # list directories first
            for name, full, rel in dirs:
                item_depth = current_depth + 1
                if item_depth <= max_depth:
                    indent = "  " * (item_depth - 1)
                    lines.append(f"{indent}- {name}/")
                    # only descend if we haven't reached max depth
                    if item_depth < max_depth:
                        lines.extend(list_children(full, base_root, item_depth))

            # then files
            for name, full, rel in files:
                item_depth = current_depth + 1
                if item_depth <= max_depth:
                    indent = "  " * (item_depth - 1)
                    lines.append(f"{indent}- {name}")

            return lines

        tree_lines = list_children(root, root, 0)
        tree_formatted = "\n".join(tree_lines) if tree_lines else "(empty)"

        # Nicely formatted overview block
        overview = (
            f"\n\n<file_overview>\n"
            f"Tree:\n{tree_formatted}\n"
            f"</file_overview>\n"
        )
        return overview

    def run(self, query: str):
        current_file_overview = self.get_file_message()
        agent = Agent(
            tools=self.make_tools(),
            system_prompt=system_prompt_v2 + current_file_overview,
            model=self.model_name
        )
        result = agent.run_sync(query, deps={"cwd": self.path})
        print(result.output)
        return result.output


def check_ignore(file_name):
    # Return True is file should be ignored
    blacklist = [
        "Movies/",
        "VTK/",
        "Restart/",
        "run.sh",
        ".DS_Store",
        ".md",
        ".json",
        ".jsonl",
        ".log",
    ]
    for p in blacklist:
        if p in file_name:
            return True
    return False


system_prompt_v2 = """
You specialize to assist with RASPA simulations.
You are equipped with tools to handle RASPA's input and output.
"""
