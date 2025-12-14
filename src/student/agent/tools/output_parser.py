import json
import math
import os
import re
from typing import Dict, Union, Any

import numpy as np
from mllm import Chat
from pydantic_ai import RunContext

from student.agent.tools.output import output_parser
from student.agent.tools.tools import RaspaTool

_BLOCK_RE = re.compile(r"^Block\s*\[\s*\d+\s*\]$")
_PLUSMINUS_TOKENS = {"+/-", "±", "-", "m^2/g", "m^2/cm^3", "A^2", "K", "kJ/mol", "%"}
_UNIT_TOKEN_RE = re.compile(r"^\[[^\]]+\]$")


class OutputParser(RaspaTool):
    def __init__(self, path=None):
        name = "output_parser"
        description = """Use this tool to parse the raspa output files since they are too long to read directly.
Provide the path of the output file you want to read based on the root directory (ALWAYS include the active subdirectory). Example: path=simulation_3/Output/System_0/output_Box_1.1.1_300.000000_100000.data"""
        super().__init__(name, description, path)

    def _run(self, file_path):
        path = os.path.join(self.get_path(full=False), file_path)

        try:
            with open(path) as in_file:
                data = in_file.read()
            out = output_parser.parse(data)

            out = self.filter(out)
            out = self.strip_block_fields(out)
            out = self.filter(out)

            out = json.dumps(
                out,
                separators=(",", ":"),
                ensure_ascii=False,
                default=self._json_default,
            )

        except Exception as e:
            return self.get_output(f"Error with output parsing: {e}, (path={path})")
        return out

    def run(self, file_path):
        out = self._run(file_path)
        return self.get_output(out, LIMIT=7500)

    def _json_default(self, obj):
        # Make numpy scalars serializable; fallback to str for unknowns
        try:
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
        except Exception:
            pass
        return str(obj)

    def filter(self, d: Dict) -> Dict:
        """
        Remove keys for which check_del_key(key) or check_empty_content(value) is True.
        If a value is a dict, recurse into it.
        """
        if not isinstance(d, dict):
            return d

        for key in list(d.keys()):
            value = d[key]

            if self.check_del_key(key) or self.check_empty_content(value):
                del d[key]
                continue
            if self.check_keep_key(key):
                continue

            # Recurse into containers first so we can prune after
            if isinstance(value, dict):
                self.filter(value)
                if self.check_empty_content(value):
                    del d[key]
                    continue

            elif isinstance(value, list):
                # Clean list items (recurse into dict elements)
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self.filter(item)
                # Remove empty items
                value[:] = [v for v in value if not self.check_empty_content(v)]
                # Drop the list itself if it became empty
                if not value:
                    del d[key]
                    continue

            # 3) whitelist does not protect empties; it only prevents key-based deletion
            if self.check_keep_key(key):
                continue

        return d

    def check_keep_key(self, key):
        whitelist = [
            "Total energy",
            "Average Widom Rosenbluth factor",
            "Average Henry coefficient",
        ]
        if key in whitelist:
            return True

        return False

    def check_empty_content(self, value):
        content = value
        if self.is_empty(content):
            return True
        k = "Block[0]"
        if isinstance(content, dict):
            content = value.get(k, None)
            if content is not None and self.is_empty(content):
                return True

        return False

    def is_empty(self, content):
        if content is None:
            return True

        if isinstance(content, float) and (
            content == 0 or np.isnan(content) or np.isinf(content)
        ):
            return True

        # Strings (also catch "[]"/"{}" produced by some parsers)
        if isinstance(content, str):
            s = content.strip()
            return s == "" or s == "[]" or s == "{}"

        # Floats (treat NaN/inf as empty; keep 0.0 as valid)
        if isinstance(content, float):
            return math.isnan(content) or math.isinf(content)

        if isinstance(content, (list, tuple, set)):
            if len(content) == 0:
                return True

            has_number = any(
                isinstance(x, (int, float, np.integer, np.floating))
                and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
                for x in content
            )
            if not has_number:
                if all(
                    isinstance(x, str)
                    and (
                        x.strip() in _PLUSMINUS_TOKENS
                        or _UNIT_TOKEN_RE.match(x.strip())
                    )
                    for x in content
                ):
                    return True

            # Treat lists that contain **only** "+/-" (or "±") as empty
            # e.g., ["+/-"] → empty; but [0.12, "+/-", 0.01] stays non-empty.
            # if all(isinstance(x, str) and x.strip() in _PLUSMINUS_TOKENS for x in content):
            #    return True

            # Consider empty if all elements are empty
            return all(self.is_empty(v) for v in content)
        # Dicts
        if isinstance(content, dict):
            if len(content) == 0:
                return True
            # Consider empty if all values are empty
            return all(self.is_empty(v) for v in content.values())

        try:
            c = content[0]
            return self.is_empty(c)

        except Exception:
            return False

    def check_del_key(self, key):
        if not isinstance(key, str):
            return False
        blacklist = [
            "System Properties",
            "Cpu",
            "Total CPU timings",
            "Production run CPU timings of the MC moves",
            "Production run CPU timings of the MC moves summed over all systems and components",
            "Mutual consistent basic set of units",
            "Derived units and their conversion factors",
            "Internal conversion factors",
            "Energy conversion factors",
            "VTK",
            "MoleculeDefinitions",
            "Thermo/Baro-stat NHC parameters",
            "Method and settings for electrostatics",
            "CFC-RXMC parameters",
            "Rattle parameters",
            "Spectra parameters",
            "Minimization parameters",
            "dcTST parameters",
            "Cbmc parameters",
            "Simulation",
            "Dimensions",
            "Random number seed",
            "RASPA directory set to",
            "Properties computed",
        ]
        if key in blacklist:
            return True

        for c in ["Current", "[Init]", "Compi", "OS", "Pseudo", "Forcefield"]:
            if key.startswith(c):
                return True

        else:
            return False

    def strip_block_fields(self, obj: Union[dict, list, Any]) -> Any:
        """
        Recursively remove every key that looks like 'Block[<digits>]' (allowing spaces)
        from dictionaries, anywhere in a nested structure. Non-dict/list values are
        returned unchanged.

        Parameters
        ----------
        obj : dict | list | Any
            The data structure to clean.

        Returns
        -------
        The cleaned copy, with the same overall shape as `obj`.
        """
        if isinstance(obj, dict):
            # Rebuild the dict without the unwanted keys,
            # and recurse into each value.
            return {
                k: self.strip_block_fields(v)
                for k, v in obj.items()
                if not (_BLOCK_RE.match(str(k)))
            }

        if isinstance(obj, list):
            # Recurse through lists element-wise.
            return [self.strip_block_fields(item) for item in obj]

        # Primitive value → return as-is
        return obj


class OutputExtractor(OutputParser):
    def __init__(self, path=None):
        super().__init__(path=path)

    def _run(self, file_path: str, query: str):
        path = os.path.join(self.get_path(full=False), file_path)

        try:
            with open(path) as in_file:
                data = in_file.read()
            out = output_parser.parse(data)

            out = self.filter(out)
            out = self.strip_block_fields(out)

        except Exception as e:
            return self.get_output(f"Error with output parsing: {e}, (path={path})")

        # Extract relevant information based on query using LLM
        chat = Chat()
        chat += f"""You are an expert in RASPA simulation software output analysis.
Here is the parsed output data from a RASPA simulation in JSON format:
<output>
{out}
</output>\n
Please answer the query based on this data. Provide only the specific information requested, without additional explanation.
<query>
{query}
</query>
        """
        response = chat.complete()
        return response

    def run(self, file_path: str, query: str):
        res = self._run(file_path, query)
        return self.get_output(res)


def output_extractor(ctx: RunContext, file_path: str, query: str):
    """Use this tool to parse the raspa output files since they are too long to read directly.
    Provide the path of the output file you want to read based on the root directory (ALWAYS include the active subdirectory). Example: path=simulation_3/Output/System_0/output_Box_1.1.1_300.000000_100000.data"""
    path = ctx.deps["cwd"]
    return OutputExtractor(path=path).run(file_path, query)