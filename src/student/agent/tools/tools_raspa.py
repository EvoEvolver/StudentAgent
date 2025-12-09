import json
import math
import os
import re
import shutil
import subprocess
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
from dotenv import load_dotenv
from mllm import Chat

from ..utils import all_files, file, quick_search
from .input_gen.molecule_loader import MoleculeLoaderTrappe
from .output import output_parser
from .tools import RaspaTool


class MoleculeLoader(MoleculeLoaderTrappe):
    def __init__(self, path=None):
        name = "molecule_loader"
        description = """Generate the molecule definition (input) files and all corresponding force field and pseudoatom definition files."""
        super().__init__(name, description, path)

        # Common molecule name mappings
        # IMPORTANT: Map to names that work with BOTH:
        # 1. TraPPE fuzzy search (can match "CO2" to "carbon dioxide")
        # 2. PubChem API (recognizes chemical formulas, NOT "carbon_dioxide" with underscore)
        self.name_mappings = {
            "oxirane": "ethylene oxide",
            "co2": "carbon dioxide",
            "carbon_dioxide": "carbon dioxide",
            "co₂": "carbon dioxide",
            "n2": "nitrogen",
            "o2": "oxygen",
            "nh3": "ammonia",
            "h2s": "hydrogen sulfide",
            "ch4": "methane",
            "c2h6": "ethane",
            "c3h8": "propane",
            "c4h10": "butane",
            "c5h12": "pentane",
            "c6h14": "hexane",
            "c7h16": "heptane",
            "c8h18": "octane",
            "c6h6": "benzene",
            "c7h8": "toluene",
        }

    def normalize_name(self, name: str) -> str:
        """Convert common chemical formulas and abbreviations to standard names."""
        # Convert to lowercase and remove spaces for matching
        normalized = name.lower().strip().replace("_", " ")

        # Check if we have a mapping for this name
        if normalized in self.name_mappings:
            return self.normalize_name(self.name_mappings[normalized])

        return normalized

    def run(self, molecule_names: Union[List[str], str]):
        self.reset()
        if isinstance(molecule_names, str):
            molecule_names = [molecule_names]

        # Normalize molecule names
        normalized_names = [self.normalize_name(name) for name in molecule_names]

        try:
            out = self._run(normalized_names)
        except Exception as e:
            return self.get_output(e=e)

        response = f"""Successfully generated the molecule input files (and force field files) for:
{', '.join([file(name) for name in out.keys()])}
(IMPORTANT: use these exact names in the simulation.input file!)"""

        torsions = [name for name in out.keys() if out[name] is True]
        if len(torsions) > 0:
            response += (
                f"\nINFO The following molecules have torsions: {', '.join(torsions)}"
            )
        else:
            response += "\nINFO None of the molecules has torsions."

        return self.get_output(content=response)


class ReadFile(RaspaTool):
    def __init__(self, path=None):
        name = "read_file"
        description = """Use this tool to read the content of a text file.
You must provide the path to the file as file name (based on the root directory NOT the current working directory).
For long documents, this tool only reads the beginning. NEVER use for RASPA output files!
"""
        super().__init__(name, description, path)

        self.blacklist = ["output/", "Output/", ".data"]

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
IMPORTANT: You must provide a file name based on the root directory NOT the current working directory. This will overwrite any existing file with the same name!
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


class InputFile(WriteFile):
    def __init__(self, path=None, template_filename=None, advanced_template=False):
        super().__init__(path)

        self.name = "input_file"
        self.description = """Use this tool to write the simulation input file. The filename is always simulation.input.
ALWAYS use the template and modify it!
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
    def __init__(self, agent, path=None):
        name = "execute_raspa"
        description = "Use this to start a RASPA simulation."
        super().__init__(name, description, path)
        self.agent = agent

    def run(self):
        self.get_run_file()
        out = self.run_raspa()
        if out and isinstance(out, tuple):
            stdout, stderr = out
            if self.check_success:
                self.agent._advance_to_next_folder()
            return self.get_output(
                content=f"The simulation ran successfully:\n<terminal_output>{out.__str__()}</terminal_output>\\n (IMPORTANT: new, empty working directory created.)"
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


class CoreMofLoader(RaspaTool):

    def __init__(self, path=None):
        name = "framework_loader"
        description = """Load the framework (MOF) file using coremof."""
        super().__init__(name, description, path)
        self.has_file = False
        self.structures: Dict[str, List[str]] = None

    def run(self, mof_name: str, output_file: str = "mol.cif"):
        import CoRE_MOF

        name = self.search_names(mof_name)
        if name is None:
            return self.get_output(e="No entry found in coremof names.")
        path = self.get_path(full=True)
        out_path = os.path.join(path, output_file)
        datasets = self.get_coremof_datasets(name)
        if datasets is None:
            return self.get_output(e=f"No dataset found for {name}")
        errors = []
        for dataset in datasets:
            try:
                mof = CoRE_MOF.get_structure(dataset, name)
                mof.to_file(out_path)
                self.has_file = True
                return self.get_output(
                    content=f"Generated from Coremof: {file(output_file)}"
                )
            except Exception as e:
                errors.append(e)
        return self.get_output(content=None, e=errors)

    def get_coremof_structures(self):
        import CoRE_MOF

        structures = defaultdict(list)
        datasets = {
            "2014": "2014",
            "2019-ASR": "2019-ASR",
            "2019-FSR": "2019-FSR",
        }  # CoRE_MOF.load.__datasets
        for dataset in datasets:
            for name in CoRE_MOF.list_structures(dataset):
                structures[name].append(dataset)
        return dict(structures)

    def get_structures(self):
        if self.structures is None:
            self.structures = self.get_coremof_structures()
        return self.structures

    def get_coremof_datasets(self, framework):
        return self.get_structures().get(framework, None)

    def structures_names(self):
        return self.get_structures().keys()

    def search_names(self, query, score_cutoff=90):
        candidates = self.structures_names()
        limit = 5

        matches = quick_search(
            query, candidates, limit=limit, score_cutoff=score_cutoff
        )

        if len(matches) == 0:
            return None

        best_match = matches[0]
        return best_match[0]


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

        out = self._parse_file(file_path)
        return self._answer(query, out)

    def _check_ignore(self, file_name):
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
            ".def",
            ".input",
            ".cif",
        ]
        for p in blacklist:
            if p in file_name:
                return True
        return False

    def _filter_files(self, files: List[str]) -> str:
        """Filter out ignored files and return a formatted string of available files."""
        filtered_files = [f for f in files if not self._check_ignore(f)]
        return "\n".join(filtered_files)

    def _correct_path(self, file_path: str, e) -> str:
        """Try to correct common mistakes in the provided file path."""
        base_path = self.get_path(full=False)
        available_files = self._filter_files(all_files(base_path))

        chat = Chat()
        chat += f"""You are helping to correct file paths for RASPA output files.
Here are all available files in the accessible directory:
<files>
{available_files}
</files>
The following file_path was provided and raised a FileNotFoundError:
<path>
{file_path}
</path>
<error>
{e}
</error>
Please find the the query based on this data. Provide only the correct file path without any additional text!
"""
        response = chat.complete()

        return response

    def _parse_file(self, file_path: str, corrected: bool = False):
        try:
            path = os.path.join(self.get_path(full=False), file_path)

            with open(path) as in_file:
                data = in_file.read()
            out = output_parser.parse(data)

            out = self.filter(out)
            out = self.strip_block_fields(out)
            return out

        except FileNotFoundError as e:
            if corrected is False:
                return self._parse_file(
                    self._correct_path(file_path, e), corrected=True
                )
            else:
                raise e
        except Exception as e:
            raise e

    def _answer(self, query, out):
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
        try:
            res = self._run(file_path, query)
        except Exception as e:
            return self.get_output(e=e)
        return self.get_output(res)


class FrameworkLoader(RaspaTool):

    def __init__(self, path=None, coremof=True, csd_path="CSD-modified/", cutoff=14.0):
        name = "framework_loader"
        description = """Load a framework file as framework.cif"""
        super().__init__(name, description, path)
        self.has_file = False
        self.output_file = "framework.cif"

        self.coremof = coremof
        self.cutoff = cutoff
        self.load_local()

        if self.coremof is True:
            self.csd_path = csd_path
            self.coremof_structures = None

    def load_coremof(self):
        import pandas as pd

        path = os.path.join(self.csd_path, "CR_data_CSD_modified_20250227.csv")
        cr = pd.read_csv(path)
        cr = cr[["coreid", "refcode", "name", "VF", "PV (cm3/g)", "Density (g/cm3)"]]
        cr[["refcode", "type"]] = cr["refcode"].str.split("_", n=2, expand=True)[[0, 1]]
        self.coremof_structures = cr

    def get_coremof_structures(self):
        if self.coremof_structures is None:
            self.load_coremof()
        return self.coremof_structures

    def find_mof_in_coremof(self, query):
        cr = self.get_coremof_structures()
        search_values = list(cr["refcode"]) + [i for i in cr["name"] if i != "-"]
        matches = quick_search(query, list(search_values))
        if len(matches) == 0:
            return None
        return matches[0][0]

    def get_cif_coremof(self, name):
        cr = self.get_coremof_structures()
        row = cr[(cr["refcode"] == name) | (cr["name"] == name)]
        index = row.index
        if len(index) == 0:
            return None
        elif len(index) == 1:
            i = index[0]
        elif len(index) > 1:
            types = {cr["type"][i]: i for i in index}
            if "FSR" in types.keys():
                i = types["FSR"]
            elif "ASR" in types.keys():
                i = types["ASR"]
            else:
                raise RuntimeError("This should not happen")
        coreid = row["coreid"][i]
        typ = row["type"][i]
        vf = row["VF"][i]
        pv = row["PV (cm3/g)"][i]
        density = row["Density (g/cm3)"][i]

        filepath = os.path.join(self.cm_path, f"cifs/CR/{typ}/{coreid}.cif")
        path_new = os.path.join(self.get_path(full=True), "framework.cif")
        shutil.copy(filepath, path_new)

        r = row[row.refcode == name]["refcode"]
        if len(r) > 0:
            return r[i], vf, pv, density
        n = row[row.name == name]["name"]
        if len(n) > 0:
            return n[i], vf, pv, density
        return None

    def load_local(self):
        load_dotenv()
        raspa_dir = os.getenv("RASPA_DIR")
        self.raspa_path = f"{raspa_dir}/share/raspa/structures/cif/"
        self.structures_local = [
            i[:-4] for i in os.listdir(self.raspa_path)
        ]  # remove .cif

    def find_mof_local(self, query):
        matches = quick_search(query, self.structures_local)
        if len(matches) == 0:
            return None
        return matches[0][0]

    def get_cif_local(self, structure):
        from PACMANCharge import pmcharge

        filepath = self.raspa_path + structure + ".cif"
        path_new = os.path.join(self.get_path(full=True), "framework.cif")
        path_new_mod = os.path.join(self.get_path(full=True), "framework_pacman.cif")

        shutil.copy(filepath, path_new)
        self.clean_cif(path_new)
        pmcharge.predict(
            cif_file=path_new,
            charge_type="DDEC6",
            digits=10,
            atom_type=True,
            neutral=True,
            keep_connect=True,
        )  # > framework_pacman.cif
        os.rename(path_new_mod, path_new)
        return structure

    def run(self, framework_name: str):
        if self.coremof is True:
            name = self.find_mof_in_coremof(framework_name)
            if name is None:
                name = self.find_mof_local(framework_name)
        else:
            name = self.find_mof_local(framework_name)

        if name is None:
            return self.get_output(e="No framework found with the given name.")
        if self.coremof:
            out = self.get_cif_coremof(name)
            if out is None:
                return self.get_output(e="Error loaded framwork from CoreMOF")
            out, vf, pv, density = out
            out = f"{out} (void fraction = {vf}, pore volume = {pv} (cm3/g), density = {density} (g/cm3))"
        else:
            out = self.get_cif_local(name)
        unit_cells = self.calculate_unit_cells(
            os.path.join(self.get_path(full=True), "framework.cif"), self.cutoff
        )
        response = f"Created framework.cif for this framework: {out} (For a cutoff of {self.cutoff} angstrom, use this or more as unit cells: {unit_cells})"
        return self.get_output(content=response)

    def clean_cif(self, file):
        with open(file, "r") as f:
            lines = f.readlines()

        cleaned_lines = [line.rstrip().rstrip(",").rstrip() + "\n" for line in lines]

        with open(file, "w") as f:
            f.writelines(cleaned_lines)

    def calculate_unit_cells(self, cif_filename, cutoff_angstrom=14.0):
        # Patterns for cell lengths
        patterns = {
            "a": re.compile(r"_cell_length_a\s+([0-9.]+)"),
            "b": re.compile(r"_cell_length_b\s+([0-9.]+)"),
            "c": re.compile(r"_cell_length_c\s+([0-9.]+)"),
            "alpha": re.compile(r"_cell_angle_alpha\s+([0-9.]+)"),
            "beta": re.compile(r"_cell_angle_beta\s+([0-9.]+)"),
            "gamma": re.compile(r"_cell_angle_gamma\s+([0-9.]+)"),
        }
        cell = {}

        with open(cif_filename, "r") as f:
            for line in f:
                for axis in patterns:
                    match = patterns[axis].match(line.strip())
                    if match:
                        cell[axis] = float(match.group(1))

        if len(cell) != 6:
            raise ValueError("Could not find all cell lengths in the CIF file.")

        # Convert angles to radians
        alpha, beta, gamma = [
            math.radians(cell["alpha"]),
            math.radians(cell["beta"]),
            math.radians(cell["gamma"]),
        ]
        a, b, c = cell["a"], cell["b"], cell["c"]

        # Build unit cell vectors
        ax, ay, az = a, 0.0, 0.0
        bx = b * math.cos(gamma)
        by = b * math.sin(gamma)
        bz = 0.0
        cx = c * math.cos(beta)
        if abs(by) < 1e-8:
            cy = 0.0
        else:
            cy = (b * c * math.cos(alpha) - bx * cx) / by
        temp = c**2 - cx**2 - cy**2
        cz = math.sqrt(temp) if temp > 0 else 0.0

        # Unit cell matrix
        A = np.array([ax, ay, az])
        B = np.array([bx, by, bz])
        C = np.array([cx, cy, cz])

        # Calculate minimum perpendicular distances (cell heights)
        Wa = np.linalg.norm(np.dot(np.cross(B, C), A)) / np.linalg.norm(np.cross(B, C))
        Wb = np.linalg.norm(np.dot(np.cross(C, A), B)) / np.linalg.norm(np.cross(C, A))
        Wc = np.linalg.norm(np.dot(np.cross(A, B), C)) / np.linalg.norm(np.cross(A, B))

        # Calculate required number of unit cells along each direction
        required_length = 2 * cutoff_angstrom
        uc_x = int(math.ceil(required_length / Wa))
        uc_y = int(math.ceil(required_length / Wb))
        uc_z = int(math.ceil(required_length / Wc))

        print(f"RASPA UnitCells: {uc_x} {uc_y} {uc_z}")
        return [uc_x, uc_y, uc_z]
