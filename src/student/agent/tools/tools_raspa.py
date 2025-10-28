import math
import re
import shutil
import subprocess
from collections import defaultdict
from typing import Dict, Any, Union

import numpy as np
from dotenv import load_dotenv
from mllm import Chat

from .input_gen.molecule_loader import MoleculeLoaderTrappe
from .output import output_parser
from .tools import RaspaTool
from ..utils import *
from ..utils import quick_search


class MoleculeLoader(MoleculeLoaderTrappe):
    def __init__(self, path=None):
        name = "Molecule loader"
        description = """Generate the molecule definition (input) files and the corresponding force field and pseudoatoms files.

        Accepts common molecule names and chemical formulas such as:
        - Simple formulas: CO2, N2, O2, CH4, H2O, NH3, Ar, Kr, Xe, He
        - Common names: carbon dioxide, nitrogen, oxygen, methane, water, ammonia, argon, krypton, xenon, helium
        - Organic molecules: ethane, propane, butane, pentane, hexane, heptane, octane, benzene, toluene

        The tool will automatically map common abbreviations to their proper names."""
        super().__init__(name, description, path)

        # Common molecule name mappings
        # IMPORTANT: Map to names that work with BOTH:
        # 1. TraPPE fuzzy search (can match "CO2" to "carbon dioxide")
        # 2. PubChem API (recognizes chemical formulas, NOT "carbon_dioxide" with underscore)
        self.name_mappings = {
            # Small molecules - use formulas PubChem recognizes
            'co2': 'CO2',
            'co₂': 'CO2',
            'carbon_dioxide': 'CO2',
            'n2': 'N2',
            'nitrogen': 'N2',
            'o2': 'O2',
            'oxygen': 'O2',
            'nh3': 'NH3',
            'ammonia': 'NH3',
            'h2s': 'H2S',
            'hydrogen_sulfide': 'H2S',

            # Alkanes - keep as single-word names (no spaces)
            'ch4': 'methane',
            'c2h6': 'ethane',
            'c3h8': 'propane',
            'c4h10': 'butane',
            'c5h12': 'pentane',
            'c6h14': 'hexane',
            'c7h16': 'heptane',
            'c8h18': 'octane',

            # Aromatics - single-word names
            'c6h6': 'benzene',
            'c7h8': 'toluene',
        }

    def normalize_name(self, name: str) -> str:
        """Convert common chemical formulas and abbreviations to standard names."""
        # Convert to lowercase and remove spaces for matching
        normalized = name.lower().strip().replace(' ', '_')

        # Check if we have a mapping for this name
        if normalized in self.name_mappings:
            return self.name_mappings[normalized]

        # Return original name if no mapping found
        return name

    def run(self, molecule_names: List[str]):
        self.reset()
        if type(molecule_names) == str:
            molecule_names = [molecule_names]

        # Normalize molecule names
        normalized_names = [self.normalize_name(name) for name in molecule_names]

        try:
            out = self._run(normalized_names)
        except Exception as e:
            return self.get_output(e=e)
        response = f"""
        Successfully generated the molecule input files (and force field files) for:
        {', '.join([file(name) for name in out])}
        """
        return self.get_output(content=response)


class SystemAgent(RaspaTool):
    def __init__(self, path=None):
        name = "SystemAgent"
        description = """
        Use this tool to execute system tasks using natural language instructions.
        This tool can read files, write files, and execute system commands based on your query.
        Provide a natural language instruction describing what you want to accomplish.
        You MUST make the query very specific and carry all the necessary information to perform the task.
        """
        super().__init__(name, description, path)

    def run(self, query: str):
        """
        Execute a natural language query using Claude CLI.

        Args:
            query: Natural language instruction for file operations or system commands

        Returns:
            The output from Claude CLI
        """
        try:
            # Get the working directory path
            work_dir = self.get_path(full=True)

            # Execute claude command with the query
            process = subprocess.Popen(
                ['claude', '--dangerously-skip-permissions', '-p', query],
                cwd=work_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            stdout, stderr = process.communicate()

            if process.returncode != 0:
                error_msg = f"Command failed with return code {process.returncode}"
                if stderr:
                    error_msg += f"\nError: {stderr}"
                return self.get_output(e=error_msg)

            # Return the output
            return self.get_output(content=stdout)

        except FileNotFoundError:
            return self.get_output(
                e="Claude CLI not found. Please ensure 'claude' command is installed and available in PATH.")
        except Exception as e:
            return self.get_output(e=f"Error executing system agent: {str(e)}")



class WriteFile(RaspaTool):
    def __init__(self, path=None):
        name = "write_file"
        description = """
        Use this tool to write text into a new file.
        IMPORTANT: You must provide a file name based on the root directory NOT the current working directory.
        IMPORTANT: To edit a (small) file, you must first read a file with another tool and then write it completely new with this tool. Dont do this to copy files!
        IMPORTANT: This will overwrite any existing file with the same name!
        """
        super().__init__(name, description, path)

    def run(self, file_content, file_name):
        path = self.get_path(full=False)
        return self._run(file_content, file_name, path)

    def _run(self, file_content, file_name, path):
        e = None
        try:
            os.makedirs(path, exist_ok=True)
            new_path = os.path.join(path, file_name)
            with open(new_path, "w") as f:
                f.write(file_content)
        except Exception as exc:
            e = exc
        if e is None:
            return self.get_output(content=f"Successfully generated: {file(new_path)}")
        else:
            return self.get_output(e=e)


class InputFile(WriteFile):
    def __init__(self, path=None, template_filename=None):
        super().__init__(path)

        self.name = "input_file"
        self.description = """
        Use this tool to write the simulation input file.
        You must provide the content as string. The filename is always simulation.input
        ALWAYS use this template and modify based on examples from your memory!

        CRITICAL: RASPA2 uses 0-based indexing for systems and components:
        - First system: Box 0 or Framework 0 (NOT Box 1 or Framework 1)
        - First component: Component 0 (NOT Component 1)
        - Second component: Component 1, etc.
        Using 1-based indexing will cause "system number is incorrect" errors!
        """
        self.has_file = False
        if template_filename is None:
            template_filename = os.path.join(os.path.dirname(__file__), "templates/template_simulation.input")
        self.add_template(template_filename)

    def add_template(self, template_filename):
        if template_filename is None or not os.path.exists(template_filename):
            return False

        self.template_filename = template_filename
        with open(self.template_filename, 'r') as file:
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
        name = "execute raspa"
        description = """
        Use this to start a RASPA simulation. The output indicates the success of the simulation.
        """
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
                content=f"<terminal_output>{out.__str__()}</terminal_output>\\n (IMPORTANT: new, empty working directory created! To rerun, you must create all input files again!)")
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
            raise EnvironmentError("RASPA_DIR not found in .env which is required for running raspa!")

        content = f"#! /bin/sh -f\nexport RASPA_DIR={raspa_dir}\n$RASPA_DIR/bin/simulate"
        path = self.get_path(full=True)
        file_path = os.path.join(path, "run.sh")
        with open(file_path, "w") as f:
            f.write(content)
        os.chmod(file_path, 0o755)
        return

    def run_raspa(self):
        process = subprocess.Popen(
            ['bash', 'run.sh'],
            cwd=self.get_path(full=True),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out = process.communicate()
        return out


class CoreMofLoader(RaspaTool):

    def __init__(self, path=None):
        name = "framework loader"
        description = """
        Load the framework (MOF) file using coremof.
        """
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
                return self.get_output(content=f"Generated from Coremof: {file(output_file)}")
            except Exception as e:
                errors.append(e)
        return self.get_output(content=None, e=errors)

    def get_coremof_structures(self):
        import CoRE_MOF
        structures = defaultdict(list)
        datasets = {'2014': '2014', '2019-ASR': '2019-ASR', '2019-FSR': '2019-FSR'}  # CoRE_MOF.load.__datasets
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

        matches = quick_search(query, candidates, limit=limit, score_cutoff=score_cutoff)

        if len(matches) == 0:
            return None

        best_match = matches[0]
        return best_match[0]


_BLOCK_RE = re.compile(r'^Block\s*\[\s*\d+\s*\]$')


class OutputExtractor(RaspaTool):
    def __init__(self, path=None):
        name = "output_extractor"
        description = """
        Use this tool to extract information from the RASPA output files by query in natural language.
        Do not use for any .output file!
        Provide the path of the output file you want to read (based on the root directory, NOT the current working directory).
        """
        super().__init__(name, description, path)

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
        chat += f"""
        You are an expert in RASPA simulation software output analysis.
        Here is the parsed output data from a RASPA simulation in JSON format:
        <output>
        {out}
        </output>
        
        Please answer the query based on this data.
        <query>
        {query}
        </query>
        """
        response = chat.complete()
        return response

    def run(self, file_path: str, query: str):
        res = self._run(file_path, query)
        return res

    def filter(self, d: Dict) -> Dict:
        """
        Remove keys for which check_del_key(key) or check_empty_content(value) is True.
        If a value is a dict, recurse into it.
        """
        for key in list(d.keys()):
            value = d[key]

            if self.check_del_key(key) or self.check_empty_content(value):
                del d[key]
                continue
            if self.check_keep_key(key):
                continue

            if isinstance(value, dict):
                self.filter(value)

        return d

    def check_keep_key(self, key):
        whitelist = [
            'Total energy',
            'Average Widom Rosenbluth factor',
            'Average Henry coefficient',
        ]
        if key in whitelist:
            return True

        return False

    def check_empty_content(self, value):
        content = value
        if self.is_empty(content):
            return True
        k = 'Block[0]'
        if type(content) == dict:
            content = value.get(k, None)
            if self.is_empty(content):
                return True

        return False

    def is_empty(self, content):
        if type(content) == float and (content == 0 or np.isnan(content) or np.isinf(content)):
            return True
        try:
            c = content[0]
            return self.is_empty(c)
        except Exception as e:
            return False

    def check_del_key(self, key):
        if type(key) != str:
            return False
        blacklist = [
            'System Properties',
            "Cpu",
            'Total CPU timings',
            'Production run CPU timings of the MC moves',
            'Production run CPU timings of the MC moves summed over all systems and components',
            'Mutual consistent basic set of units',
            'Derived units and their conversion factors',
            'Internal conversion factors',
            'Energy conversion factors',
            'VTK', 'MoleculeDefinitions',
            'Thermo/Baro-stat NHC parameters',
            'Method and settings for electrostatics',
            'CFC-RXMC parameters',
            'Rattle parameters',
            'Spectra parameters',
            'Minimization parameters',
            'dcTST parameters',
            'Cbmc parameters',
        ]
        if key in blacklist:
            return True

        for c in ["Current", "[Init]", "Compi", "OS", "Pseudo", 'Forcefield']:
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


class FrameworkLoader(RaspaTool):

    def __init__(self, path=None, coremof=True, csd_path="CSD-modified/", cutoff=14.0):
        name = "framework loader"
        description = """
        Load a framework file as framework.cif
        """
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
        cr = cr[["coreid", "refcode", "name", "VF", 'PV (cm3/g)', 'Density (g/cm3)']]
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
        pv = row['PV (cm3/g)'][i]
        density = row['Density (g/cm3)'][i]

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
        self.structures_local = [i[:-4] for i in os.listdir(self.raspa_path)]  # remove .cif

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
        pmcharge.predict(cif_file=path_new, charge_type="DDEC6", digits=10, atom_type=True, neutral=True,
                         keep_connect=True)  # > framework_pacman.cif
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
        unit_cells = self.calculate_unit_cells(os.path.join(self.get_path(full=True), "framework.cif"), self.cutoff)
        response = f"Created framework.cif for this framework: {out} (For a cutoff of {self.cutoff} angstrom, use this or more as unit cells: {unit_cells})"
        return self.get_output(content=response)

    def clean_cif(self, file):
        with open(file, "r") as f:
            lines = f.readlines()

        cleaned_lines = [line.rstrip().rstrip(',') + '\n' for line in lines]

        with open(file, "w") as f:
            f.writelines(cleaned_lines)

    def calculate_unit_cells(self, cif_filename, cutoff_angstrom=14.0):
        # Patterns for cell lengths
        patterns = {
            'a': re.compile(r'_cell_length_a\s+([0-9.]+)'),
            'b': re.compile(r'_cell_length_b\s+([0-9.]+)'),
            'c': re.compile(r'_cell_length_c\s+([0-9.]+)'),
            'alpha': re.compile(r'_cell_angle_alpha\s+([0-9.]+)'),
            'beta': re.compile(r'_cell_angle_beta\s+([0-9.]+)'),
            'gamma': re.compile(r'_cell_angle_gamma\s+([0-9.]+)')
        }
        cell = {}

        with open(cif_filename, 'r') as f:
            for line in f:
                for axis in patterns:
                    match = patterns[axis].match(line.strip())
                    if match:
                        cell[axis] = float(match.group(1))

        if len(cell) != 6:
            raise ValueError("Could not find all cell lengths in the CIF file.")

        # Convert angles to radians
        alpha, beta, gamma = [math.radians(cell['alpha']), math.radians(cell['beta']), math.radians(cell['gamma'])]
        a, b, c = cell['a'], cell['b'], cell['c']

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
        temp = c ** 2 - cx ** 2 - cy ** 2
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


class ReportToHuman(RaspaTool):
    """Tool to generate markdown reports with datetime-based filenames."""

    def __init__(self, path=None):
        name = "report_to_human"
        description = """
        Use this tool to report to human your result in markdown when you finished or failed your task.
        """
        super().__init__(name, description, path)

    def run(self, report_content: str):
        """
        Generate a markdown report with a datetime-based filename.

        Args:
            report_content: The content of the markdown report to write

        Returns:
            Success message with the filename, or error message
        """
        try:
            from datetime import datetime

            # Generate filename with current datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = f"report_{timestamp}.md"

            # Get the full path for saving the report
            full_path = self.get_path(full=True)
            os.makedirs(full_path, exist_ok=True)

            # Full file path
            file_path = os.path.join(full_path, filename)

            # Write the report content to the file
            with open(file_path, "w") as f:
                f.write(report_content)

            result = f"Successfully generated markdown report: {filename}\nLocation: {file_path}"
            return self.get_output(content=result)

        except Exception as e:
            return self.get_output(e=f"Error generating markdown report: {str(e)}")


class AskHuman(RaspaTool):
    """Tool to ask questions to a human user via console input."""

    def __init__(self, path=None):
        name = "ask_human"
        description = """
        Use this tool when you need to ask the human user a question during execution.
        This is useful when you need clarification, additional information, or decisions from the user.
        Provide a clear question, and the tool will prompt the user for input via the console.
        """
        super().__init__(name, description, path)

    def run(self, question: str):
        """
        Ask a question to the human user and get their response.

        Args:
            question: The question to ask the user

        Returns:
            The user's response from console input
        """
        try:
            # Print the question to console
            print(f"\n[AGENT QUESTION] {question}")
            print("[Waiting for your input...]")

            # Get input from user
            user_response = input("Your answer: ").strip()

            if not user_response:
                return self.get_output(e="No response provided by user.")

            result = f"Question: {question}\nUser's answer: {user_response}"
            return self.get_output(content=result)

        except EOFError:
            return self.get_output(e="Input stream closed. Cannot read from console.")
        except Exception as e:
            return self.get_output(e=f"Error getting user input: {str(e)}")


class ImageQuestionTool(RaspaTool):
    """Tool to ask questions about images using OpenAI's vision API."""

    def __init__(self, path=None):
        name = "ask_image_question"
        description = """
        Ask a question about an image using AI vision capabilities.
        Provide a query (question) and the path to an image file.
        Supported formats: JPG, JPEG, PNG, GIF, WebP.
        The tool will analyze the image and return an answer to your question.
        """
        super().__init__(name, description, path)
        self._init_vision_client()

    def _init_vision_client(self):
        """Initialize OpenAI client for vision API."""
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.vision_model = "gpt-4o"
        except ImportError:
            self.client = None
            print("Warning: OpenAI package not found. Image question tool will not work.")

    def run(self, query: str, image_path: str):
        """
        Ask a question about an image.

        Args:
            query: The question to ask about the image
            image_path: Path to the image file (relative to working directory or absolute)

        Returns:
            The answer from the vision model
        """
        if self.client is None:
            return self.get_output(e="OpenAI client not initialized. Please install openai package.")

        try:
            import base64

            # Handle both absolute and relative paths
            if not os.path.isabs(image_path):
                # Try relative to full path first
                full_image_path = os.path.join(self.get_path(full=True), image_path)
                if not os.path.exists(full_image_path):
                    # Try relative to base path
                    full_image_path = os.path.join(self.get_path(full=False), image_path)
                    if not os.path.exists(full_image_path):
                        # Try as-is
                        full_image_path = image_path
            else:
                full_image_path = image_path

            # Validate image exists
            if not os.path.exists(full_image_path):
                return self.get_output(e=f"Image file not found: {full_image_path}")

            # Read and encode image
            with open(full_image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # Determine image format from file extension
            ext = os.path.splitext(full_image_path)[1].lower()
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }

            if ext not in mime_types:
                return self.get_output(e=f"Unsupported image format: {ext}. Supported: {list(mime_types.keys())}")

            mime_type = mime_types[ext]

            # Call OpenAI vision API
            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": query
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            answer = response.choices[0].message.content.strip()
            result = f"Question: {query}\nImage: {image_path}\n\nAnswer: {answer}"
            return self.get_output(content=result)

        except Exception as e:
            return self.get_output(e=f"Error analyzing image: {str(e)}")
