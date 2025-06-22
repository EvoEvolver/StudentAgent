import shutil
import os
import numpy as np
import subprocess
from typing import List, Dict, Set, Any, Union
from collections import defaultdict
import re

from dotenv import load_dotenv
import CoRE_MOF
from PACMANCharge import pmcharge

from .tools import Tool, RaspaTool
from ..utils import *

from .input_gen.molecule_loader import MoleculeLoaderTrappe
from .output import output_parser
from ..utils import quick_search

from .input_gen.generate_mol_definition import generate_molecule_def


class MoleculeLoader(MoleculeLoaderTrappe):
    def __init__(self, path=None):
        name = "Molecule loader"
        description = "Generate the molecule definition (input) files and the corresponding force field and pseudoatoms files."
        super().__init__(name, description, path)

    def run(self, molecule_names : List[str]):
        self.reset()
        if type(molecule_names) == str:
            molecule_names = [molecule_names]

        try:
            out = self._run(molecule_names)
        except Exception as e:
            raise e
            # return self.get_output(e=e)
        return self.get_output(filenames=out)


    def get_output(self, filenames=None, e=None):
        if filenames is not None:
            response = f"""
            Successfully generated the molecule input files (and force field files) for: 
            {''.join([file(name) for name in filenames])}
            """
            return tool_response(self.name, response)
        else:
            return tool_response(self.name, error(e))


class InspectFiles(RaspaTool):
    def __init__(self, path=None):
        name = "inspect_files"
        description = """
        Use this tool to get all the files that you can access.
        """
        super().__init__(name, description, path)
    
    def run(self):
        '''
        Return files hierarchy starting from self.get_path()
        '''
        path = self.get_path(full=True)
        
        files : List[str] = all_files(path)
        return tool_response(self.name, files)
    
    
class ReadFile(RaspaTool):
    def __init__(self, path=None):
        name = "read_file"
        description = """
        Use this tool to read the content of a text file.
        You must provide the path to the file.
        """
        super().__init__(name, description, path)
        
    def run(self, file_name):
        path = self.get_path(full=True)
        
        content = None
        file_name = os.path.join(path, file_name)
        
        if os.path.exists(file_name):
        
            with open(file_name, "r") as f:
                content = f.read()

        return self.get_output(file_name, content)


    def get_output(self, file_name, content=None):
        if content is not None:
            return tool_response(self.name, file(file_name, content))
        else:
            return error("File not found {file_name}")
            
            

class WriteFile(RaspaTool):
    def __init__(self, path=None):
        name = "write_file"
        description = """
        Use this tool to write text into a new file.
        You must provide a file name and its content string.
        """
        super().__init__(name, description, path)
        
    def run(self, file_content, file_name):
        path = self.get_path(full=True)
        e = None
        try:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, file_name), "w") as f:
                f.write(file_content)
        except Exception as e:
            pass
        return self.get_output(file_name, e)

    def get_output(self, file_name, e=None):
        if e is None:
            return tool_response(self.name, file(file_name))
        else:
            return error(e)
            

class InputFile(WriteFile):
    def __init__(self, path=None):
        super().__init__(path=path)
        self.name = "input_file"
        self.description = """
        Use this tool to write the simulation input file.
        You must provide the content as string. The filename is always simulation.input
        ALWAYS use a template or example as reference
        """
        self.has_file = False

    def run(self, file_content):
        file_name="simulation.input"
        out = super().run(file_content, file_name)

        if not out.startswith("<error>"):
            self.has_file = True
        return tool_response(self.name, out)


class ExecuteRaspa(RaspaTool):
    def __init__(self, path=None):
        name = "execute raspa"
        description = """
        Use this to start a RASPA simulation. The output indicates the success of the simulation.
        """
        super().__init__(name, description, path)
        

    def run(self):
        self.get_run_file()
        out = self.run_raspa()
        return self.get_output(out)
    
    def get_output(self, out):
        # TODO
        return tool_response(self.name, out)

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


class TrappeLoader(RaspaTool):

    def __init__(self, path=None):
        name = "molecule definition generator"
        description = """
        Load the molecule data using Trappe and generate the molecule definition files and corresponding force field files..
        """
        super().__init__(name, description, path)
        self.has_file = False
        self.molecules = self.load_molecule_names()
    

    def run(self, molecule_names : List[str]):
        if type(molecule_names) == str:
            molecule_names = [molecule_names]

        molecule_names = [name.replace(" ", "_") for name in molecule_names]
        
        res = self.search_names(molecule_names)
        ids = [self.get_molecule_id(name) for name in res]
        if len(ids) == 0:
            return self.get_output(e="No corresponding molecules found. Try a different name!")

        out_path = self.get_path(full=True)

        try:
            filenames = generate_molecule_def(molecule_ids=ids, names=molecule_names, output_dir=out_path)
            self.has_file = True
        except Exception as e:
            #echo(f"There was some error with the molecule file generation: {e}")
            return self.get_output(e=e)
        
        return self.get_output(filenames=filenames)
    
    
    def get_output(self, filenames=None, e=None):
        if filenames is not None:
            response = f"""
            Successfully generated the molecule input files (and force field files) for: 
            {''.join([file(name) for name in filenames])}
            """
            return tool_response(self.name, response)
        else:
            return error(e)


    def _load_trappe_names(self):
        # URL to scrape
        url = "http://trappe.oit.umn.edu/scripts/search_select.php"
        # check if the data is already downloaded
        path = self.get_path(full=False)
        file_path = os.path.join(path, "trappe_molecule_list.json")
        try:
            with open(file_path) as f:
                return json.load(f)
        except FileNotFoundError:
            pass
        os.makedirs(path, exist_ok=True)
        res_dict = json.loads(request_by_post(url))['search']
        
        with open(file_path, "w") as f:
            json.dump(res_dict, f)

        return res_dict
    
    def load_molecule_names(self, families=["UA", "small"]):
        mols = self._load_trappe_names()
        molecules = {}
        for m in mols:
            if m['family'] in families:
                name = m["name"].replace("<em>", "").replace("</em>", "")
                molecules[name] =  m["molecule_ID"] 
        return molecules

    def get_molecule_id(self, mol):
        return self.molecules.get(mol, None)

    def molecule_names(self):
        return self.molecules.keys()
    
    def _search_name(self, query, score_cutoff=90):
        candidates = self.molecule_names()
        matches = quick_search(query, candidates, limit=5, score_cutoff=score_cutoff)
        
        if len(matches) == 0:
            return None
        best_match = matches[0]
        return best_match[0]

    def search_names(self, names, score_curoff = 90):
        res = []
        for name in names:
            res.append(self._search_name(name, score_curoff))
        return res
    
    def init_memory_prompt(self):
        prompt = f""""
        This is a list of molecule names you might want to use for {tool(self.name)} but which can only be found with alternative names:
        {mol_name("carbon dioxide", ["CO2", "carbon", "co2", "co", "carbon oxide"])}
        {mol_name("nitrogen", ["N2", "dinitrogen"])}
        """
        return prompt
    

class CoreMofLoader(RaspaTool):
    
    def __init__(self, path=None):
        name = "framework loader"
        description = """
        Load the framework (MOF) file using coremof.
        """
        super().__init__(name, description, path)
        self.has_file = False
        self.structures : Dict[str, List[str]] = self.get_coremof_structures()

    def run(self, mof_name : str, output_file : str = "mol.cif"):

        name = self.search_names(mof_name)
        if name is None:
            return self.get_output("", e="No entry found in coremof names.")
        path = self.get_path(full=True)
        out_path = os.path.join(path, output_file)
        datasets = self.get_coremof_datasets(name)
        if datasets is None:
            return self.get_output(output_file, e=f"<error>No dataset found for {name}</error>")
        
        errors=[]

        for dataset in datasets:
            try:
                mof = CoRE_MOF.get_structure(dataset, name)
                mof.to_file(out_path)
                self.has_file = True
                return self.get_output(output_file)
            
            except Exception as e:
                errors.append(e)
                
        return self.get_output(output_file, errors)
        

    def get_output(self, file_name, e=None):
        if e is None:
            return tool_response(self.name, file(file_name))
        else:
            return error(e)


    def get_coremof_structures(self):
        structures = defaultdict(list)
        datasets = {'2014': '2014', '2019-ASR': '2019-ASR', '2019-FSR': '2019-FSR'} # CoRE_MOF.load.__datasets
        for dataset in datasets:
            for name in CoRE_MOF.list_structures(dataset):
                structures[name].append(dataset)
        return dict(structures)


    def get_coremof_datasets(self, framework):
        return self.structures.get(framework, None)
    

    def structures_names(self):
        return self.structures.keys()
    

    def search_names(self, query, score_cutoff=90):
        candidates = self.structures_names()
        limit = 5

        matches = quick_search(query, candidates, limit=limit, score_cutoff=score_cutoff)
        
        if len(matches) == 0:
            return None
            
        best_match = matches[0]
        return best_match[0]
    

_BLOCK_RE = re.compile(r'^Block\s*\[\s*\d+\s*\]$')


class OutputParser(RaspaTool):
    def __init__(self, path=None):
        name = "output_parser"
        description = """
        Use this tool to parse the raspa output files since they are too long to read directly.
        Provide the path of the output file you want to read.
        """
        super().__init__(name, description, path)
    
    def run(self, file_path):
        path = os.path.join(self.get_path(full=True), file_path)
        
        try:
            with open(path) as in_file:
                data = in_file.read()
            out = output_parser.parse(data)
            
            out = self.filter(out)
            out = self.strip_block_fields(out)
            
        except Exception as e:
            return error(f"Error with output parsing: {e}, (path={path})")
        return tool_response(self.name, out.__str__(), LIMIT=7500)
    

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
            
            if isinstance(value, dict):
                self.filter(value)

        return d


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
            'Properties computed',
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
    
    def __init__(self, path=None, coremof=True, csd_path="CSD-modified/"):
        name = "framework loader"
        description = """
        Load a framework file as framework.cif
        """
        super().__init__(name, description, path)
        self.has_file = False
        self.output_file = "framework.cif"
        
        self.coremof = coremof
        self.load_local()
        
        if self.coremof is True:
            self.csd_path = csd_path
            self.load_coremof()

    def load_coremof(self):
        import pandas as pd
        path = os.path.join(self.csd_path, "CR_data_CSD_modified_20250227.csv")
        cr = pd.read_csv(path)
        cr = cr[["coreid", "refcode", "name"]]
        cr[["refcode", "type"]] = cr["refcode"].str.split("_", n=2, expand=True)[[0, 1]]
        self.coremof_structures = cr

    def find_mof_in_coremof(self, query):
        cr = self.coremof_structures
        search_values = list(cr["refcode"]) + [i for i in cr["name"] if i != "-"]
        matches = quick_search(query, list(search_values))
        if len(matches) == 0:
            return None
        return matches[0][0]

    def get_cif_coremof(self, name):
        cr = self.coremof_structures
        row = cr[(cr["refcode"] == name) | (cr["name"] == name)]
        index = row.index
        if len(index) == 0:
            return None
        elif len(index) == 1:
            i = index[0]
        elif len(index) > 1:
            types = {cr["type"][i] : i for i in index}
            if "FSR" in types.keys():
                i = types["FSR"]
            elif "ASR" in types.keys():
                i = types["ASR"]
            else:
                raise RuntimeError("This should not happen")
        coreid = row["coreid"][i]
        typ = row["type"][i]
        
        filepath = os.path.join(self.cm_path, f"cifs/CR/{typ}/{coreid}.cif")
        path_new = os.path.join(self.get_path(full=True), "framework.cif")
        shutil.copy(filepath, path_new)
        
        r = row[row.refcode == name]["refcode"]
        if len(r) > 0:
            return r[i]
        n = row[row.name == name]["name"]
        if len(n) > 0:
            return n[i]
    

    def load_local(self):
        load_dotenv()
        raspa_dir = os.getenv("RASPA_DIR")
        self.raspa_path = f"{raspa_dir}/share/raspa/structures/cif/"
        self.structures_local = [i[:-4] for i in os.listdir(self.raspa_path)] # remove .cif

    def find_mof_local(self, query):
        matches = quick_search(query, self.structures_local)
        if len(matches) == 0:
            return None
        return matches[0][0]

    def get_cif_local(self, structure):
        filepath = self.raspa_path+structure+".cif"
        path_new = os.path.join(self.get_path(full=True), "framework.cif")
        path_new_mod = os.path.join(self.get_path(full=True), "framework_pacman.cif")

        shutil.copy(filepath, path_new)
        self.clean_cif(path_new)
        pmcharge.predict(cif_file=path_new,charge_type="DDEC6",digits=10,atom_type=True,neutral=True,keep_connect=True) # > framework_pacman.cif
        os.rename(path_new_mod, path_new)
        return structure


    def run(self, framework_name):
        if self.coremof is True:
            name = self.find_mof_in_coremof(framework_name)
            if name is None:
                name = self.find_mof_local(framework_name)
        else:
            name = self.find_mof_local(framework_name)

        if name is None:
            return self.get_output("", e="No framework found with the given name.")
        
        if self.coremof:
            out = self.get_cif_coremof(name)
        else:
            out = self.get_cif_local(name)
        return self.get_output(out)

    def get_output(self, name, e=None):
        if e is None:
            return tool_response(self.name, f"Created framework.cif for this framework: {name}")
        else:
            return error(e)

    def clean_cif(self, file):
        with open(file, "r") as f:
            lines = f.readlines()

        cleaned_lines = [line.rstrip().rstrip(',') + '\n' for line in lines]

        with open(file, "w") as f:
            f.writelines(cleaned_lines)