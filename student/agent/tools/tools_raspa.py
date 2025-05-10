import os
import numpy as np
import subprocess
from typing import List, Dict, Set
from collections import defaultdict

from dotenv import load_dotenv
import CoRE_MOF

from .tools import Tool
from ..utils import *
from .input_gen.generate_mol_definition import generate_molecule_def
from .output import output_parser


class RaspaTool(Tool):
    def __init__(self, name, description, path=None):
        super().__init__(name, description)
        self.path = path
    
    def get_path(self):
        if self.path is  None:
            #raise RuntimeWarning(f"No path was set for {self.name}.")
            print(f"Warning: No path was set for {self.name}!")
            return "./"
        else:
            return self.path


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
        path = self.get_path()
        
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
        path = self.get_path()
        
        content = None
        file_name = os.path.join(path, file_name)
        
        if os.path.exists(file_name):
        
            with open(file_name, "r") as f:
                content = f.read()

        return self.get_output(file_name, content)


    def get_output(self, file_name, content=None):
        if content is None:
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
        path = self.get_path()
        error = None
        try:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, file_name), "w") as f:
                f.write(file_content)
        except Exception as e:
            error = e
        return self.get_output(file_name, error)

    def get_output(self, file_name, error=None):
        if error is None:
            return tool_response(self.name, file(file_name))
        else:
            return error(error)
            

class InputFile(WriteFile):
    def __init__(self, path=None):
        super().__init__(path=path)
        self.name = "input_file"
        self.description = """
        Use this tool to write the simulation input file.
        You must provide the content as string. The filename is always simulation.input
        """
        self.has_file = False

    def run(self, file_content):
        file_name="simulation.input"
        out = super.run(file_content, file_name)

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
        path = self.get_path()
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
        path = self.get_path()
        file_path = os.path.join(path, "run.sh")
        with open(file_path, "w") as f:
            f.write(content)
        os.chmod(file_path, 0o755)
        return
    
    def run_raspa(self):
        process = subprocess.Popen(
            ['bash', 'run.sh'],
            cwd=self.get_path(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out = process.communicate()
        return out


class TrappeLoader(RaspaTool):

    def __init__(self, path=None):
        name = "molecule loader"
        description = """
        Load the molecule data using Trappe.
        """
        super().__init__(name, description, path)
        self.has_file = False
        self.molecules = self.load_molecule_names()
    

    def run(self, molecule_names : List[str]):
        molecule_names = [name.replace(" ", "_") for name in molecule_names]
        
        res = self.search_names(molecule_names)
        ids = [self.get_molecule_id(name) for name in res]

        out_path = self.get_path()

        try:
            filenames = generate_molecule_def(molecule_ids=ids, names=molecule_names, output_dir=out_path)
            self.has_file = True
        except Exception as e:
            #echo(f"There was some error with the molecule file generation: {e}")
            return self.get_output(error=e)
        
        return self.get_output(filenames=filenames)
    
    
    def get_output(self, filenames=None, error=None):
        if filenames is not None:
            response = f"""
            Successfully generated the molecule input files (and force field files) for: 
            {''.join([file(name) for name in filenames])}
            """
            return tool_response(self.name, response)
        else:
            return error(error)


    def _load_trappe_names(self):
        # URL to scrape
        url = "http://trappe.oit.umn.edu/scripts/search_select.php"
        # check if the data is already downloaded
        file_path = os.path.join(self.get_path(), "trappe_molecule_list.json")
        try:
            with open(file_path) as f:
                return json.load(f)
        except FileNotFoundError:
            pass

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
            return self.get_output("", error="No entry found in coremof names.")
        path = self.get_path()
        out_path = os.path.join(path, output_file)
        datasets = self.get_coremof_datasets(name)
        if datasets is None:
            return self.get_output(output_file, error=f"<error>No dataset found for {name}</error>")
        
        errors=[]

        for dataset in datasets:
            try:
                mof = CoRE_MOF.get_structure(dataset, mof_name)
                mof.to_file(out_path)
                self.has_file = True
                return self.get_output(output_file)
            
            except Exception as e:
                errors.append(e)
                
        return self.get_output(self, output_file, errors)
        

    def get_output(self, file_name, error=None):
        if error is None:
            return tool_response(self.name, file(file_name))
        else:
            return error(error)


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
        matches = quick_search(query, candidates, limit=5, score_cutoff=score_cutoff)
        
        if len(matches) == 0:
            return None
        best_match = matches[0]
        return best_match[0]
    
    '''
    def init_memory_prompt(self):
        prompt = f""""
        This is the list of all names for frameworks/systems/MOFs, you can load with {tool("coremof")}:"
        <framework names>{self.structures.keys()}</framework names>
        """
        return prompt
    '''
    

class OutputParser(RaspaTool):
    def __init__(self, path=None):
        name = "output_parser"
        description = """
        Use this tool to parse the raspa output files since they are too long to read directly.
        Provide the path of the output file you want to read.
        """
        super().__init__(name, description, path)
    
    def run(self, file_path):
        path = os.path.join(self.get_path(), file_path)
        
        try:
            with open(path) as in_file:
                data = in_file.read()
            out = output_parser.parse(data)
            
            out = self.filter(out)
            return out
            
        except Exception as e:
            raise e
            return error(f"Error with output parsing: {e}")
        return tool_response(self.name, out)
    

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
            'Simulation',
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
    
    