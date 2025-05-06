from .tools import Tool
import os
from typing import List, Dict, Set

class WriteFile(Tool):
    def __init__(self):
        name = "write_file"
        description = """
        Use this tool to write text into a new file.
        You must provide a file name and its content string.
        """
        super().__init__(name, description)
        

    def run(self, file_content, path, file_name):
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
            return f"<file>{file_name}</file>"
        else:
            return f"<error>{error}</error>"
            


class ReadFile(Tool):
    def __init__(self):
        name = "read_file"
        description = """
        Use this tool to read the content of a text file.
        You must provide the path to the file.
        """
        super().__init__(name, description)
        

    def run(self, file_path):
        content = None

        if os.path.exists(file_path):
        
            with open(file_path, "r") as f:
                content = f.read()

        return self.get_output(file_path, content)


    def get_output(self, file_path, content=None):
        if content is None:
            return f"<file>File not found {file_path}.</file>"
        else:
            return f"<file filename = {file_path}>{content}</file>"
            


class ExecuteRaspa(Tool):
    def __init__(self):
        name = "execute raspa"
        description = """
        Use this to start a RASPA simulation. The output indicates the success of the simulation.
        """
        super().__init__(name, description)

    def run(self):
        return self.get_output()
    
    def get_output(self):
        return super().get_output()


class TrappeLoader(Tool):
    def __init__(self, memory, path):
        name = "molecule loader"
        description = """
        Load the molecule data using Trappe.
        """
        super().__init__(name, description)

        self.memory = memory
        self.path = path
    
    def run(self, molecule_names : List[str]):
        molecule_names = [name.replace(" ", "_") for name in molecule_names]
        
        res = self.memory.search(molecule_names, top_k=len(molecule_names))

        names_i = [i.content for i in res]
        names = [name.replace(" ", "_") for name in names_i]

        ids = [i.data['molecule_ID'] for i in res]
        try:
            from student.input_gen.generate_mol_definition import generate_molecule_def
            generate_molecule_def(molecule_ids=ids, names=names, output_dir=self.path)

            #echo(f"Generated molecule.def file for '{', '.join(names)}' from Trappe Data")
        except Exception as e:
            #echo(f"There was some error with the molecule file generation: {e}")
            return self.get_output(error=e)
        
        return self.get_output(names=names)
    
    def get_output(self, names=None, error=None):
        if names is not None:
            return f"<molecule names>{names}</molecule names>"
        else:
            return f"<error>{error}</error>"



class CoreMofLoader(Tool):
    
    def __init__(self, memory, path):
        name = "framework loader"
        description = """
        Load the framework (MOF) file using coremof.
        """
        super().__init__(name, description)
        self.memory = memory
        self.path = path

    def run(self, framework : str, output_file : str = "mol.cif"):
        

        res = self.memory.search([framework], top_k=1)

        if len(res) ==  0:
            #echo("No corresponding molecule found in the coremof database.")
            return self.output(error="No corresponding entry found in the coremof database")
        #elif res[0].data['name'] == "box":
        #    return "box"

        try:
            import CoRE_MOF
            mof_name = framework.data['name']
            dataset = framework.data["datasets"][-1]
            mof = CoRE_MOF.get_structure(dataset, mof_name)
            mof.to_file(output_file)

            return self.get_output(self, output_file)
        
        except Exception as e:
            return self.get_output(self, output_file, e)
        

    def get_output(self, file_name, error=None):
        if error is None:
            return f"<file>{file_name}</file>"
        else:
            return f"<error>{error}</error>"



