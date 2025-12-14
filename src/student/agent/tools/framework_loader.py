import math
import os
import re
import shutil

import numpy as np
from dotenv import load_dotenv
from pydantic_ai import RunContext

from student.agent.tools.tools import RaspaTool
from student.agent.utils import quick_search


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


def framework_loader(ctx: RunContext, framework_name: str):
    """Load a framework file as framework.cif"""
    path = ctx.deps["cwd"]
    return FrameworkLoader(path=path).run(framework_name)