from student.agent.tools.tools import RaspaTool
from student.agent.tools.input_gen.data.unknown_trappe import *

import os
import json
import requests
from typing import *
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from io import StringIO

from student.agent.utils import quick_search, request_by_post
from .utils_molecules import get_mol
#from .generate_mol_definition import *

from .trappe_loader import download_parameters, download_properties
from .pseudoatoms import PseudoAtoms, Atom, PseudoAtomsBag

PATH = os.path.dirname(__file__)

def get_trappe_properties(molecule_id: int):
    """
    Returns: critical constants: Temperature [T] in Kelvin, Pressure [Pa], and Acentric factor [-]
    """
    molecule_id = int(molecule_id)
    # Incomplete property files for some small molecules are handled separately:
    if molecule_id == 119: # nitrogen
        return (126.192, 3395800.0, 0.0372)
    if molecule_id in [120, 117]: # no trappe data available for the properties!
        return None

    df = pd.read_csv(StringIO(download_properties(molecule_id)), skiprows=1)

    crit_rows = df[df['T [K]'] == "crit_T [K]"]
    if not crit_rows.empty:
        try:
            # Assuming the CSV stores the critical temperature in the 'dens_liq [g/ml]' column for this row
            Tc = float(crit_rows.iloc[0]['dens_liq [g/ml]'])
        except Exception as e:
            raise ValueError("Error converting critical temperature to float: " + str(e))
    else:
        raise ValueError("crit_T [K] not found in CSV data")
    
    df = df[df['T [K]'] != "crit_T [K]"]
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    
    T = df['T [K]'].values
    P = df['P_vap [kPa]'].values
    
    lnp = np.log(P)
    invT = 1 / T

    def linear_model(x, a, b):
        return a * x + b
    
    params, _ = curve_fit(linear_model, invT, lnp)
    slope, intercept = params

    # Use the model to calculate the critical pressure at T = Tc (x = 1/Tc)
    ln_pc = linear_model(1 / Tc, slope, intercept)
    pc = np.exp(ln_pc) * 1000  # Convert from kPa to Pa

    # Calculate the vapor pressure at T = 0.7 * Tc (x = 1/(0.7*Tc))
    ln_pr = linear_model(1 / (0.7 * Tc), slope, intercept)
    pr = np.exp(ln_pr) * 1000  # Convert from kPa to Pa

    # Calculate the acentric factor:
    w = -np.log10(pr / pc) - 1

    return Tc, float(pc), float(w)


class MoleculeLoaderTrappe(RaspaTool):

    def __init__(self, name, description, path=None):
        super().__init__(name, description, path)
        self.molecules = self.load_molecule_names()
        self.ps_bag = PseudoAtomsBag()
        self.blacklist = ["methyl acetate", "ethyl acetate", "methyl propionate", "vinyl acetate"]
        self.make_files = True
        self.verbose = False

    def reset(self):
        self.ps_bag = PseudoAtomsBag()

    def _run(self, molecule_names : List[str]):
        molecule_names = [name.replace(" ", "_") for name in molecule_names]
        out_names = []
        for name in molecule_names:

            res = self._search_name(name)
            
            if res is not None and molecule_name_to_smiles(res) == molecule_name_to_smiles(name):
                id = self.get_molecule_id(res)
                mol_def = self.build_molecule_definition(id, res)
            else:
                res = name
                mol_def = self.load_unknown_molecule(name)
                
            self.make_file(mol_def, f"{res}.def")
            out_names.append(res)
        self.make_ff_ps_files()
        
        return out_names
    
    def make_ff_ps_files(self):
        self.make_file(self.ps_bag.build_pseudoatoms(), "pseudo_atoms.def")
        self.make_file(self.ps_bag.build_ff_mixing(), "force_field_mixing_rules.def")
        self.make_file(self.ps_bag.build_ff(), "force_field.def")
        return
    
    ################################################################################
    ################################      Utils          ###########################
    ################################################################################

    def make_file(self, content, file_name):
        if self.make_files is False:
            pass
        if type(content) == list:
            content = "\n".join(content)

        output_dir = self.get_path(full=True)
        with open(os.path.join(output_dir, file_name), "w") as f:
            f.write(content)

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
                if name.startswith("n-"):
                    name = name[2:]
                molecules[name] =  m["molecule_ID"] 
        return molecules

    def get_molecule_id(self, mol):
        return self.molecules.get(mol, None)

    def molecule_names(self):
        return self.molecules.keys()
    
    def _search_name(self, query, score_cutoff=80):
        candidates = self.molecule_names()
        matches = quick_search(query, candidates, limit=5, score_cutoff=score_cutoff)
        matches = [i for i in matches if i not in self.blacklist]

        if len(matches) == 0:
            return None
        best_match = matches[0]

        return best_match[0]
    
    def parse_section(self, param_str: str, section_key: str, min_parts: int) -> list:
        lines = param_str.splitlines()
        section_lines = []
        in_section = False
        for line in lines:
            if not line.strip():
                continue
            if line.startswith(section_key):
                in_section = True
                continue
            if in_section:
                if line.startswith("#,"):
                    break  # end of this section
                line = line.replace("(cyc,", "(cyc")
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= min_parts:
                    section_lines.append(parts[:min_parts])
        return section_lines
    


    def load_unknown_molecule(self, name):
        
        smiles = molecule_name_to_smiles(name)
        Tc, pc, acentric_factor = get_properties(smiles)


        with open(os.path.join(PATH,"data/parameters/ps_type_to_smarts.pkl"), "rb") as f:
            type_to_smarts = pickle.load(f)

        with open(os.path.join(PATH,"data/parameters/param_types.pkl"), "rb") as f:
            param_types = pickle.load(f)

        with open(os.path.join(PATH,"data/parameters/bonded.pkl"), "rb") as f:
            bonded = pickle.load(f)
        
        mol = get_mol(name)

        if mol is None:
            return None
        
        mol = assign_atom_types_by_smarts(mol, type_to_smarts)
        
        ps = PseudoAtoms()
        ps.parse_mol(mol, param_types)
        self.ps_bag.add(name, ps)

        bonds, bends, torsions = assign_bonded_interactions(mol, bonded)

        bonded = bonded_definitions(bonds, bends, torsions, bonded)
        interactions = self.get_intramol_interactions(mol)
        n_vdw = len(interactions['vdw'])
        n_coulomb = len(interactions['coulomb'])


        lines = []
        
        # Critical constants section
        lines.append("# critical constants: Temperature [T], Pressure [Pa], and Acentric factor [-]")
        lines.append(f"{Tc}")
        lines.append(f"{pc}")
        lines.append(f"{acentric_factor}")
        
        # Molecular composition section
        lines.append("# Number Of Atoms")
        lines.append(f"{mol.GetNumAtoms()}")
        lines.append("# Number Of Groups")
        lines.append(f"{1}")
        
        # Group information
        lines.append(f"# Group")
        lines.append(f"flexible")
        lines.append("# number of atoms")
        lines.append(f"{mol.GetNumAtoms()}")
        
        # Atomic positions
        lines.append("# atomic positions")
        for i, atom in enumerate(mol.GetAtoms()):
            lines.append(f"{i} {atom.GetProp('label')}")
        
        # Intramolecular interaction flags
        lines.append("# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb")
        intramol = [
            0,                      # Chiral centers
            len(bonds),             # Bond
            0,                      # BondDipoles
            len(bends),             # Bend
            0,                      # UrayBradley
            0,                      # InvBend
            len(torsions),          # Torsion
            0,                      # Imp. Torsion
            0,                      # Bond/Bond 
            0,                      # Stretch/Bend 
            0,                      # Bend/Bend 
            0,                      # Stretch/Torsion 
            0,                      # Bend/Torsion 
            n_vdw,
            n_coulomb
        ]

        flag_format = (
            "{:16d}"  # Chiral centers
            "{:5d}"   # Bond
            "{:13d}"  # BondDipoles
            "{:5d}"   # Bend
            "{:13d}"  # UrayBradley
            "{:8d}"   # InvBend
            "{:9d}"   # Torsion
            "{:13d}"  # Imp. Torsion
            "{:10d}"  # Bond/Bond 
            "{:13d}"  # Stretch/Bend
            "{:10d}"  # Bend/Bend 
            "{:16d}"  # Stretch/Torsion 
            "{:13d}"  # Bend/Torsion 
            "{:9d}"   # IntraVDW 
            "{:12d}"  # IntraCoulomb
        )
        intramolecular_flags = flag_format.format(*intramol)
        lines.append(intramolecular_flags)

        lines.append(bonded)
        
        # Intra-molecular interactions
        lines.append(self.get_intramol_string(interactions))
        
        # Partial reinsertion moves
        lines.append(self.get_nr_fixed_section(mol))
        
        lines.append("")
        return lines


    def build_molecule_definition(self, id, name) -> list:
        
        Tc, pc, acentric_factor = get_trappe_properties(id)
        params = self.load_trappe_parameters(id)
        
        ps = params["pseudoatoms"]
        self.ps_bag.add(name, ps)

        mol = get_mol(name.replace("_", " "), verbose=self.verbose)
        
        if mol is None:
            raise RuntimeError("No molecule could be generated for ", mol)
        
        atoms = ps.get_atoms_main()
        bonds = params['bonds']
        mol = self.align_mol_indeces(mol, atoms, bonds, verbose=self.verbose)

        interactions = self.get_intramol_interactions(mol)
        n_vdw = len(interactions['vdw'])
        n_coulomb = len(interactions['coulomb'])


        lines = []
        
        # Critical constants section
        lines.append("# critical constants: Temperature [T], Pressure [Pa], and Acentric factor [-]")
        lines.append(f"{Tc}")
        lines.append(f"{pc}")
        lines.append(f"{acentric_factor}")
        
        # Molecular composition section
        lines.append("# Number Of Atoms")
        lines.append(f"{params['num_atoms']}")
        lines.append("# Number Of Groups")
        lines.append(f"{params['num_groups']}")
        
        # Group information
        lines.append(f"# {params['group_name']}")
        lines.append(f"{params['group_flexibility']}")
        lines.append("# number of atoms")
        lines.append(f"{params['group_atom_count']}")
        
        # Atomic positions
        lines.append("# atomic positions")
        for i, (index, atom_type) in enumerate(ps.get_atoms()):
            lines.append(f"{i} {atom_type}")
        
        # Intramolecular interaction flags
        lines.append("# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb")
        intramol = params['intramolecular_flags'] + [n_vdw, n_coulomb]
        flag_format = (
            "{:16d}"  # Chiral centers
            "{:5d}"   # Bond
            "{:13d}"  # BondDipoles
            "{:5d}"   # Bend
            "{:13d}"  # UrayBradley
            "{:8d}"   # InvBend
            "{:9d}"   # Torsion
            "{:13d}"  # Imp. Torsion
            "{:10d}"  # Bond/Bond 
            "{:13d}"  # Stretch/Bend
            "{:10d}"  # Bend/Bend 
            "{:16d}"  # Stretch/Torsion 
            "{:13d}"  # Bend/Torsion 
            "{:9d}"   # IntraVDW 
            "{:12d}"  # IntraCoulomb
        )
        intramolecular_flags = flag_format.format(*intramol)
        lines.append(intramolecular_flags)
        
        # Bond stretching parameters
        lines.append("# Bond stretch: atom n1-n2, type, parameters")
        for bond in params['bond_stretches']:
            atom1, atom2, bond_type, force_constant, eq_length = bond
            lines.append(f"{atom1} {atom2} {bond_type} {force_constant} {eq_length}")
        
        # Bond bending parameters (if available)
        if "bond_bends" in params and params["bond_bends"]:
            lines.append("# Bond bending: atom n1-n2-n3, type, parameters")
            for bend in params["bond_bends"]:
                atom1, atom2, atom3, bend_type, force_constant, theta = bend
                lines.append(f"{atom1} {atom2} {atom3} {bend_type} {force_constant} {theta}")
        
        # Torsion parameters (if available)
        if "bond_torsions" in params and params["bond_torsions"]:
            lines.append("# Torsion: atom n1-n2-n3-n4, type, parameters")
            for torsion in params["bond_torsions"]:
                atom1, atom2, atom3, atom4, torsion_type, c0, c1, c2, c3 = torsion
                lines.append(f"{atom1} {atom2} {atom3} {atom4} {torsion_type} {c0} {c1} {c2} {c3}")
        
        # Intra-molecular interactions
        lines.append(self.get_intramol_string(interactions))
        
        # Partial reinsertion moves
        lines.append(self.get_nr_fixed_section(mol))
        
        lines.append("")
        return lines
    

    def load_trappe_parameters(self, molecule_id: int) -> dict:
        """
        Retrieves TraPPE parameters for a given molecule_id and returns a dictionary.
        The returned dictionary includes pseudoatom information, bond stretching, bending,
        and torsion parameters, as well as a formatted intramolecular flag string.
        """

        PARAM_STRING = download_parameters(molecule_id)

        # --- Pseudoatom Section ---
        pseudoatoms = self.parse_section(PARAM_STRING, "#,(pseudo)atom", 6)
        ps = PseudoAtoms()
        ps.parse_trappe(pseudoatoms)

        num_atoms = len(pseudoatoms)

        num_groups = 1  # (could be set to 2 if num_atoms > 8, etc.)
        group_name = "Group"
        group_flexibility = "flexible"
        group_atom_count = num_atoms // num_groups
        # atomic_positions = [(int(p[0]) - 1, p[1]) for p in pseudoatoms]

        # --- Bond Stretching Parameters ---
        stretches = self.parse_section(PARAM_STRING, "#,stretch", 4)
        default_force_constant = 96500
        bond_stretches = []
        bonds = []
        for bond in stretches:
            # bond: [index, bond_range, bond_type, length_str]
            _, bond_range, bond_type, length_str = bond
            bond_range = bond_range.strip(' "\'')
            parts = [p.strip(' "\'') for p in bond_range.split('-')]
            if len(parts) == 2:
                try:
                    atom1 = int(parts[0]) - 1  # Convert from 1-indexed to 0-indexed.
                    atom2 = int(parts[1]) - 1
                    eq_length = float(length_str)
                    #if family == "small":
                    #    bond_stretches.append((atom1, atom2, "RIGID_BOND", "", ""))
                    bond_stretches.append((atom1, atom2, "HARMONIC_BOND", default_force_constant, eq_length))
                    #bond_stretches.append((atom1, atom2, "RIGID_BOND", "", ""))
                    bonds.append((atom1, atom2))
                except Exception:
                    continue
        # --- Bond Bending Parameters ---
        bends = self.parse_section(PARAM_STRING, "#,bend", 5)
        bond_bends = []
        for bend in bends:
            # bend: [index, bend_range, bend_type, theta_str, k_theta_str]
            _, bend_range, bend_type, theta_str, k_theta_str = bend
            bend_range = bend_range.strip(' "\'')
            parts = [p.strip(' "\'') for p in bend_range.split('-')]
            if len(parts) == 3:
                try:
                    atom1 = int(parts[0]) - 1
                    atom2 = int(parts[1]) - 1
                    atom3 = int(parts[2]) - 1
                    theta = float(theta_str)
                    force_constant = float(k_theta_str)
                    bond_bends.append((atom1, atom2, atom3, "HARMONIC_BEND", force_constant, theta))
                except Exception:
                    continue
                
        
        # --- Torsion Parameters ---
        torsions = self.parse_section(PARAM_STRING, "#,torsion", 7)
        bond_torsions = []
        for torsion in torsions:
            # torsion: [index, torsion_range, torsion_type, c0_str, c1_str, c2_str, c3_str]
            _, torsion_range, torsion_type, c0_str, c1_str, c2_str, c3_str = torsion
            torsion_range = torsion_range.strip(' "\'')
            parts = [p.strip(' "\'') for p in torsion_range.split('-')]
            if len(parts) == 4:
                try:
                    atom1 = int(parts[0]) - 1
                    atom2 = int(parts[1]) - 1
                    atom3 = int(parts[2]) - 1
                    atom4 = int(parts[3]) - 1
                
                    c0 = float(c0_str)
                    c1 = float(c1_str)
                    c2 = float(c2_str)
                    c3 = float(c3_str)
                    bond_torsions.append((atom1, atom2, atom3, atom4, "TRAPPE_DIHEDRAL", c0, c1, c2, c3))
                except Exception:
                    continue
        
        num_bond_stretches = len(bond_stretches)
        num_bond_bends     = len(bond_bends)
        num_bond_torsions  = len(bond_torsions)
        
        # Create intramolecular flags using the prescribed format.
        fields = [
            0,                      # Chiral centers
            num_bond_stretches,     # Bond
            0,                      # BondDipoles
            num_bond_bends,         # Bend
            0,                      # UrayBradley
            0,                      # InvBend
            num_bond_torsions,      # Torsion
            0,                      # Imp. Torsion
            0,                      # Bond/Bond 
            0,                      # Stretch/Bend 
            0,                      # Bend/Bend 
            0,                      # Stretch/Torsion 
            0,                      # Bend/Torsion 
        ]
        parameters = {
            "num_atoms": num_atoms,
            "num_groups": num_groups,
            "group_name": group_name,
            "group_flexibility": group_flexibility,
            "group_atom_count": group_atom_count,
            "intramolecular_flags": fields,
            "bond_stretches": bond_stretches,
            "bond_bends": bond_bends,
            "bond_torsions": bond_torsions,
            "pseudoatoms" : ps,
            "bonds" : bonds
        }
        return parameters


    def get_intramol_interactions(self, mol: Chem.Mol, k: int = 4) -> tuple:
        """
        Computes a dictionary of atom-atom interactions that are separated by at least k bonds.
        """
        n_atoms = mol.GetNumAtoms()
        interactions = {"vdw" : [], "coulomb" : []}

        def is_coulomb(a1: Chem.Atom, a2: Chem.Atom):
            #polar_atoms = {"N", "O", "F", "Cl", "Br", "I", "P", "S"}
            #return (a1.GetSymbol() in polar_atoms and a2.GetSymbol() in polar_atoms) 
            if a1.HasProp("charge") and a1.HasProp("charge"):
                return a1.GetDoubleProp("charge") != 0 and a2.GetDoubleProp("charge") != 0
            else:
                #raise RuntimeError("Atoms dont have charge assigned!")
                return False

        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                path = Chem.rdmolops.GetShortestPath(mol, i, j)
                dist = len(path) - 1
                if dist >= k:
                    interactions["vdw"].append((i,j))

                    atom_i = mol.GetAtomWithIdx(i)
                    atom_j = mol.GetAtomWithIdx(j)
                    
                    if is_coulomb(atom_i, atom_j):
                        interactions["coulomb"].append((i,j))

        
        return interactions
    

    def align_mol_indeces(self, mol, atoms, bonds, verbose=False):
        '''
        Align indices from external atom/bond list to RDKit mol atom indices.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            The RDKit molecule.
        atoms : dict
            {index: (main_atom, charge)} - external indices and atom properties.
        bonds : list of tuples
            [(index1, index2), ...] - external bond pairs using atom indices.

        Returns
        -------
        atom_map : dict
            Mapping from mol index -> atoms.keys() (external indices)
        '''

        # 1. Create a reference structure of atom symbols and bond topology from input
        def extract_main_atom(label):
            # Normalize pseudo-groups like CHx, CFx, etc. to the main atom
            if label.startswith("CH") or label.startswith("CF"):
                return "C"
            elif label.startswith("NH"):
                return "N"
            elif label.startswith("OH"):
                return "O"
            elif label.startswith("SH"):
                return "S"
            elif label.startswith("Hx") or label == "H":
                return "H"
            elif label.startswith("PH"):
                return "P"
            else:
                return label

        # Now build the symbol map with normalization
        atom_idx_to_symbol = {
            i: extract_main_atom(data[0]) for i, data in atoms.items() if data[0] != "M"
        }
        atom_idx = set(atom_idx_to_symbol.keys())
        
        # 2. Build an adjacency graph from external atoms and bonds
        adjacency = defaultdict(set)
        for i, j in bonds:
            if i in atom_idx and j in atom_idx:
                adjacency[i].add(j)
                adjacency[j].add(i)

        # 3. Build a similar graph from mol
        mol_atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        mol_adjacency = defaultdict(set)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            mol_adjacency[i].add(j)
            mol_adjacency[j].add(i)

        if verbose:
            print(atom_idx_to_symbol)
            print(adjacency)
            print(mol_adjacency)

        # 4. Try to match external atom graph to mol atom graph
        # Naive approach: assume same number of atoms and that a unique match exists based on symbols and connectivity
        atom_map = {}
        used_atoms = set()

        for mol_idx, symbol in enumerate(mol_atom_symbols):
            candidates = [ext_idx for ext_idx, sym in atom_idx_to_symbol.items()
                        if sym == symbol and ext_idx not in used_atoms]

            for ext_idx in candidates:
                ext_neighbors = {atom_idx_to_symbol[n] for n in adjacency[ext_idx]}
                mol_neighbors = {mol_atom_symbols[n] for n in mol_adjacency[mol_idx]}
                if ext_neighbors == mol_neighbors:
                    atom_map[mol_idx] = ext_idx
                    used_atoms.add(ext_idx)
                    break
            else:
                if self.verbose:
                    print((f"Could not find a match for mol atom {mol_idx} ({symbol})"))
                raise ValueError

        # Create inverse map: from external index -> mol index
        ext_to_mol_map = {v: k for k, v in atom_map.items()}

        # Build new ordering: mol indices sorted by external index order
        new_order = [ext_to_mol_map[i] for i in sorted(atom_idx)]

        # Renumber atoms in the molecule
        mol = Chem.RenumberAtoms(mol, new_order)

        # Reassign charges to the reordered atoms
        for i, atom in enumerate(mol.GetAtoms()):
            charge = atoms[i][1]
            atom.SetDoubleProp("charge", charge)

        return mol



    def get_intramol_string(self, interactions: dict) -> str:
        """
        Generates a formatted string for the IntraVDW and IntraCoulomb sections of the molecule.def file.
        """
        lines = []
        n_vdw = len(interactions['vdw'])
        n_coulomb = len(interactions['coulomb'])

        if n_vdw > 0:
            lines.append("# Intra VDW: atom n1-n2")
            for (i, j) in sorted(interactions["vdw"]):
                lines.append(f"{i} {j}")

        if n_coulomb > 0:    
            lines.append("# Intra Coulomb: atom n1-n2")
            for (i, j) in sorted(interactions["coulomb"]):
                lines.append(f"{i} {j}")
            
        return "\n".join(lines)
        

    def get_nr_fixed_section(self, mol: Chem.Mol) -> str:
        """
        Generates the "nr fixed" section (fixed fragments list) for a molecule.def file.
        """
        n = mol.GetNumAtoms()
        if n < 2:
            return "\n".join(["# Number of config moves", str(len(lines) - 1)])
        
        degrees = {atom.GetIdx(): len(atom.GetNeighbors()) for atom in mol.GetAtoms()}
        terminals = [idx for idx, deg in degrees.items() if deg == 1]
        branch_fragments = []
        
        for t in terminals:
            branch = [t]
            current = t
            prev = -1
            while True:
                nbrs = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(current).GetNeighbors() if nbr.GetIdx() != prev]
                if not nbrs:
                    break
                if len(nbrs) != 1:
                    branch.append(nbrs[0])
                    break
                next_idx = nbrs[0]
                branch.append(next_idx)
                if degrees[next_idx] != 2:
                    break
                prev, current = current, next_idx
            branch_fragments.append(branch)
        
        if len(terminals) == 2 and all(degrees[i] == 2 for i in range(n) if i not in terminals):
            lower_terminal = min(terminals)
            upper_terminal = max(terminals)
            branch_lower = list(range(lower_terminal, lower_terminal + (n - 2)))
            branch_upper = list(range(upper_terminal, upper_terminal - (n - 2), -1))
            unique_branches = [branch_lower, branch_upper]
        else:
            unique_branches = []
            seen = set()
            for branch in branch_fragments:
                t_branch = tuple(branch)
                t_branch_rev = tuple(reversed(branch))
                if t_branch in seen or t_branch_rev in seen:
                    continue
                seen.add(t_branch)
                unique_branches.append(branch)
        
        fixed_subchains = []
        for branch in unique_branches:
            L = len(branch)
            for sub_len in range(L, 0, -1):
                subchain = branch[:sub_len]
                fixed_subchains.append((sub_len, subchain))
        
        fixed_subchains.sort(key=lambda x: (-x[0], x[1]))
        
        lines = ["# nr fixed, list"]
        for length, chain in fixed_subchains:
            lines.append(f"{length} " + " ".join(map(str, chain)))
        
        out = ["# Number of config moves", str(len(lines) - 1)]
        out.extend(lines)
        out.append("")
            
        return "\n".join(out)