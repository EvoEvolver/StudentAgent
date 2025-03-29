import os
import numpy as np
import pandas as pd
from io import StringIO

from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import curve_fit

from student.input_gen.utils_molecules import get_mol
from student.input_gen.trappe_loader import download_parameters, download_properties
from student.input_gen.pseudoatoms import PseudoAtoms, Atom

def get_trappe_properties(molecule_id: int):
    """
    Returns: critical constants: Temperature [T] in Kelvin, Pressure [Pa], and Acentric factor [-]
    """
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


def get_trappe_parameters(molecule_id: int, n_vdw, n_coulomb) -> dict:
    """
    Retrieves TraPPE parameters for a given molecule_id and returns a dictionary.
    The returned dictionary includes pseudoatom information, bond stretching, bending,
    and torsion parameters, as well as a formatted intramolecular flag string.
    """

    def parse_section(param_str: str, section_key: str, min_parts: int) -> list:
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
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= min_parts:
                    section_lines.append(parts[:min_parts])
        return section_lines

    PARAM_STRING = download_parameters(molecule_id)

    # --- Pseudoatom Section ---
    pseudoatoms = parse_section(PARAM_STRING, "#,(pseudo)atom", 6)
    ps = PseudoAtoms()
    ps.parse([pseudoatoms])

    num_atoms = len(pseudoatoms)

    num_groups = 1  # (could be set to 2 if num_atoms > 8, etc.)
    group_name = "Group"
    group_flexibility = "flexible"
    group_atom_count = num_atoms // num_groups
    # atomic_positions = [(int(p[0]) - 1, p[1]) for p in pseudoatoms]

    # --- Bond Stretching Parameters ---
    stretches = parse_section(PARAM_STRING, "#,stretch", 4)
    default_force_constant = 96500
    bond_stretches = []
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
            except Exception:
                continue
    # --- Bond Bending Parameters ---
    bends = parse_section(PARAM_STRING, "#,bend", 5)
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
    torsions = parse_section(PARAM_STRING, "#,torsion", 7)
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
        n_vdw,                # IntraVDW 
        n_coulomb             # IntraCoulomb
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
    intramolecular_flags = flag_format.format(*fields)

    parameters = {
        "num_atoms": num_atoms,
        "num_groups": num_groups,
        "group_name": group_name,
        "group_flexibility": group_flexibility,
        "group_atom_count": group_atom_count,
        "intramolecular_flags": intramolecular_flags,
        "bond_stretches": bond_stretches,
        "bond_bends": bond_bends,
        "bond_torsions": bond_torsions,
        "pseudoatoms" : ps
    }
    
    return parameters


def get_intramol_interactions(mol: Chem.Mol, k: int = 4) -> tuple:
    """
    Computes a dictionary of atom-atom interactions that are separated by at least k bonds.
    Returns a tuple: (n_vdw, n_coulomb, interactions)
    """
    mol = Chem.RemoveHs(mol)
    n_atoms = mol.GetNumAtoms()
    interactions = {}
    n_vdw = 0
    n_coulomb = 0

    def classify_interaction(atom1: Chem.Atom, atom2: Chem.Atom) -> str:
        polar_atoms = {"N", "O", "F", "Cl", "Br", "I", "P", "S"}
        return "coulomb" if (atom1.GetSymbol() in polar_atoms or atom2.GetSymbol() in polar_atoms) else "vdw"

    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            path = Chem.rdmolops.GetShortestPath(mol, i, j)
            dist = len(path) - 1
            if dist >= k:
                atom_i = mol.GetAtomWithIdx(i)
                atom_j = mol.GetAtomWithIdx(j)
                classification = classify_interaction(atom_i, atom_j)
                if classification == 'vdw':
                    n_vdw += 1
                else:
                    n_coulomb += 1

                interactions[(i, j)] = {
                    "distance": dist,
                    "classification": classification,
                    "atoms": (atom_i.GetSymbol(), atom_j.GetSymbol())
                }
    
    return n_vdw, n_coulomb, interactions


def get_intramol_string( n_vdw: int, n_coulomb: int, interactions: dict) -> str:
    """
    Generates a formatted string for the IntraVDW and IntraCoulomb sections of the molecule.def file.
    """
    lines = []
    if n_vdw > 0:
        lines.append("# Intra VDW: atom n1-n2")
        for (i, j), info in sorted(interactions.items()):
            if info['classification'] == 'vdw':
                lines.append(f"{i} {j}")

    if n_coulomb > 0:    
        lines.append("# Intra Coulomb: atom n1-n2")
        for (i, j), info in sorted(interactions.items()):
            if info['classification'] == 'coulomb':
                lines.append(f"{i} {j}")
        
    return "\n".join(lines)


def get_nr_fixed_section(mol: Chem.Mol) -> str:
    """
    Generates the "nr fixed" section (fixed fragments list) for a molecule.def file.
    """
    mol = AllChem.RemoveHs(mol)
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


def build_molecule_definition_lines(ps: PseudoAtoms, Tc, pc, acentric_factor, params, n_vdw, n_coulomb, interactions) -> list:
    """
    Constructs the list of lines that will make up the molecule.def file.
    """
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
    for index, atom_type in ps.get_atoms():
        lines.append(f"{index} {atom_type}")
    
    # Intramolecular interaction flags
    lines.append("# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb")
    lines.append(params['intramolecular_flags'])
    
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
    lines.append(get_intramol_string(n_vdw, n_coulomb, interactions))
    
    return lines


def generate_molecule_def(molecule_id: int, name: str, output_dir: str):
    """
    Automatically generates a molecule.def file for RASPA using TraPPE data.
    The interface of this function remains fixed.
    
    Parameters:
        molecule_id: Unique identifier for the molecule (used in TraPPE API calls)
        name: Molecule name (passed to get_mol)
        output_file: Path to the output molecule.def file
    
    Returns:
        True if file generation succeeds.
    """
    mol = get_mol(name)
    n_vdw, n_coulomb, interactions = get_intramol_interactions(mol)
    
    Tc, pc, acentric_factor = get_trappe_properties(molecule_id)
    
    params = get_trappe_parameters(molecule_id, n_vdw, n_coulomb)
    ps = params["pseudoatoms"]
    ps.generate_ps_file(output_dir=output_dir)
    ps.generate_ff_file(output_dir=output_dir)
    
    lines = build_molecule_definition_lines(ps, Tc, pc, acentric_factor, params, n_vdw, n_coulomb, interactions)
    
    with open(os.path.join(output_dir, "molecule.def"), "w") as f:
        f.write("\n".join(lines))
    
    return True
