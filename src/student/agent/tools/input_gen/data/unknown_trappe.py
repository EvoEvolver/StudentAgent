import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Atom
from student.agent.tools.input_gen.utils_molecules import get_mol, molecule_name_to_smiles
import os

PATH = os.path.dirname(__file__)


###### Properties ######

def load_properties():
    data = pd.read_csv(os.path.join(PATH,"properties/critprop_data_only_smiles_mean_value_expt.csv"), usecols=[0,1,2,4])
    return data

def get_properties(smiles):
    data = load_properties() # load data only if required, store for next time
    properties = data[data["smiles"] == smiles]
    p = properties.to_dict('records')[0]         # keys = ['smiles', 'Tc (K)', 'Pc (bar)', 'omega (-)']
    return p['Tc (K)'], p['Pc (bar)']*100000, p['omega (-)']


###### Parameters ######


def get_main_type(atom : Atom):
    """
    Returns the main_type string for a given atom, or None if no match.
    """
    symbol = atom.GetSymbol()
    if symbol != 'C':
        if symbol in ['C', 'N', 'O', 'H', 'S', 'P', 'F']:
            return symbol
        else:
            return None

    num_h = atom.GetTotalNumHs()
    num_f = sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'F')
    
    if num_f == 3:
        return 'CF3'
    elif num_f == 2:
        return 'CF2'
    elif num_h == 4:
        return 'CH4'
    elif num_h == 3:
        return 'CH3'
    elif num_h == 2:
        return 'CH2'
    elif num_h == 1:
        return 'CH'
    elif num_h == 0:
        return 'C'

    return None

def load_smarts(type_to_smarts):
    # Precompile SMARTS
    compiled_smarts = {
        main_type: {
            label: Chem.MolFromSmarts(smarts)
            for label, smarts in label_dict.items()
            if smarts is not None and Chem.MolFromSmarts(smarts) is not None
        }
        for main_type, label_dict in type_to_smarts.items() if main_type != "M"
    }
    return compiled_smarts


def assign_atom_types_by_smarts(mol, type_to_smarts):
    """
    Assigns:
    - atom.SetProp("main_type", main_type) using get_main_type()
    - atom.SetProp("label", label) based on SMARTS match and declared priority
    """
    compiled_smarts = load_smarts(type_to_smarts)
    
    # First pass: assign main_type to all atoms
    for atom in mol.GetAtoms():
        main_type = get_main_type(atom)
        if main_type in type_to_smarts:
            atom.SetProp("main_type", main_type)
    
    # Process atoms by main_type
    for main_type, patterns in compiled_smarts.items():
        atoms_of_type = [
            atom for atom in mol.GetAtoms()
            if atom.HasProp("main_type") and atom.GetProp("main_type") == main_type
        ]
        if not atoms_of_type:
            continue

        # Precompute all SMARTS matches for this type
        pattern_matches = {}
        for label, pattern in patterns.items():
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                for atom_idx in match:
                    if atom_idx not in pattern_matches:
                        pattern_matches[atom_idx] = label  # first match wins

        # Assign labels
        for atom in atoms_of_type:
            idx = atom.GetIdx()
            if idx in pattern_matches:
                atom.SetProp("label", pattern_matches[idx])
            else:
                atom.SetProp("label", main_type)
                print("Warning: No corresponding pseudoatom type could be assigned to atom with index ", atom.GetIdx(), " and main type ", main_type)
        # print(pattern_matches)
    return mol



def bonded_definitions(stretches, bends, torsions, bonded):
    stretch_labels, _, _, bend_labels, _, torsion_labels, _ = bonded

    # Bond stretching parameters
    lines = []
    
    if len(stretches) > 0:
        lines.append("# Bond stretch: atom n1-n2, type, parameters")
        for atoms, bond_type in stretches.items():
            atom1, atom2 = atoms
            eq_length = list(stretch_labels[bond_type])[0]
            lines.append(f"{atom1} {atom2} {bond_type} 96500 {eq_length}")

    # Bond bending parameters
    if len(bends) > 0:
        lines.append("# Bond bending: atom n1-n2-n3, type, parameters")
        for atoms, bend_type in bends.items():
            atom1, atom2, atom3 = atoms 
            force_constant, theta = list(bend_labels[bend_type])[0]
            lines.append(f"{atom1} {atom2} {atom3} {bend_type} {force_constant} {theta}")

    # Torsion parameters
    if len(torsions) > 0:
        lines.append("# Torsion: atom n1-n2-n3-n4, type, parameters")
        for atoms, torsion_type in torsions.items():
            atom1, atom2, atom3, atom4 = atoms
            c0, c1, c2, c3 = list(torsion_labels[torsion_type])[0]
            lines.append(f"{atom1} {atom2} {atom3} {atom4} {torsion_type} {c0} {c1} {c2} {c3}")
    return"\n".join(lines)

def assign_bonded_interactions(mol, bonded):
    _, _, stretches_atoms_bonds, _, bend_atoms, _, torsion_atoms = bonded

    bonds = {}

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetProp("label")
        a2 = bond.GetEndAtom().GetProp("label")

        label = stretches_atoms_bonds.get(a1, {}).get(a2, None)
        if label is not None:
            bond.SetProp("label", label)
            bonds[(bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx())] = label
            continue
        
        print("No bond label could be matched for ", a1, " <-> ", a2)
        

    bends = {}

    for a in mol.GetAtoms():
        neighbors = a.GetNeighbors()
        if len(neighbors) < 2:
            continue
        
        bends_a = bend_atoms.get(a.GetProp("label"), None)
        if bends_a is None:
            print("No bends known for atom with type ", a.GetProp("label"))
            continue

        for n1 in neighbors:
            bends_t = bends_a.get(n1.GetProp("label"), None)
            if bends_t is None:
                print("No bends known for atoms with type ", a.GetProp("label"), " and ", n1.GetProp("label"))
                break

            for n2 in neighbors:
                if n1 == n2:
                    break
                b = bends_t.get(n2.GetProp("label"), None)
                if b is None:
                    print("No bends known for atoms with type ", a.GetProp("label"), " and ", n1.GetProp("label"), " and ", n2.GetProp("label"))
                    break
                
                bends[(n1.GetIdx(), a.GetIdx(), n2.GetIdx())] = b

    torsions = {}
    for bond in mol.GetBonds():
        b_id = bond.GetIdx()

        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        l1 = a1.GetProp("label")
        l2 = a2.GetProp("label")

        # find all pairs of neighboring bonds
        
        b1 = [b for b in a1.GetBonds() if b.GetIdx() != b_id]
        b2 = [b for b in a2.GetBonds() if b.GetIdx() != b_id]

        if len(b1) == 0 or len(b2) == 0:    # >= 1 atom is terminal
            continue

        for b0 in b1:
            a0 = [a for a in [b0.GetBeginAtom(), b0.GetEndAtom()] if a.GetIdx() != a1.GetIdx()][0]
            l0 = a0.GetProp("label")
            
            t0 = torsion_atoms.get(l0, None)
            if t0 is None:
                print("No torsion for atom ", l0)
                continue

            t1 = t0.get(l1, None)
            if t1 is None:
                print("No torsion for atom ",l0, " <-> ", l1)
                continue

            t2 = t1.get(a2.GetProp("label"), None)
            if t2 is None:
                print("No torsion for atom ", l0, " <-> ", l1, " <-> ", l2)
                continue

            for b3 in b2:
                a3 = [a for a in [b3.GetBeginAtom(), b3.GetEndAtom()] if a.GetIdx() != a2.GetIdx()][0]
                l3 = a3.GetProp("label")
                
                t3 = t2.get(l3, None)
                if t3 is None:
                    print("No torsion for atom ", l0, " <-> ", l1, " <-> ", l2, " <-> ", l3)
                    continue

                # torsion of atoms a0 - a1 - a2 - a3

                if a0 == a3:    # 3-ring has no torsions
                    continue

                torsions[(a0.GetIdx(), a1.GetIdx(), a2.GetIdx(), a3.GetIdx())] = t3

    return bonds, bends, torsions    