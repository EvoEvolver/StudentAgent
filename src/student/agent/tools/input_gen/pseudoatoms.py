import os
import re
import copy
import pickle
from typing import Dict, List

PATH = os.path.dirname(__file__)

with open(os.path.join(PATH, "ps_type_to_label.pkl"), "rb") as f:
    TYPE_TO_LABEL = pickle.load(f)


class Atom:
    def __init__(self, id, atom_type: str, ps_type, epsilon, sigma, charge):
        self.id = [id]
        self.atom_type = atom_type      # main atom
        self.ps_type = ps_type        # ps_type
        self.epsilon = epsilon
        self.sigma = sigma
        self.charge = charge
        self.radius = self.get_radius()
        self.mass = self.get_mass()
        self.label = self.get_label()
        self.add_id = 0

    def get_label(self):
        return TYPE_TO_LABEL.get(self.atom_type, {}).get(self.ps_type, None)

    def get_radius(self):
        label = self.atom_type
        if label.startswith("CH"):
            label = "CHx"

        radii = {
            # collected from example pseudoatom files
            "C" : 0.72,
            "CHx" : 1.0,
            "O" : 0.68,
            "H" : 0.32,
            "N" : 0.7,
            "He": 1.0,
            "Ar": 0.7,
            "S" : 0.9,
            "F" : 0.68, # as oxygen
            "P" : 0.9,  # as sulfur
            "CFx" : 1.0, # as CHx
            "M" : 0,
        }
        return radii.get(label, 0)
        
    def get_mass(self):
        masses = {
            "C" : 12.0107,
            "O" : 15.9994,
            "H" : 1.00794,
            "N" : 14.00674,
            "He": 4.002602,
            "Ar": 39.948,
            "S" : 32.065,
            "F" : 18.998,
            "P" : 30.974,
            "M" : 0,
        }
        atom_type = self.atom_type
        if atom_type in masses.keys():
            return masses.get(atom_type, 0)
        elif atom_type.startswith("CH"):
            n_h = 1 if len(atom_type) == 2 else int(atom_type[2])
            mass = masses.get("C") + n_h * masses.get("H")
            return mass
        elif atom_type.startswith("CF"):
            n_f = 1 if len(atom_type) == 2 else int(atom_type[2])
            mass = masses.get("C") + n_f * masses.get("F")
            return mass
        else:
            return NotImplemented     

    def __eq__(self, other) -> bool:
        if not isinstance(other, Atom):
            return NotImplemented
        return (self.atom_type == other.atom_type and
                self.ps_type == other.ps_type and
                self.epsilon == other.epsilon and
                self.sigma == other.sigma and
                self.charge == other.charge)
    
    def __add__(self, other):
        if self == other:
            self.id.extend(other.id)
            return self
        
        return False
    
    def __repr__(self):
        if self.add_id == 0:
            return self.label
        return f"{self.label}_{self.add_id}"



class PseudoAtoms:
    def __init__(self):
        self.atoms = {} # id -> Atom
        self.ps_labels = {} # ps_type -> [ids]

    def __add__(self, other):
        new_ps = copy.deepcopy(self)

        for id in other.atoms.keys():
            if id not in new_ps.atoms.keys():
                new_ps.atoms[id] = copy.deepcopy(other.atoms[id])    
            else:
                print("Error with Pseudoatom additions: IDs not unique!")
                return self
            
        for atom_type in other.ps_labels.keys():
            if atom_type in new_ps.ps_labels.keys():
                new_ps.ps_labels[atom_type].extend(copy.deepcopy(other.ps_labels[atom_type]))
            else:
                new_ps.ps_labels[atom_type] = copy.deepcopy(other.ps_labels[atom_type])

        return new_ps
    

    def get_atoms(self):
        atomic_positions = []
        for id, atom in self.atoms.items():
            atomic_positions.append((id, atom.__repr__()))
        return atomic_positions

    def parse_trappe(self, section: list[str]) -> None:
        for parts in section:
            try:
                id = int(parts[0])-1
                main_atom = parts[1]
                type_val = parts[2]
                epsilon = float(parts[3])
                sigma = float(parts[4])
                charge = float(parts[5])
            
            except Exception as e:
                print("Error parsing pseudoatom line:", e)
                print("You might need to check if you used a list in the input!")
                return

            a = Atom(id, main_atom, type_val, epsilon, sigma, charge)
            self.atoms[id] = a
            self.ps_labels[a.label] = self.ps_labels.get(a.label, []) + [id]

    def get_atoms_with_label(self, label: str):
        return [self.atoms[id] for id in self.ps_labels.get(label, [])]


    def get_atoms_main(self):
        return {index : (atom.atom_type, atom.charge) for index, atom in self.atoms.items()}


class PseudoAtomsBag:
    def __init__(self):
        self.pseudoatoms : Dict[str: PseudoAtoms] = {} # name : Pseudoatoms
        self.labels = set()

    def add(self, name, ps : PseudoAtoms):
        self.pseudoatoms[name] = ps
        for label in ps.ps_labels.keys():
            self.labels.add(label)

    def get_atoms_with_label(self, label : str) -> List:
        atoms = []
        for ps in self.pseudoatoms.values():
            atoms.extend(ps.get_atoms_with_label(label))
        return atoms
    
    def get_unique_atoms_with_label(self, label: str):
        atoms = self.get_atoms_with_label(label)
        
        unique_atoms = []
        for atom in atoms:
            duplicate_found = False
            
            for idx, u in enumerate(unique_atoms):
                if u == atom:
                    unique_atoms[idx] = u + atom
                    duplicate_found = True
                    break
                elif u.label == atom.label: # if label is identical, but different parameters, add distinct identifier
                    atom.add_id = u.add_id + 1
            
            if not duplicate_found:
                unique_atoms.append(atom)
        
        return unique_atoms

    def get_unique_atoms(self):
        atoms = []
        for label in self.labels:
            atoms.extend(self.get_unique_atoms_with_label(label))
        return atoms

    def build_ff_mixing(self):
        atoms = self.get_unique_atoms()
        n = len(atoms)

        framework_ff, n_ff = self.get_generic_mof()

        lines = []
        lines.append("# general rule for shifted vs truncated")
        lines.append("truncated")
        lines.append("# general rule tailcorrections")
        lines.append("yes")
        lines.append("# number of defined interactions")
        lines.append(f"{n+n_ff}")
        lines.append("# type interaction")
        lines.append(framework_ff)
        
        for atom in atoms:
            ps_type = atom.__repr__()
            epsilon = atom.epsilon
            sigma = atom.sigma
            
            fields = [
                ps_type,        # type
                "lennard-jones",
                f"{epsilon:.5g}",        # epsilon
                f"{sigma:.5g}",          # sigma
            ]

            flag_format = (
                "{:15}"   # type
                "{:18s}"  
                "{:9s}"   # epsilon
                "{:12s}"  # sigma
            )     
            line = flag_format.format(*fields)
            lines.append(line)
        lines.append("# general mixing rule for Lennard-Jones")
        lines.append("Lorentz-Berthelot")
        lines.append("")  
        
        return "\n".join(lines)
    
    def build_ff(self):
        lines = []
        lines.append("# rules to overwrite")
        lines.append("0")
        lines.append("# number of defined interactions")
        lines.append("0")
        lines.append("# mixing rules to overwrite")
        lines.append("0")
        lines.append("")
        return "\n".join(lines)
        
    def get_generic_mof(self):
        with open(os.path.join(os.path.dirname(__file__), "forcefields/generic_ff_mof.def"), "r") as f:
            s = f.read()
        return s, 45
    
    def get_generic_zeolites(self):
        with open(os.path.join(os.path.dirname(__file__), "forcefields/generic_ff_zeolites.def"), "r") as f:
            s = f.read()
        return s, 16

    def build_pseudoatoms(self):
        atoms = self.get_unique_atoms()
        n = len(atoms)

        lines = []
        lines.append("#number of pseudo atoms")
        lines.append(f"{n}")
        lines.append("#type      print   as    chem  oxidation   mass        charge   polarization B-factor radii  connectivity anisotropic anisotropic-type   tinker-type")

        for atom in atoms:
            ps_type = atom.__repr__()
            #alias =  re.sub(r'\d', '', ps_type)[:-3]
            alias = atom.atom_type
            if alias[-1] == "H" and len(alias) > 1:
                alias = alias[:-1]
            
            chem = alias
            charge = atom.charge
            mass = atom.mass
            radii = atom.radius
            
            fields = [
                ps_type,  # type
                "yes", # print
                alias,    # as
                chem,     # chem
                "0",        # oxidation
                f"{mass:.5g}",     # mass
                f"{charge:.5g}",   # charge
                "0.0",      # polarization
                "1.0",      # B-factor     
                f"{radii:.5g}",    # radii
                "0",        # connectivity
                "0",        # anisotropic
                "relative", # anisotropic-type 
                "0"        # tinker-type
            ]

            flag_format = (
                "{:11s}"   # type
                "{:8s}"  # print
                "{:6s}"   # as
                "{:6s}"  # chem
                "{:12s}"   # oxidation
                "{:12}"   # mass
                "{:9s}"  # charge
                "{:13s}"  # polarization
                "{:9s}"  # B-factor     
                "{:7s}"  # radii
                "{:14s}"  # connectivity
                "{:11s}"  # anisotropic
                "{:19}"   # anisotropic-type 
                "{:11s}"  # tinker-type
            )     
            line = flag_format.format(*fields)
            lines.append(line)
        lines.append("")  

        return "\n".join(lines)