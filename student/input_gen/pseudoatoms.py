import os
import re

class Atom:
    def __init__(self, id, atom_type: str, type_val, epsilon, sigma, charge):
        self.id = [id]
        self.atom_type = atom_type 
        self.type_val = type_val
        self.epsilon = epsilon
        self.sigma = sigma
        self.charge = charge
        self.radius = self.get_radius(atom_type)
        self.mass = self.get_mass(atom_type)
        self.label = self.__repr__()

    def get_radius(self, atom_type):
        label = atom_type
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
        }
        return radii.get(label, 0)
        
    def get_mass(self, atom_type):
        masses = {
            "C" : 12.0107,
            "O" : 15.9994,
            "H" : 1.00794,
            "N" : 14.00674,
            "He": 4.002602,
            "Ar": 39.948,
            "S" : 32.065,
        }
        if atom_type in masses.keys():
            return masses.get(atom_type, 0)
        elif atom_type.startswith("CH"):
            n_h = 1 if len(atom_type) == 2 else int(atom_type[2])
            mass = masses.get("C") + n_h * masses.get("H")
            return mass
        else:
            return NotImplemented     

    def __eq__(self, other) -> bool:
        if not isinstance(other, Atom):
            return NotImplemented
        return (self.atom_type == other.atom_type and
                # self.type_val == other.type_val and
                self.epsilon == other.epsilon and
                self.sigma == other.sigma and
                self.charge == other.charge)
    
    def __add__(self, other):
        if self == other:
            self.id.extend(other.id)
            return self
        return NotImplemented
    
    def __repr__(self):
        return f"{self.atom_type}_{self.id[0]}"


class PseudoAtoms:
    def __init__(self):
        """
        Container for multiple PseudoAtom objects, keyed by main atom type.
        """
        self.atoms = {} # id -> Atom
        self.atom_types = {} # atom_type -> [ids]

    def parse(self, sections: list[list[str]]) -> None:
        """
        Parses a list of pseudoatom sections (each section as a multiline string)
        and updates the object in place by populating its dictionary of PseudoAtom objects.
        
        Parameters:
            sections (list[str]): A list of pseudoatom section strings.
        """

        for section in sections:
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

                #if main_atom not in self.atoms:
                self.atoms[id] = Atom(id, main_atom, type_val, epsilon, sigma, charge)
                ids = self.atom_types.get(main_atom, [])
                self.atom_types[main_atom] = ids + [id]

    def get_atoms_with_type(self, atom_type: str):
        ids = self.atom_types.get(atom_type, [])
        atoms = []
        for id in ids:
            atoms.append(self.atoms[id])
        return atoms
    
    def get_atom_types(self):
        return list(self.atoms.keys())
    
    def get_unique_atoms_with_type(self, atom_type: str):
        atoms = self.get_atoms_with_type(atom_type)
        if atoms is None:
            return None

        unique_atoms = []
        for atom in atoms:
            duplicate_found = False
            for idx, u in enumerate(unique_atoms):
                if u == atom:
                    unique_atoms[idx] = u + atom
                    duplicate_found = True
                    break
            if not duplicate_found:
                unique_atoms.append(atom)
        return unique_atoms

    def get_unique_atoms(self):
        atoms = []
        for atom_type in self.atom_types:
            atoms.extend(self.get_unique_atoms_with_type(atom_type))
        return atoms


    def get_pesudoatoms_string(self):
        # atoms = self.get_unique_atoms()
        atoms = self.atoms.values()
        n = len(atoms)

        lines = []
        lines.append("#number of pseudo atoms")
        lines.append(f"{n}")
        lines.append("#type      print   as    chem  oxidation   mass        charge   polarization B-factor radii  connectivity anisotropic anisotropic-type   tinker-type")

        for atom in atoms:
            ps_type = atom.__repr__()
            alias =  re.sub(r'\d', '', ps_type)[:-1]
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
            
        return "\n".join(lines)

    def get_atoms(self):
        atomic_positions = []
        for id, p in self.atoms.items():
            atomic_positions.append((id, p.__repr__()))
        return atomic_positions
    
    def get_ff_string(self):
        atoms = self.atoms.values()
        n = len(atoms)

        lines = []
        lines.append("# general rule for shifted vs truncated")
        lines.append("shifted")
        lines.append("# general rule tailcorrections")
        lines.append("no")
        lines.append("# number of defined interactions")
        lines.append(f"{n}")
        lines.append("# type interaction")
        
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
        
        return "\n".join(lines)

    def generate_ff_file(self, output_dir=None):
        s = self.get_ff_string()
        with open(os.path.join(output_dir, "force_field_mixing_rules.def"), "w") as f:
            f.write(s)

    def generate_ps_file(self, output_dir=None):
        s = self.get_pesudoatoms_string()
        with open(os.path.join(output_dir, "pseudo_atoms.def"), "w") as f:
            f.write(s)
