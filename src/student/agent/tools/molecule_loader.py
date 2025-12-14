from typing import Union, List

from pydantic_ai import RunContext

from student.agent.tools.input_gen.molecule_loader import MoleculeLoaderTrappe
from student.agent.utils import file


class MoleculeLoader(MoleculeLoaderTrappe):
    def __init__(self, path=None):
        name = "molecule_loader"
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
            "oxirane": "ethylene oxide",
            "co2": "carbon dioxide",
            "carbon_dioxide": "carbon dioxide",
            "co₂": "carbon dioxide",
            "n2": "nitrogen",
            "o2": "oxygen",
            "nh3": "ammonia",
            "h2s": "hydrogen sulfide",
            "ch4": "methane",
            "c2h6": "ethane",
            "c3h8": "propane",
            "c4h10": "butane",
            "c5h12": "pentane",
            "c6h14": "hexane",
            "c7h16": "heptane",
            "c8h18": "octane",
            "c6h6": "benzene",
            "c7h8": "toluene",
        }

    def normalize_name(self, name: str) -> str:
        """Convert common chemical formulas and abbreviations to standard names."""
        # Convert to lowercase and remove spaces for matching
        normalized = name.lower().strip().replace("_", " ")

        # Check if we have a mapping for this name
        if normalized in self.name_mappings:
            return self.normalize_name(self.name_mappings[normalized])

        return normalized

    def run(self, molecule_names: Union[List[str], str]):
        self.reset()
        if isinstance(molecule_names, str):
            molecule_names = [molecule_names]

        # Normalize molecule names
        normalized_names = [self.normalize_name(name) for name in molecule_names]

        try:
            out = self._run(normalized_names)
        except Exception as e:
            return self.get_output(e=e)

        response = f"""Successfully generated the molecule input files (and force field files) for:
{', '.join([file(name) for name in out.keys()])}
(IMPORTANT: use these exact names in the simulation.input file!)"""

        torsions = [name for name in out.keys() if out[name] is True]
        if len(torsions) > 0:
            response += f"The following molecules have torsions: {', '.join(torsions)}"
        else:
            response += "\nNone of the molecules has torsions."

        return self.get_output(content=response)


def molecule_loader(ctx: RunContext, molecule_names: Union[List[str], str]):
    """Generate the molecule definition (input) files and the corresponding force field and pseudoatoms files.
    Accepts common molecule names and chemical formulas such as:
    - Simple formulas: CO2, N2, O2, CH4, H2O, NH3, Ar, Kr, Xe, He
    - Common names: carbon dioxide, nitrogen, oxygen, methane, water, ammonia, argon, krypton, xenon, helium
    - Organic molecules: ethane, propane, butane, pentane, hexane, heptane, octane, benzene, toluene

    The tool will automatically map common abbreviations to their proper names."""
    path = ctx.deps["cwd"]
    return MoleculeLoader(path=path).run(molecule_names)