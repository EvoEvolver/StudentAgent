from rdkit import Chem
import requests
from rdkit.Chem import AllChem


def molecule_name_to_smiles(name: str):
    """
    Convert one molecule name to its Canonical SMILES using the PubChem API.
    Args:
        name (str): Molecule name that is used as query.
    Returns:
        smiles: The corresponding SMILES string or None.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        properties = data.get("PropertyTable", {}).get("Properties", [])
        if properties and "CanonicalSMILES" in properties[0]:
            smiles = properties[0]["CanonicalSMILES"]
        else:
            smiles = None
    except Exception as e:
        return None, None
    return smiles

def mol_from_smiles(smiles: str):
    """
    Convert a Canonical SMILES to an RDKit Mol object with connectivity.
    Args:
        smiles (str): SMILES string.
    Returns:
        mol: The Mol object with a 3D conformer.
    """
    mol = None
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        return None
    return mol

def get_mol(name):
    smiles = molecule_name_to_smiles(name)
    mol = mol_from_smiles(smiles)
    return mol

