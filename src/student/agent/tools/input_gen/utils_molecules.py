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
    url_smiles = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/SMILES/JSON"
    try:
        response = requests.get(url_smiles)
        response.raise_for_status()
        data = response.json()

        properties = data.get("PropertyTable", {}).get("Properties", [])
        if properties and "SMILES" in properties[0]:
            smiles = properties[0]["SMILES"]
        else:
            smiles = None

    except Exception as e:
        if type(e) == requests.exceptions.ConnectionError:
            print("No connection PubChem API could be established.")
        return None
    return smiles


def molecule_name_to_inchi(name: str):
    """
    Convert one molecule name to its Inchi using the PubChem API.
    Args:
        name (str): Molecule name that is used as query.
    Returns:
        inchi: The corresponding inchi string or None.
    """
    url_inchi = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/InChI/JSON"
    try:
        response = requests.get(url_inchi)
        response.raise_for_status()
        data = response.json()

        properties = data.get("PropertyTable", {}).get("Properties", [])
        if properties and "InChI" in properties[0]:
            inchi = properties[0]["InChI"]
        else:
            inchi = None
    except Exception as e:
        if type(e) == requests.exceptions.ConnectionError:
            print("No connection PubChem API could be established.")
        return None
    return inchi[6:]


def pubchem_id_to_property(id, property_name: str):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{id}/property/{property_name}/JSON"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        properties = data.get("PropertyTable", {}).get("Properties", [])
        if properties and property_name in properties[0]:
            p = [i[property_name] for i in properties]
        else:
            p = None
    except Exception as e:
        if type(e) == requests.exceptions.ConnectionError:
            print("No connection PubChem API could be established.")
        return None
    return p


def smiles_to_pubchem_id(smiles: str, same_isotope=True):
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastidentity/smiles/cids/txt"     
    if same_isotope:
        url += "?identity_type=same_isotope"
    try:
        response = requests.post(url, data={'smiles':smiles}).text.strip().split()
        
    except Exception as e:
        if type(e) == requests.exceptions.ConnectionError:
            print("No connection PubChem API could be established.")
        return None
    return response




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


def get_mol(name, partial_Hs=True, verbose=False):
    smiles = molecule_name_to_smiles(name)
    if verbose:
        print("SMILES found for ", name, " :" ,smiles)
    if smiles is None:
        return None
    mol = mol_from_smiles(smiles)

    if partial_Hs:
        mol = Chem.RemoveHs(mol)
        heteroatoms = [7, 8, 15, 16]
        hetero_idx = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in heteroatoms:
                hetero_idx.append(atom.GetIdx())
        if len(hetero_idx) > 0:
            mol = Chem.AddHs(mol, onlyOnAtoms=hetero_idx)
    return mol

