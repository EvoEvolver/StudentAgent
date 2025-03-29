from rdkit import Chem
from rdkit.Chem import Draw

# Load the molecule from a PDB file
mol = Chem.MolFromPDBFile("/Users/zijian/PycharmProjects/StudentAgent/Movies/System_0/Movie_Box_1.1.1_300.000000_0.000000_component_methane_0.pdb", removeHs=False)

if mol:
    # Generate a 2D depiction
    mol_2d = Chem.rdDepictor.Compute2DCoords(mol)

    # Draw the molecule and save it as an image
    img = Draw.MolToImage(mol, size=(1000, 1000))
    img.save("molecule.png")

    print("Image saved as molecule.png")
else:
    print("Failed to read PDB file.")