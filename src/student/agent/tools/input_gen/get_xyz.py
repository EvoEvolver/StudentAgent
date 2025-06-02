import numpy as np

def convert_to_zmatrix(bonds, angles):
    zmatrix = []

    # Process bond lengths
    atom_map = {}  # Map atom indices to Z-matrix indices
    atom_count = 0
    for bond in bonds[1:]:  # Skip header row
        idx, pair, bond_type, length = bond
        atoms = pair.replace("'", "").replace('"',"").strip().split(" - ")
        atom1, atom2 = int(atoms[0]), int(atoms[1])

        if atom1 not in atom_map:
            atom_count += 1
            atom_map[atom1] = atom_count
            zmatrix.append([f"Atom{atom1}"])

        if atom2 not in atom_map:
            atom_count += 1
            atom_map[atom2] = atom_count
            zmatrix.append([f"Atom{atom2}", atom_map[atom1], float(length)])

    # Process bond angles
    for angle in angles[1:]:  # Skip header row
        idx, triplet, angle_type, theta, _ = angle
        atoms = triplet.replace("'", "").replace('"',"").strip().split(" - ")
        atom1, atom2, atom3 = int(atoms[0]), int(atoms[1]), int(atoms[2])

        if atom3 in atom_map:
            zmatrix[atom_map[atom3] - 1].extend([atom_map[atom2], float(theta)])

    return zmatrix


def zmat_to_xyz(zmat):
    """
    Convert a Z-matrix to Cartesian coordinates.

    Parameters:
        zmat (list of tuples): Each tuple defines an atom.
            - First atom: (label,)
            - Second atom: (label, ref1, distance)
            - Third atom: (label, ref1, distance, ref2, angle)
            - Subsequent atoms: (label, ref1, distance, ref2, angle, ref3, dihedral)

    Returns:
        list of tuples: Each tuple is (label, np.array([x, y, z]))
    """
    coords = []  # to store Cartesian coordinates
    labels = []  # to store atom labels

    for i, row in enumerate(zmat):
        if i == 0:
            # First atom at the origin.
            label = row[0]
            labels.append(label)
            coords.append(np.array([0.0, 0.0, 0.0]))
        elif i == 1:
            # Second atom: place along the x-axis.
            label, ref1, r = row
            labels.append(label)
            coords.append(np.array([r, 0.0, 0.0]))
        elif i == 2:
            # Third atom: place in the xy-plane.
            label, ref1, r, ref2, angle = row
            labels.append(label)
            angle = np.radians(angle)
            # For simplicity, we define this atom relative to the first reference atom.
            # Place the atom so that it forms the given angle with the bond defined by ref2 -> ref1.
            # Here we assume ref1 is the immediate bond connection.
            # We position the third atom relative to the second atom.
            x2, y2, z2 = coords[ref1 - 1]
            # Place in xy-plane:
            x = x2 - r * np.cos(angle)
            y = r * np.sin(angle)
            coords.append(np.array([x, y, 0.0]))
        else:
            # For subsequent atoms, use three reference atoms.
            label, ref1, r, ref2, angle, ref3, dihedral = row
            labels.append(label)
            angle = np.radians(angle)
            dihedral = np.radians(dihedral)

            # Get positions of the three reference atoms (convert from 1-indexed to 0-indexed)
            a = coords[ref1 - 1]  # primary reference for the bond
            b = coords[ref2 - 1]
            c = coords[ref3 - 1]

            # Build the local coordinate system based on the reference atoms:
            # e1 is along the bond from ref1 to a.
            e1 = (a - b)
            e1 /= np.linalg.norm(e1)

            # e2 is perpendicular to e1 in the plane defined by a, b, c.
            # First, form a vector from b to c.
            bc = (c - b)
            # Then get a vector perpendicular to e1.
            e2 = np.cross(e1, bc)
            e2 /= np.linalg.norm(e2)

            # e3 is perpendicular to both e1 and e2.
            e3 = np.cross(e1, e2)

            # In the local frame (centered at 'a'), the coordinates of the new atom are:
            #   x_local = -r*cos(angle)
            #   y_local = r*sin(angle)*cos(dihedral)
            #   z_local = r*sin(angle)*sin(dihedral)
            local = np.array([
                -r * np.cos(angle),
                r * np.sin(angle) * np.cos(dihedral),
                r * np.sin(angle) * np.sin(dihedral)
            ])

            # Transform the local coordinates to the global frame.
            global_vector = local[0] * e1 + local[1] * e2 + local[2] * e3
            new_coord = a + global_vector
            coords.append(new_coord)

    return list(zip(labels, coords))

def from_bonds_angles_to_xyz(bonds, angles):
    zmatrix = convert_to_zmatrix(bonds, angles)
    xyz_atoms = zmat_to_xyz(zmatrix)
    return xyz_atoms

def get_xyz_file(bond_lengths, bond_angles):
    return from_bonds_angles_to_xyz(bond_lengths, bond_angles)


if __name__ == '__main__1':

    # Example usage:
    zmat = [
        ("H",),  # Atom 1
        ("O", 1, 0.96),  # Atom 2: distance from Atom 1
        ("H", 2, 0.96, 1, 104.5),  # Atom 3: distance from Atom 2, angle with Atom 1
        # Additional atoms would follow the pattern:
        # ("X", ref1, r, ref2, angle, ref3, dihedral)
    ]

    xyz_atoms = zmat_to_xyz(zmat)
    for label, pos in xyz_atoms:
        print(f"{label}: {pos}")

if __name__ == '__main__':
    # Example input
    bonds = [['stretch', 'type', '"length [Ang.]"'], ['1', '"\'1 - 2\'"', 'CHx-OH', '1.43'],
             ['2', '"\'2 - 3\'"', 'O-H', '0.945']]
    angles = [['bend', 'type', '"theta [degrees]"', '"k_theta/kB [K/rad^2]"'],
              ['1', '"\'1 - 2 - 3\'"', 'CHx-(O)-H', '108.50', '55400']]

    # Convert and print
    zmatrix = convert_to_zmatrix(bonds, angles)
    #for line in zmatrix:
    #    print(" ".join(map(str, line)))

    # Convert to Cartesian coordinates
    xyz_atoms = zmat_to_xyz(zmatrix)
    for label, pos in xyz_atoms:
        print(f"{label}: {pos}")