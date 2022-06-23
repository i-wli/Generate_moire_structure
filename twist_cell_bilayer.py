import sys
import argparse
import numpy as np
import ase.io
from ase.atoms import Atoms
from ase.build import make_supercell

"""This is a version where to input superperiodicity and output twist angle"""

"""please check PHYSICAL REVIEW B 90, 155451 (2014) for theoretical background.
supercell 1 with (N,M) and (−M,N + M) vectors, followed by the twist rotation with +θ/2. 
supercell 2 with (M,N) and (−N,M + N) vectors, followed by the rotation with −θ/2
The commensurate tBLG with the twist angle θ and the periodicity Lcell is obtained"""

class Pyatoms():
    def __init__(self,vectors, positions, atomic_numbers):
        self.vectors = vectors
        self.positions = positions
        self.atomic_numbers = atomic_numbers
class theta():
    def __init__(self, radian, degree):
        self.radian = radian
        self.degree = degree

def calc_vectors_angle(v1, v2):
    """calculate the angle between two vectors"""
    v1dotv2 = np.dot(v1, v2)
    length_v1 = np.sqrt(np.dot(v1, v1))
    length_v2 = np.sqrt(np.dot(v2, v2))
    cos_theta = v1dotv2 / (length_v1 * length_v2)
    # In the case cos(theta) little larger than 1.0
    if float_eq(cos_theta, 1.0):
      cos_theta = 1.0
    angle = np.arccos(cos_theta)
    degree = angle * 180 /np.pi
    return theta(angle, degree)

FLOAT_PREC = 1e-4
def float_eq(f1, f2, prec=FLOAT_PREC):
    """float equal"""
    return abs(f1 - f2) < prec

def exit(contect='[error] Unknown: Something goes wrong...'):
    """exit the program with some error msg."""
    print(contect)
    sys.exit(1)    


    
def ase_atoms_to_py_atoms(atoms: ase.atoms.Atoms) -> Pyatoms:
    vectors = np.array(atoms.cell.copy())
    positions = atoms.positions
    atomic_numbers = atoms.numbers
    atoms = Pyatoms(vectors, positions, atomic_numbers)
    return atoms

def py_atoms_to_ase_atoms(atoms: Pyatoms) -> ase.atoms.Atoms:
    vectors = [[j for j in k] for k in atoms.vectors]
    positions = [[j for j in k] for k in atoms.positions]
    numbers = [i for i in atoms.atomic_numbers]
    atoms = Atoms(
        numbers=numbers, positions=positions, cell=vectors, pbc=[True, True, True]
    )
    return atoms    
    

def check_vectors(atoms: ase.atoms.Atoms):
    vectors = atoms.cell
    len_ang = atoms.get_cell_lengths_and_angles()
    if not (float_eq(vectors[0][2], 0.0) and
            float_eq(vectors[1][2], 0.0) and
            float_eq(vectors[2][0], 0.0) and
            float_eq(vectors[2][1], 0.0)):
        exit('[error] Input structures: c axis must be in z direction')
    if not (float_eq(len_ang[0], len_ang[1]) and
            float_eq(len_ang[5], 60)):
        exit('[error] This version is only for P6 (a=b & γ=60)')


def coord_cart2frac(cell_vecs, cart_vec):
    """Transfrom the cart coords to frac coords"""
    cell_vecs_inv = np.linalg.inv(cell_vecs)
    frac_vec = np.dot(cart_vec, cell_vecs_inv)
    return frac_vec

def coord_frac2cart(cell_vecs, frac_vec):
    """Transfrom the frac coords to cart coords"""
    return np.dot(frac_vec, cell_vecs)

def get_supercell_vecs(trans_2D, unit_vecs, super_z):
    """Get the supercell lattice"""
    super_2d = np.dot(trans_2D, unit_vecs[0:2,0:2])
    supercell_vecs = np.array([[super_2d[0,0], super_2d[0,1], 0],
                               [super_2d[1,0], super_2d[1,1], 0],
                               [0, 0, super_z]])
    return supercell_vecs

def lattice_points_in_supercell(supercell_matrix): #copy from ASE
    """Find all lattice points contained in a supercell.

    Adapted from pymatgen, which is available under MIT license:
    The MIT License (MIT) Copyright (c) 2011-2012 MIT & The Regents of the
    University of California, through Lawrence Berkeley National Laboratory
    """

    diagonals = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    d_points = np.dot(diagonals, supercell_matrix)

    mins = np.min(d_points, axis=0)
    maxes = np.max(d_points, axis=0) + 1

    ar = np.arange(mins[0], maxes[0])[:, None] * np.array([1, 0, 0])[None, :]
    br = np.arange(mins[1], maxes[1])[:, None] * np.array([0, 1, 0])[None, :]
    cr = np.arange(mins[2], maxes[2])[:, None] * np.array([0, 0, 1])[None, :]

    all_points = ar[:, None, None] + br[None, :, None] + cr[None, None, :]
    all_points = all_points.reshape((-1, 3))

    frac_points = np.dot(all_points, np.linalg.inv(supercell_matrix))

    tvects = frac_points[
        np.all(frac_points < 1 - 1e-10, axis=1)
        & np.all(frac_points >= -1e-10, axis=1)
    ]
    assert len(tvects) == round(abs(np.linalg.det(supercell_matrix)))
    return tvects

def get_supercell_atoms_pos(supercell_vecs: "np.array",
                            trans_2D: "np.array",
                            unit: ase.atoms.Atoms) -> ase.atoms.Atoms:
    trans_matrix = np.array([[trans_2D[0,0], trans_2D[0,1], 0],
                               [trans_2D[1,0], trans_2D[1,1], 0],
                               [0, 0, 1]])
    lattice_points_frac = lattice_points_in_supercell(trans_matrix)
    lattice_points = np.dot(lattice_points_frac, supercell_vecs)
    superatoms = Atoms(cell=supercell_vecs, pbc=unit.pbc)
    for lp in lattice_points:
        shifted_atoms = unit.copy()
        shifted_atoms.positions += lp
        superatoms.extend(shifted_atoms)
    return superatoms

def gen_supercell(bottom: ase.atoms.Atoms,
                  top: ase.atoms.Atoms,
                  tran_2D_b, tran_2D_t, 
                  super_z: float,
                  delta_z: float) -> ase.atoms.Atoms :
    """get supercell lattice"""
    vectors_b = get_supercell_vecs(tran_2D_b, bottom.cell, super_z)
    vectors_t = get_supercell_vecs(tran_2D_t, bottom.cell, super_z)
    """get atom position"""
    bottom_super = get_supercell_atoms_pos(vectors_b, tran_2D_b, bottom)
    bottom_super_pos = bottom_super.positions
    top_super = get_supercell_atoms_pos(vectors_t, tran_2D_t, top)
    top_super_frac = coord_cart2frac(vectors_t, top_super.positions)
    top_super_pos = coord_frac2cart(vectors_b, top_super_frac)
    #top_super_pos = np.dot(tran_2D_b,np.dot(np.linalg.inv(tran_2D_t), top_super.positions))
    top_super_pos[..., 2] = top_super_pos[..., 2] + delta_z
    positions = np.vstack((bottom_super_pos, top_super_pos))
    number = np.hstack((bottom_super.numbers, top_super.numbers))
    results = py_atoms_to_ase_atoms(Pyatoms(vectors_b, positions, number))
    return results


def get_parser():
    parser = argparse.ArgumentParser(description="A simply script for twist bilayers, could not gernerate heterostures")
    parser.add_argument('-b','--bottom', required=True, help='Path to lower layer, need to be recognized by ASE')
    parser.add_argument('-t','--top', required=True, help='Path to upper layer, need to be recognized by ASE')
    parser.add_argument('-M', required=True, help='supercell matrix with np.array([[M,N],[-N,M+N]])', type=int)
    parser.add_argument('-N', required=True, help='supercell matrix with np.array(([M,N],[-N,M+N]])', type=int)
    parser.add_argument('-z', help='super lattice of z direction, default = 20 A', type=float, default = 20.0)
    parser.add_argument('-d', help='distance of two layers, default = 4 A', type=float, default = 4.0)
    parser.add_argument('-w','--write', help='Path to write supercell, need to be recognized by ASE')
    parser.add_argument('-o','--outformat', help='output file-format, like lammps-data', default = None)
    parser.add_argument('-a','--angle', help='output twist angle with format of rad or deg, default = deg', default='deg', choices=['rad', 'deg'])
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    bottom = ase.io.read(args.bottom)
    top = ase.io.read(args.top)
    check_vectors(bottom)
    check_vectors(top)
    tran_2D_b = np.array([[args.M,args.N],[-args.N,args.M+args.N]])
    tran_2D_t = np.array([[args.N, args.M], [-args.M, args.M + args.N]])
    result = gen_supercell(bottom, top, tran_2D_b, tran_2D_t, args.z, args.d)
    theta = calc_vectors_angle(result.cell[0], bottom.cell[0])
    if args.angle == 'rad':
        print("generating twist supercell to arg.write with twist angle {}".format(np.pi/3 - 2*theta.radian))
    else:
        print("generating twist supercell to arg.write with twist angle {}".format(60 - 2*theta.degree))
    print("Supercell lattice size: {}".format(result.cell.cellpar()[0]))
    print("Number of atoms: {}".format(len(result)))
    if args.write == None:
        result.write("Super_{0:.2f}_{1}.xsf".format(60 - 2*theta.degree, len(result)))
    else:
        result.write(args.write, format=args.outformat)

