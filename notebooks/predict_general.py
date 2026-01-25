"""
An example script make predictions of any tensor.
"""

from pymatgen.core import Structure

from matten.predict import predict


def get_structure():
    a = 5.46
    lattice = [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]
    basis = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    Si = Structure(lattice, ["Si", "Si"], basis)

    return Si


if __name__ == "__main__":
    structure = get_structure()

    tensor = predict( structure, is_elasticity_tensor=True)

    print("value:", tensor)
    print("type:", type(tensor))
    print("shape:", tensor.shape)
