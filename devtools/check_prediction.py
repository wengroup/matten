"""
Check the prediction script get the correct results as in fitting.

Using the same dataset as in fitting, we check the prediction script recovers
the MAE.
"""
from pathlib import Path

import pandas as pd
import torch
from pymatgen.core import Structure

from matten.predict import predict
from matten.utils import CartesianTensorWrapper


def get_data():
    filename = (
        Path(__file__).parent.parent
        / "datasets"
        / "example_crystal_elasticity_tensor_n100.json"
    )
    df = pd.read_json(filename)

    structures = df["structure"].apply(lambda x: Structure.from_dict(x)).tolist()

    targets = df["elastic_tensor_full"].tolist()
    targets = [torch.tensor(t) for t in targets]

    return structures, targets


def get_spherical_tensor(t: list[torch.Tensor]):
    """Convert a Cartesian tensor to a spherical tensor."""
    t = torch.stack(t)
    converter = CartesianTensorWrapper("ijkl=jikl=klij")
    spherical = converter.from_cartesian(t)
    return spherical


def mae(a: torch.Tensor, b: torch.Tensor):
    return torch.mean(torch.abs(a - b))


if __name__ == "__main__":
    structures, targets = get_data()
    predictions = predict(structures)
    print("Finish getting predictions")

    targets = get_spherical_tensor(targets)
    predictions = get_spherical_tensor([torch.tensor(t) for t in predictions])
    print("Finish get spherical tensor")

    print(mae(predictions, targets))
