from pathlib import Path

import ase.io
import numpy as np
from torch_geometric.data import Batch

from matten.data.data import DataPoint

TEST_FILE_DIR = Path(__file__).resolve().parents[1].joinpath("test_files")


def test_AtomicData():
    natoms = 4
    pos = np.random.randn(natoms, 3)
    edge_index = np.random.randint(0, natoms - 1, (2, 6))
    x = {"species": np.asarray([0, 0, 1, 4]), "coords": pos}
    y = {"energy": np.asarray([0.1]), "forces": np.random.randn(natoms, 3)}

    data = DataPoint(pos, edge_index=edge_index, x=x, y=y)

    batch = Batch.from_data_list([data, data])

    print(data)
    print("\n\n\n")
    print(batch)
