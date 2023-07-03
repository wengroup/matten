"""
Test the tfn tensor model to ensure:
- equivariance: basically using the formula: $f(Qx) = Qf(x)$, where $f$ is the model
  and $x$ are the coords in input molecules.
"""
from pathlib import Path

import pandas as pd
import pytorch_lightning
import torch
from e3nn import o3
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Structure

from matten.dataset.structure_scalar_tensor import TensorDataModule
from matten.model_factory.tfn_scalar_tensor import create_model
from matten.utils import ToCartesian

TESTFILE_DIR = Path(__file__).parents[1]


def get_model(output_format="cartesian", output_formula="ijkl=jikl=klij"):
    hparams = {
        "species_embedding_dim": 32,
        "irreps_edge_sh": "0e + 1o + 2e + 3o + 4e",
        "num_radial_basis": 10,
        "radial_basis_start": 0.0,
        "radial_basis_end": 5.0,
        "radial_basis_type": "bessel",
        "num_layers": 3,
        "invariant_layers": 2,
        "invariant_neurons": 32,
        "average_num_neighbors": None,
        "conv_layer_irreps": "32x0o+32x0e + 16x1o+16x1e + 8x2o+8x2e + 4x3o+4x3e + 4x4o+4x4e",
        "nonlinearity_type": "gate",
        "normalization": None,
        "resnet": True,
        "conv_to_output_hidden_irreps_out": "2x0e + 2x2e + 4e",  # elastic tensor
        "output_format": output_format,
        "output_formula": output_formula,
        "reduce": "mean",
    }

    dataset_hyarmas = {"allowed_species": [8, 52]}  # O and Te

    model = create_model(hparams, dataset_hyarmas)

    return model


def load_dataset(
    filename, root, output_format="cartesian", output_formula="ijkl=jikl=klij"
):
    dm = TensorDataModule(
        trainset_filename=filename,
        valset_filename=filename,
        testset_filename=filename,
        r_cut=5.0,
        tensor_target_name="elastic_tensor_full",
        root=root,
        reuse=False,
        tensor_target_format=output_format,
        tensor_target_formula=output_formula,
        compute_dataset_statistics=False,
    )
    dm.setup()

    return dm.train_dataloader()


def _rotate_struct(filename):
    #
    # generate a file with rotated atoms, and get it data loader
    #

    # original mol
    df = pd.read_json(filename)
    struct = df["structure"][0]
    struct = Structure.from_dict(struct)

    # transformation matrix
    torch.manual_seed(35)
    Q = o3.rand_matrix()
    _symop = SymmOp.from_rotation_and_translation(rotation_matrix=Q.numpy())
    struct.apply_operation(_symop)

    # assign back to df
    df["structure"] = [struct.as_dict()]

    # write it out
    filename2 = "elastic_tensor_one-rotated.json"
    filename2 = Path("/tmp").joinpath(filename2)
    df.to_json(filename2)

    return filename2, Q


def test_model_equivariance():
    pytorch_lightning.seed_everything(35)

    output_format = "cartesian"
    output_formula = "ijkl=jikl=klij"
    model = get_model(output_format, output_formula)

    filename = TESTFILE_DIR.joinpath("test_files", "elastic_tensor_one.json")
    loader = load_dataset(filename, root="/tmp")

    filename2, Q = _rotate_struct(filename)

    loader_rotated = load_dataset(filename=filename2, root="/tmp")

    def get_result(model, loader):
        model.eval()

        with torch.no_grad():
            for batch in loader:
                graphs = batch.tensor_property_to_dict()
                out = model(graphs)
                out = out["my_model_output"]  # this is only the
                if output_format == "cartesian":
                    return ToCartesian(output_formula)(out)
                else:
                    return out

    # [0] to get that of the first materials
    pred_cart = get_result(model, loader)[0]
    pred_rotated_cart = get_result(model, loader_rotated)[0]

    # test symmetry ijkl=ijlk=klij
    assert torch.allclose(pred_cart, torch.swapaxes(pred_cart, 0, 1))
    assert torch.allclose(pred_cart, torch.swapaxes(pred_cart, 2, 3))
    assert torch.allclose(
        pred_cart, torch.swapaxes(torch.swapaxes(pred_cart, 0, 2), 1, 3)
    )

    # test equivariance
    x = torch.einsum("im, jn, kp, lq, mnpq -> ijkl", Q, Q, Q, Q, pred_cart)

    assert torch.allclose(x, pred_rotated_cart, atol=1e-4)
