import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import tqdm
from pymatgen.analysis.elasticity import ElasticTensor
from pymatgen.core import Element
from pymatgen.core.structure import Structure
from torch_geometric.loader import DataLoader

from matten.dataset.structure_scalar_tensor import TensorDatasetPrediction
from matten.log import set_logger
from matten.model_factory.tfn_scalar_tensor import ScalarTensorModel
from matten.utils import CartesianTensorWrapper, yaml_load


def get_pretrained_model_dir(identifier: str) -> Path:
    """
    Get the directory of the pretrained model.

    Args:
        identifier: if it is a path, return the path. Otherwise, return the path to
            the directory of the pretrained model.
    """
    if Path(identifier).exists() and Path(identifier).is_dir():
        return Path(identifier)
    else:
        return Path(__file__).parent.parent.parent / "pretrained" / identifier


def get_pretrained_model(identifier: str, checkpoint: str = "model_final.ckpt"):
    directory = get_pretrained_model_dir(identifier)
    model = ScalarTensorModel.load_from_checkpoint(
        checkpoint_path=directory.joinpath(checkpoint).as_posix(),
        map_location="cpu",
    )
    return model


def get_pretrained_config(identifier: str, config_filename: str = "config_final.yaml"):
    directory = get_pretrained_model_dir(identifier)
    config = yaml_load(directory / config_filename)

    return config


def get_data_loader(
    structures: list[Structure], identifier: str, batch_size: int = 200
) -> DataLoader:
    # config contains info for dataset and data loader, we only use the dataset part,
    # and adjust some parameters
    config = get_pretrained_config(identifier)
    config = config["data"].copy()

    for k in [
        "loader_kwargs",
        "root",
        "trainset_filename",
        "valset_filename",
        "testset_filename",
        "compute_dataset_statistics",
    ]:
        config.pop(k)

    r_cut = config.pop("r_cut")
    config["dataset_statistics_fn"] = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        # The filename is not input filename, but name for processed data.
        # The input is from `structures`.
        # The filename extension does not matter, it will be replaced by `_data.pt`
        root = tmp_dir
        filename = "data_for_prediction.txt"

        dataset = TensorDatasetPrediction(
            root=root,
            filename=filename,
            r_cut=r_cut,
            structures=structures,
            **config,
        )

    return DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)


def check_species(model, structures: list[Structure]):
    """
    Check if the species in the structures are support by the model.
    """
    supported_numbers = set(model.hparams["dataset_hparams"]["allowed_species"])
    for i, s in enumerate(structures):
        species = set(s.species)
        numbers = {s.number for s in species}

        if not numbers.issubset(supported_numbers):
            formula = s.composition.reduced_formula
            not_supported = [Element.from_Z(n) for n in numbers - supported_numbers]
            not_supported = [f"{s.symbol} ({s.number})" for s in not_supported]
            not_supported = ", ".join(not_supported)
            raise RuntimeError(
                f"Cannot make predictions for structure {i} ({formula}). It contains "
                f"species {not_supported} not supported by the model. "
                f"The model were trained with species {supported_numbers}."
            )


def evaluate(
    model,
    loader,
    target_name: str = "elastic_tensor_full",
) -> list[torch.Tensor]:
    """
    Evaluate the model to generate predictions.

    Args:
        model: the model to evaluate.
        loader: the data loader.
        target_name: the name of the target property.

    Returns:
        a list of predicted elastic tensors.
    """

    converter = CartesianTensorWrapper("ijkl=jikl=klij")

    predictions = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            preds, _ = model(batch, task_name=target_name)
            p = preds[target_name]
            p = converter.to_cartesian(p)
            predictions.extend(p)

    return predictions


def predict(
    structure: Structure | list[Structure],
    model_identifier="20230627",
    batch_size: int = 200,
    logger_level: str = "ERROR",
) -> ElasticTensor | list[ElasticTensor]:
    f"""
    Predict the property of a structure or a list of structures.

    Args:
        structure: a structure or a list of structures to predict.
        model_identifier: the identifier of the model to use. All models are placed in
            `matten/pretrained/{model_identifier}`.
        batch_size: the batch size for prediction. In general, the larger the faster,
            but it may be limited by the CPU memory.
        logger_level: the level of the logger. Pptions are `DEBUG`, `INFO`, `WARNING`,
            `ERROR`, and `CRITICAL`.


    Returns:
        Predicted elastic tensor(s). `None` if the model cannot make prediction for a
            structure.
    """
    set_logger(logger_level)

    if isinstance(structure, Structure):
        structure = [structure]
        single_struct = True
    else:
        single_struct = False

    model = get_pretrained_model(identifier=model_identifier)
    check_species(model, structure)
    loader = get_data_loader(structure, model_identifier, batch_size=batch_size)

    predictions = evaluate(model, loader)
    predictions = [ElasticTensor(t) for t in predictions]

    # deal with failed entries
    failed = set(loader.dataset.failed_entries)

    if failed:
        idx = 0
        elastic_tensors = []
        for i in range(len(structure)):
            if i in failed:
                elastic_tensors.append(None)
            else:
                elastic_tensors.append(predictions[idx])
                idx += 1

        warnings.warn(
            "Cannot make predictions for the following structures. Their returned "
            f"elasticity tensor set to `None`: {sorted(failed)}."
        )
    else:
        elastic_tensors = predictions

    if single_struct:
        return elastic_tensors[0]
    else:
        return elastic_tensors


if __name__ == "__main__":
    struct = Structure(
        lattice=np.asarray(
            [
                [3.348898, 0.0, 1.933487],
                [1.116299, 3.157372, 1.933487],
                [0.0, 0.0, 3.866975],
            ]
        ),
        species=["Si", "Si"],
        coords=[[0.25, 0.25, 0.25], [0, 0, 0]],
    )
    structures = [struct, struct]

    elasticity = predict(structures, batch_size=2)

    print(elasticity)
