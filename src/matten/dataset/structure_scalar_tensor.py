import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from e3nn.io import CartesianTensor
from monty.serialization import loadfn
from pymatgen.core.structure import Structure

from matten.data.data import Crystal
from matten.data.datamodule import BaseDataModule
from matten.data.dataset import InMemoryDataset
from matten.data.transform import FeatureTensorScalarTargetTransform
from matten.utils import CartesianTensorWrapper


class TensorDataset(InMemoryDataset):
    """
    A dataset for tensors (e.g. elastic) and derived scalar properties (e.g. bulk
    modulus).

    This also provide the possibility to use global features.

    Args:
        filename: name of input data file.
        r_cut: neighbor cutoff distance, in unit Angstrom.
        tensor_target_name: name of the tensor target. If `None`, the tensor is not
            used as a training target.
        tensor_target_format: {`cartesian`, `irreps`}. The target tensor in the
            data file is provided in cartesian or irreps format.
        tensor_target_formula: formula specifying symmetry of tensor, e.g.
            `ijkl=jikl=klij` for a elastic tensor.
        tensor_target_scale: multipler to be multiplied to the tensor target.
        normalize_tensor_target: whether to normalize the tensor target.
        scalar_target_names: names of the target scalar properties. If `None`, no scalar
            s used as training target.
        log_scalar_targets: whether to log transform the scalar targets; one for each
            scalar target given in `scalar_target_names`. Note, log is performed before
            computing any statistics, controlled by `normalize_scalar_targets`.
            If `None`, no log is performed.
        normalize_scalar_targets: whether to normalize the scalar targets; one for each
            scalar target given in `scalar_target_names`. If `None`, no normalize is
            performed.
        tensor_target_weight:  weight of the targets, {column_name: {value: weight}}.
            `column_name` is the column in the df to look for 1/0 value and weight.
            So, when the value of `column_name` matches `value`, it will have the
            corresponding `weight`. For example, given {'minLC_less_than_0': {1:
            10.0, 0:1.0}}, then for data points with minLC less than 0, it will have
            a weight of 10, and for those with minLC larger than 0, it will have a
            weight of 1. By default `None` means all data points has a weight of 1.
        global_featurizer: featurizer to compute global features.
        normalize_global_features: whether to normalize the global feature.
        atom_featuruzer:  featurizer to compute atom features.
        normalize_atom_features: whether to normalize the atom feature.
        root: root directory that stores the input and processed data.
        reuse: whether to reuse the preprocessed data.
        dataset_statistics_fn: callable to compute dataset statistics. If `None`, do not
            compute statistics. Note this is typically used together with
            `normalize_tensor_target`, `normalize_scalar_targets`,
            and `normalize_global_features`. However, as long as `dataset_statistics_fn`
            is provided, the statistics will be computed, and a file named
            `dataset_statistics.pt` is generated in $CWD. Whether to use the computed
            dataset statistics for normalization is determined by the normalize flags.
    """

    def __init__(
        self,
        filename: str,
        r_cut: float,
        *,
        tensor_target_name: str = "elastic_tensor_full",
        tensor_target_format: str = "irreps",
        tensor_target_formula: str = "ijkl=jikl=klij",
        tensor_target_scale: float = 1.0,
        normalize_tensor_target: bool = False,
        scalar_target_names: List[str] = None,
        log_scalar_targets: List[bool] = None,
        normalize_scalar_targets: List[bool] = None,
        tensor_target_weight: dict[str, dict[str, float]] = None,
        global_featurizer: None = None,
        normalize_global_features: bool = False,
        atom_featurizer: None = None,
        normalize_atom_features: bool = False,
        root: Union[str, Path] = ".",
        reuse: bool = True,
        dataset_statistics_fn: Callable = None,
    ):
        self.filename = filename
        self.r_cut = r_cut

        self.tensor_target_name = tensor_target_name
        self.tensor_target_format = tensor_target_format
        self.tensor_target_formula = tensor_target_formula
        self.tensor_target_scale = tensor_target_scale
        self.normalize_tensor_target = normalize_tensor_target

        self.tensor_target_weight = tensor_target_weight

        self.scalar_target_names = (
            [] if scalar_target_names is None else scalar_target_names
        )
        self.log_scalar_targets = (
            [False] * len(self.scalar_target_names)
            if log_scalar_targets is None
            else log_scalar_targets
        )
        self.normalize_scalar_targets = (
            [False] * len(self.scalar_target_names)
            if normalize_scalar_targets is None
            else normalize_scalar_targets
        )

        self.global_featurizer = global_featurizer
        self.normalize_global_features = normalize_global_features
        self.atom_featurizer = atom_featurizer
        self.normalize_atom_features = normalize_atom_features

        # check compatibility
        # if normalize_scalar_targets is not None:
        #     raise ValueError(
        #         "`normalize_scalar_targets` should always be `None` when tensor "
        #         "is trained, since we need to get the derived scalar properties from "
        #         "the tensor in its original space. Then the predicted scalar will "
        #         "always be in the unscaled space."
        #     )
        # if log_scalar_targets is not None:
        #     raise ValueError(
        #         "`log_scalar_targets` should always be `None` when tensor "
        #         "is trained, since we need to get the derived scalar properties from "
        #         "the tensor in its original space. Then the predicted scalar will "
        #         "always be in the unscaled space."
        #     )

        if not self.tensor_target_name and not self.scalar_target_names:
            raise ValueError(
                "At least one of `tensor_target_name` and `scalar_target_names` "
                "should be provided."
            )

        if self.normalize_global_features and self.global_featurizer is None:
            raise ValueError(
                "`normalize_global_features=True`, but `global_featurizer=None`"
            )

        if self.normalize_atom_features and self.atom_featurizer is None:
            raise ValueError(
                "`normalize_atom_features=True`, but `atom_featurizer=None`"
            )

        processed_dirname = (
            f"processed_"
            f"tensor_name={self.tensor_target_name}-"
            f"tensor_format={self.tensor_target_format}-"
            f"normalize_tensor={self.normalize_tensor_target}-"
            f"scalar_names={'-'.join(self.scalar_target_names)}_"
            f"log_scalars={str(self.log_scalar_targets).replace(' ', '')}-"
            f"normalize_scalars={str(self.normalize_scalar_targets).replace(' ', '')}-"
            f"normalize_global_feats={self.normalize_global_features}-"
            f"normalize_atom_feats={self.normalize_atom_features}"
        )

        # Normalize tensor/scalar targets and global features
        if (
            self.normalize_tensor_target
            or any(self.normalize_scalar_targets)
            or self.normalize_global_features
            or self.normalize_atom_features
        ):
            if normalize_tensor_target:
                t_name = tensor_target_name
                t_irreps = str(CartesianTensor(self.tensor_target_formula))
            else:
                t_name = None
                t_irreps = None

            s_names = [
                s
                for s, n in zip(self.scalar_target_names, self.normalize_scalar_targets)
                if n
            ]
            if not s_names:
                s_names = None

            if self.normalize_global_features:
                f_names = ["global_feats"]
                f_sizes = [len(self.global_featurizer.feature_names)]
            else:
                f_names = None
                f_sizes = None

            if self.normalize_atom_features:
                raise ValueError("normalization not allowed")
                # af_names = ["atom_feats"]
                # af_sizes = [len(self.atom_featurizer.feature_names)]
                # if self.normalize_global_features:
                #     f_names += af_names
                #     f_sizes += af_sizes
                # else:
                #     f_names = af_names
                #     f_sizes = af_sizes

            pre_transform = FeatureTensorScalarTargetTransform(
                feature_names=f_names,
                feature_sizes=f_sizes,
                tensor_target_name=t_name,
                tensor_irreps=t_irreps,
                scalar_target_names=s_names,
                dataset_statistics_path="./dataset_statistics.pt",
            )
        else:
            pre_transform = None

        self._failed_entries = []

        super().__init__(
            filenames=[filename],
            root=root,
            processed_dirname=processed_dirname,
            reuse=reuse,
            dataset_statistics_fn=dataset_statistics_fn,
            pre_transform=pre_transform,
        )

    def get_data(self):
        filepath = self.raw_paths[0]
        df = pd.read_json(filepath)

        assert "structure" in df.columns, (
            f"Unsupported input data from file `{self.filename}`. Geometric "
            f"information (e.g. pymatgen Structure) is needed, but the dataset does "
            f"not have it."
        )

        # convert structure
        df["structure"] = df["structure"].apply(lambda s: Structure.from_dict(s))

        return self._get_crystals(df)

    def _get_crystals(self, df):
        # add global features
        if self.global_featurizer is not None:
            df = self.global_featurizer(df)
            feats = df[self.global_featurizer.feature_names].to_numpy().tolist()
            df["global_feats"] = feats

        # add atom features
        if self.atom_featurizer is not None:
            df = self.atom_featurizer(df)  # this will add `atom_feats` to the df

        # convert tensor target to tensor
        if self.tensor_target_name:
            df[self.tensor_target_name] = df[self.tensor_target_name].apply(
                lambda x: torch.as_tensor(x)
            )

            if self.tensor_target_format == "irreps":
                # TODO, convert to irreps tensor, assuming all input tensor is Cartesian
                converter = CartesianTensorWrapper(formula=self.tensor_target_formula)
                df[self.tensor_target_name] = df[self.tensor_target_name].apply(
                    lambda x: converter.from_cartesian(x).reshape(1, -1)
                )
            elif self.tensor_target_format == "cartesian":
                df[self.tensor_target_name] = df[self.tensor_target_name].apply(
                    lambda x: torch.unsqueeze(x, 0)
                )
            else:
                raise ValueError(
                    f"Unsupported target tensor format `{self.tensor_target_format}`"
                )

        # convert scalar targets to 2D shape
        for name in self.scalar_target_names:
            df[name] = df[name].apply(lambda y: torch.atleast_2d(torch.as_tensor(y)))

        # log scalar targets
        for name, do in zip(self.scalar_target_names, self.log_scalar_targets):
            if do:
                df[name] = df[name].apply(lambda y: torch.log(y))

        # all_targets, tensor and scalars
        target_columns = (
            [] if not self.tensor_target_name else [self.tensor_target_name]
        )
        target_columns += self.scalar_target_names

        crystals = []

        # convert to crystal data point
        for i, (irow, row) in enumerate(df.iterrows()):
            try:
                # get structure
                struct = row["structure"]

                # atomic numbers, shape (N_atom,)
                atomic_numbers = np.asarray(struct.atomic_numbers, dtype=np.int64)

                # get targets
                y = {name: row[name] for name in target_columns}
                if self.tensor_target_name:
                    y[self.tensor_target_name] = (
                        y[self.tensor_target_name] * self.tensor_target_scale
                    )

                x = None
                if self.global_featurizer:
                    # feats
                    gf = torch.as_tensor(row["global_feats"])
                    if torch.isnan(gf).any():
                        raise ValueError("NaN in global feats")
                    x = {
                        "global_feats": torch.reshape(
                            gf, (1, -1)
                        )  # reshape to a 2D tensor
                    }

                if self.atom_featurizer:
                    af = torch.as_tensor(row["atom_feats"]).reshape(len(struct), -1)
                    if torch.isnan(af).any():
                        raise ValueError("NaN in atom feats")
                    if x is None:
                        x = {"atom_feats": af}
                    else:
                        x["atom_feats"] = af

                # target weight
                if self.tensor_target_weight:
                    seen = False
                    for column_name, d in self.tensor_target_weight.items():
                        if seen:
                            raise ValueError(
                                "now, can only deal with one target " "weight"
                            )
                        seen = True

                        w = d[row[column_name]]
                        x["target_weight"] = torch.as_tensor([[w]])

                c = Crystal.from_pymatgen(
                    struct=struct,
                    r_cut=self.r_cut,
                    x=x,
                    y=y,
                    atomic_numbers=atomic_numbers,
                )
                crystals.append(c)

            except Exception as e:
                warnings.warn(
                    f"Failed converting structure {i} of index {irow}, "
                    f"Skip it. {str(e)}"
                )
                self._failed_entries.append(i)

        if not crystals:
            raise RuntimeError("Cannot successfully convert any structures.")

        return crystals

    @property
    def failed_entries(self):
        """
        Row index of the failed structures.
        """
        return self._failed_entries


class TensorDatasetPrediction(TensorDataset):
    """
    Tensor dataset for prediction.

    Compared to the base TensorDataset, this class:
    - directly takes a list of pymatgen Structure as input
    - does not require target values
    """

    def __init__(
        self,
        filename: str,  # this can be None
        r_cut: float,
        structures: list[Structure],
        **kwargs,
    ):
        self.structures = structures
        super().__init__(filename, r_cut, **kwargs)

    def get_data(self):
        # create a dummy target tensor
        zero_tensor = np.zeros((3, 3, 3, 3)).tolist()
        zero_tensors = [zero_tensor] * len(self.structures)

        df = pd.DataFrame(
            {"structure": self.structures, self.tensor_target_name: zero_tensors}
        )

        return self._get_crystals(df)


class TensorDataModule(BaseDataModule):
    def __init__(
        self,
        trainset_filename: str,
        valset_filename: str,
        testset_filename: str,
        *,
        r_cut: float,
        tensor_target_name: str = "elastic_tensor_full",
        tensor_target_format: str = "irreps",
        tensor_target_formula: str = "ijkl=jikl=klij",
        tensor_target_scale: float = 1.0,
        normalize_tensor_target: bool = False,
        tensor_target_weight: dict[str, dict[str, float]] = None,
        scalar_target_names: List[str] = None,
        log_scalar_targets: List[bool] = None,
        normalize_scalar_targets: List[bool] = None,
        global_featurizer: Union[str, list[str]] = None,
        normalize_global_features: bool = False,
        atom_featurizer: Union[str, list[str]] = None,
        normalize_atom_features: bool = False,
        root: Union[str, Path] = ".",
        reuse: bool = True,
        compute_dataset_statistics: bool = False,
        loader_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """

        Args:
            trainset_filename:
            valset_filename:
            testset_filename:
            r_cut:
            tensor_target_name:
            tensor_target_format:
            tensor_target_formula:
            normalize_tensor_target:
            scalar_target_names:
            log_scalar_targets:
            normalize_scalar_targets:
            global_featurizer: path to a .yaml file containing the names of global
                features.
            normalize_global_features:
            atom_featurizer: path to a .yaml file containing the names of atom features.
            root:
            reuse:
            compute_dataset_statistics:
            loader_kwargs:
        """
        self.r_cut = r_cut

        self.tensor_target_name = tensor_target_name
        self.tensor_target_formula = tensor_target_formula
        self.tensor_target_format = tensor_target_format
        self.tensor_target_scale = tensor_target_scale
        self.normalize_tensor_target = normalize_tensor_target
        self.tensor_target_weight = tensor_target_weight

        self.scalar_target_names = scalar_target_names
        self.log_scalar_targets = log_scalar_targets
        self.normalize_scalar_targets = normalize_scalar_targets

        self.global_featurizer = global_featurizer
        self.normalize_global_features = normalize_global_features
        self.atom_featurizer = atom_featurizer
        self.normalize_atom_features = normalize_atom_features

        self.compute_dataset_statistics = compute_dataset_statistics

        self.root = root

        super().__init__(
            trainset_filename,
            valset_filename,
            testset_filename,
            reuse=reuse,
            loader_kwargs=loader_kwargs,
        )

    def setup(self, stage: Optional[str] = None):
        # global featurizer
        if self.global_featurizer:
            raise NotImplementedError("global featurizer not implemented")

            if isinstance(self.global_featurizer, list):
                feature_names = self.global_featurizer
            elif isinstance(
                self.global_featurizer, (str, Path)
            ) and self.global_featurizer.endswith(".yaml"):
                feature_names = loadfn(self.global_featurizer)
            else:
                raise ValueError

            gf = GlobalFeaturizer(feature_names=feature_names)
            gf_name = ["global_feats"]
            gf_size = [len(gf.feature_names)]
        else:
            gf = None
            gf_name = None
            gf_size = None

        # atom featuruzer
        if self.atom_featurizer:
            if isinstance(
                self.atom_featurizer, (str, Path)
            ) and self.atom_featurizer.endswith(".yaml"):
                atom_feats = loadfn(self.atom_featurizer)
            else:
                raise ValueError

            af = PrecomputedAtomFeaturizer(features=atom_feats)
            af_name = ["atom_feats"]
            af_size = [len(af.feature_names)]
        else:
            af = None
            af_name = None
            af_size = None

        if gf and af:
            f_name = gf_name + af_name
            f_size = gf_size + af_size
        elif gf:
            f_name = gf_name
            f_size = gf_size
        elif af:
            f_name = af_name
            f_size = af_size
        else:
            f_name = None
            f_size = None

        if self.normalize_tensor_target:
            ttn = self.tensor_target_name
            tensor_irreps = str(CartesianTensor(self.tensor_target_formula))
        else:
            ttn = None
            tensor_irreps = None

        if self.compute_dataset_statistics:
            normalizer = FeatureTensorScalarTargetTransform(
                feature_names=f_name,
                feature_sizes=f_size,
                tensor_target_name=ttn,
                tensor_irreps=tensor_irreps,
                scalar_target_names=self.scalar_target_names,
                dataset_statistics_path=None,
            )
            statistics_fn = normalizer.compute_statistics
        else:
            statistics_fn = None

        self.train_data = TensorDataset(
            self.trainset_filename,
            r_cut=self.r_cut,
            tensor_target_name=self.tensor_target_name,
            tensor_target_format=self.tensor_target_format,
            tensor_target_formula=self.tensor_target_formula,
            tensor_target_scale=self.tensor_target_scale,
            normalize_tensor_target=self.normalize_tensor_target,
            scalar_target_names=self.scalar_target_names,
            log_scalar_targets=self.log_scalar_targets,
            normalize_scalar_targets=self.normalize_scalar_targets,
            tensor_target_weight=self.tensor_target_weight,
            global_featurizer=gf,
            normalize_global_features=self.normalize_global_features,
            atom_featurizer=af,
            normalize_atom_features=self.normalize_atom_features,
            root=self.root,
            reuse=self.reuse,
            dataset_statistics_fn=statistics_fn,
        )

        self.val_data = TensorDataset(
            self.valset_filename,
            r_cut=self.r_cut,
            tensor_target_name=self.tensor_target_name,
            tensor_target_format=self.tensor_target_format,
            tensor_target_formula=self.tensor_target_formula,
            tensor_target_scale=self.tensor_target_scale,
            normalize_tensor_target=self.normalize_tensor_target,
            scalar_target_names=self.scalar_target_names,
            log_scalar_targets=self.log_scalar_targets,
            normalize_scalar_targets=self.normalize_scalar_targets,
            tensor_target_weight=self.tensor_target_weight,
            global_featurizer=gf,
            normalize_global_features=self.normalize_global_features,
            atom_featurizer=af,
            normalize_atom_features=self.normalize_atom_features,
            root=self.root,
            reuse=self.reuse,
            dataset_statistics_fn=None,  # do not need to compute for valset
        )

        self.test_data = TensorDataset(
            self.testset_filename,
            r_cut=self.r_cut,
            tensor_target_name=self.tensor_target_name,
            tensor_target_format=self.tensor_target_format,
            tensor_target_formula=self.tensor_target_formula,
            tensor_target_scale=self.tensor_target_scale,
            normalize_tensor_target=self.normalize_tensor_target,
            scalar_target_names=self.scalar_target_names,
            log_scalar_targets=self.log_scalar_targets,
            normalize_scalar_targets=self.normalize_scalar_targets,
            tensor_target_weight=self.tensor_target_weight,
            global_featurizer=gf,
            normalize_global_features=self.normalize_global_features,
            atom_featurizer=af,
            normalize_atom_features=self.normalize_atom_features,
            root=self.root,
            reuse=self.reuse,
            dataset_statistics_fn=None,  # do not need to compute for valset
        )

    def get_to_model_info(self) -> Dict[str, Any]:
        atomic_numbers = set()
        num_neigh = []
        for data in self.train_dataloader():
            a = data.atomic_numbers.tolist()
            atomic_numbers.update(a)
            num_neigh.append(data.num_neigh)

        if self.global_featurizer:
            global_feats_size = data.x["global_feats"].shape[1]
        else:
            global_feats_size = None

        if self.atom_featurizer:
            atom_feats_size = data.x["atom_feats"].shape[1]
        else:
            atom_feats_size = None

        # .item to convert to float so that lightning cli can save it to yaml
        average_num_neighbors = torch.mean(torch.cat(num_neigh)).item()

        return {
            "allowed_species": tuple(atomic_numbers),
            "average_num_neighbors": average_num_neighbors,
            "global_feats_size": global_feats_size,
            "atom_feats_size": atom_feats_size,
        }
