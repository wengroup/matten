"""
Utilities to normalize a set of tensors.

This is for data standardization. Unlike scalars, where we can treat each component
separately and obtain statistics from them, tensors need to be treated at least on
the irreps level.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from sklearn.preprocessing import StandardScaler
from torchtyping import TensorType

from matten.data.data import Crystal


class Normalize(nn.Module):
    """
    Base class for tensor standardization.
    """

    def __init__(self, irreps: Union[str, Irreps]):
        super().__init__()

        self.irreps = Irreps(irreps)

    def forward(
        self, data: TensorType["batch", "D"]  # noqa: F821
    ) -> TensorType["batch", "D"]:  # noqa: F821
        """
        Transform the data.

        Args:
            data: tensors to normalize. `batch`: batch dim of the tensors; `D`
                dimension of the tensors (should be compatible with self.irreps).
        """
        raise NotImplementedError

    def inverse(
        self, data: TensorType["batch", "D"]  # noqa: F821
    ) -> TensorType["batch", "D"]:  # noqa: F821
        """
        Inverse transform the data.

        Args:
            data: tensors to normalize. `batch`: batch dim of the tensors; `D`
                dimension of the tensors (should be compatible with self.irreps).
        """
        raise NotImplementedError


# Based on e3nn BatchNorm
class MeanNormNormalize(Normalize):
    """
    Normalize tensors like e3nn BatchNorm:
    - scalars are normalized by subtracting mean and then dividing by norms
    - higher order irreps are normalized by dividing norms

    Note, each irrep is treated separated. For example, for irreps "3x1e", the norm
    is computed separated for each 1e.

    Args:
        irreps: irreps of the tensor to normalize.
        mean: means used for normalization. If None, need to call
            `self.compute_statistics()` first to generate it.
        norm: norm used for normalization, If None, need to call
            `self.compute_statistics()` first to generate it.
        normalization: {'component', 'norm'}
        reduce: {'mean', 'max'}
        eps: epsilon to avoid diving be zero error
        scale: scale factor to multiply by norm. Because the data to be normalized
            will divide the norm, a value smaller than 1 will result in wider data
            distribution after normalization and a value larger than 1 will result in
            tighter data distribution.
    """

    def __init__(
        self,
        irreps: Union[str, Irreps],
        mean: TensorType = None,
        norm: TensorType = None,
        normalization: str = "component",
        reduce: str = "mean",
        eps: float = 1e-5,
        scale: float = 1.0,
    ):
        super().__init__(irreps)

        self.normalization = normalization
        self.reduce = reduce
        self.eps = eps
        self.scale = scale

        # Cannot register None as buffer for mean and norm, which means this module
        # does not need them. As a result, we cannot load them via state dict.
        if mean is None or norm is None:
            self.mean_norm_initialized = False
        else:
            self.mean_norm_initialized = True

        if mean is None:
            mean = torch.zeros(self.irreps.dim)
        if norm is None:
            norm = torch.zeros(self.irreps.dim)

        self.register_buffer("mean", mean)
        self.register_buffer("norm", norm)

    def forward(
        self, data: TensorType["batch", "D"]  # noqa: F821
    ) -> TensorType["batch", "D"]:  # noqa: F821
        if not self.mean_norm_initialized:
            raise RuntimeError("mean and norm not initialized.")

        return (data - self.mean) / (self.norm * self.scale)

    def inverse(
        self, data: TensorType["batch", "D"]  # noqa: F821
    ) -> TensorType["batch", "D"]:  # noqa: F821
        if not self.mean_norm_initialized:
            raise RuntimeError("mean and norm not initialized.")

        return data * (self.norm * self.scale) + self.mean

    # mean and norm
    def load_state_dict(
        self, state_dict: "OrderedDict[str, Tensor]", strict: bool = True
    ):
        super().load_state_dict(state_dict, strict)
        self.mean_norm_initialized = True

    def compute_statistics(self, data: TensorType["batch", "D"]):  # noqa: F821
        """
        Compute the mean and norm statistics.
        """

        all_mean = []
        all_norm = []

        ix = 0

        for mul, ir in self.irreps:  # mul: multiplicity, ir: an irrep
            d = ir.dim
            field = data[:, ix : ix + mul * d]  # [batch, mul * repr]
            ix += mul * d

            field = field.reshape(-1, mul, d)  # [batch, mul, repr]

            if ir.is_scalar():
                # compute mean of scalars (higher order tensors does not use mean)
                field_mean = field.mean(dim=0).reshape(mul)  # [mul]

                # subtract mean for computing stdev as norm below
                field = field - field_mean.reshape(-1, mul, 1)
            else:
                # set mean to zero for high order tensors
                field_mean = torch.zeros(mul)

            # expand to the repr dimension, shape [mul*repr]
            field_mean = torch.repeat_interleave(field_mean, repeats=d, dim=0)
            all_mean.append(field_mean)

            #
            # compute the rescaling factor (norm of each feature vector)
            #
            # 1. rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                field_norm = field.pow(2).sum(dim=-1)  # [batch, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(dim=-1)  # [batch, mul]
            else:
                raise ValueError(f"Invalid normalization option {self.normalization}")

            # 2. reduction method
            if self.reduce == "mean":
                field_norm = field_norm.mean(dim=0)  # [mul]
            elif self.reduce == "max":
                field_norm = field_norm.max(dim=0)  # [mul]
            else:
                raise ValueError("Invalid reduce option {}".format(self.reduce))

            # Then apply the rescaling
            # divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(0.5)  # [mul]

            # expand to the repr dimension, shape [mul*repr]
            field_norm = torch.repeat_interleave(field_norm, repeats=d, dim=0)
            all_norm.append(field_norm)

        dim = data.shape[-1]
        assert (
            ix == dim
        ), f"`ix` should have reached data.size(-1)={dim}, but it ended at {ix}"

        all_mean = torch.cat(all_mean)  # [dim]
        all_norm = torch.cat(all_norm)  # [dim]

        assert len(all_mean) == dim, (
            f"Expect len(all_mean) and data.shape[-1] to be equal; got {len(all_mean)} "
            f"and {dim}."
        )
        assert len(all_norm) == dim, (
            f"Expect len(all_norm) and data.shape[-1] to be equal; got {len(all_norm)} "
            f"and {dim}."
        )

        # Warning, do not delete this line
        self.load_state_dict({"mean": all_mean, "norm": all_norm})

        return all_mean, all_norm


class ScalarNormalize(nn.Module):
    """
    Normalize scalar quantities of shape [num_samples, num_features], each feature
    is normalized individually.

    Args:
        num_features: feature dim for the data to be normalized.
        scale: scale factor to multiply by norm. Because the data to be normalized
            will divide the norm, a value smaller than 1 will result in wider data
            distribution after normalization and a value larger than 1 will result in
            tighter data distribution.

    """

    def __init__(
        self,
        num_features: int,
        mean: TensorType = None,
        norm: TensorType = None,
        scale: float = 1.0,
    ):
        super().__init__()

        self.scale = scale

        # Cannot register None as buffer for mean and norm, which means this module
        # does not need them. As a result, we cannot load them via state dict.
        if mean is None or norm is None:
            self.mean_norm_initialized = False
        else:
            self.mean_norm_initialized = True

        if mean is None:
            mean = torch.zeros(num_features)
        if norm is None:
            norm = torch.zeros(num_features)

        self.register_buffer("mean", mean)
        self.register_buffer("norm", norm)

    def forward(
        self, data: TensorType["batch", "D"]  # noqa: F821
    ) -> TensorType["batch", "D"]:  # noqa: F821
        if not self.mean_norm_initialized:
            raise RuntimeError("mean and norm not initialized.")

        return (data - self.mean) / (self.norm * self.scale)

    def inverse(
        self, data: TensorType["batch", "D"]  # noqa: F821
    ) -> TensorType["batch", "D"]:  # noqa: F821
        if not self.mean_norm_initialized:
            raise RuntimeError("mean and norm not initialized.")

        return data * (self.norm * self.scale) + self.mean

    # mean and norm
    def load_state_dict(
        self, state_dict: "OrderedDict[str, Tensor]", strict: bool = True
    ):
        super().load_state_dict(state_dict, strict)
        self.mean_norm_initialized = True

    def compute_statistics(self, data: TensorType["batch", "D"]):  # noqa: F821
        """
        Compute the mean and norm statistics.
        """

        assert data.ndim == 2, "Can only deal with tensor [N_samples, N_features]"

        dtype = data.dtype
        data = data.numpy()

        scaler = StandardScaler()
        scaler.fit(data)
        mean = scaler.mean_
        std = scaler.scale_

        mean = torch.as_tensor(mean, dtype=dtype)
        std = torch.as_tensor(std, dtype=dtype)

        # Warning, do not delete this line,
        self.load_state_dict({"mean": mean, "norm": std})

        return mean, std


class ScalarFeatureTransform(torch.nn.Module):
    """
    Forward and inverse normalization of scalar features in datapoint.x.

    As long as each feature in datapoint.x is a 2D array, this should work for both
    atom features (different number of rows, one for each atom) and global feature
    (only one for the crystal).

    Forward is intended to be used as `pre_transform` of dataset.
    Inverse is not intended to be called.

    Args:
        dataset_statistics_path: path to the dataset statistics file. Will be delayed to
            load when the forward or inverse function is called.
        feature_names: names of the feature
        feature_sizes: size of each feature

    """

    def __init__(
        self,
        feature_names: List[str],
        feature_sizes: List[int],
        dataset_statistics_path: Union[str, Path] = None,
    ):
        super().__init__()
        self.dataset_statistics_path = dataset_statistics_path
        self.dataset_statistics_loaded = False

        self.feature_names = feature_names
        self.feature_sizes = feature_sizes

        self.normalizers = torch.nn.ModuleDict(
            {
                name: ScalarNormalize(num_features=size)
                for name, size in zip(self.feature_names, self.feature_sizes)
            }
        )

    def forward(self, struct: Crystal) -> Crystal:
        """
        Update feature (i.e. x) of Crystal.

        Note, should return it even we modify it in place.
        """

        # Instead of providing mean and norm of normalizer at instantiation, we delay
        # the loading here because this method will typically be called after dataset
        # statistics has been generated.
        self._fill_state_dict(struct.x[self.feature_names[0]].device)

        for name in self.feature_names:
            feat = struct.x[name]
            struct.x[name] = self.normalizers[name](feat)

        return struct

    def inverse(self, data: TensorType["batch", "D"], name: str):
        raise RuntimeError("This is supposed not to be called.")

    def _fill_state_dict(self, device):
        """
        Because this is delayed be to called when actual forward or inverse is
        happening, need to move the tensor to the correct device.
        """
        if not self.dataset_statistics_loaded:
            statistics = torch.load(self.dataset_statistics_path)
            for name in self.feature_names:
                self.normalizers[name].load_state_dict(statistics[name])
            self.to(device)

            self.dataset_statistics_loaded = True

    def compute_statistics(self, data: List[Crystal]) -> Dict[str, Any]:
        """Compute statistics of datasets.

        This requires each feature in Crystal.x is a 2D array. The features
        from different crystals are concatenated into a single 2D array, and the
        statistics are individually for each column.

        Args:
            data: list of Crystal objects.

        Returns:
            A dictionary of statistics.
        """

        features = defaultdict(list)
        atomic_numbers = set()
        num_neigh = []

        for struct in data:
            for key in self.feature_names:
                t = struct.x[key]
                features[key].append(t)
            atomic_numbers.update(struct.atomic_numbers.tolist())
            num_neigh.append(struct.num_neigh)

        statistics = {
            "allowed_species": tuple(atomic_numbers),
            "average_num_neigh": torch.cat(num_neigh).mean(),
        }

        for key in self.feature_names:
            t = torch.cat(features[key])
            assert t.ndim == 2
            self.normalizers[key].compute_statistics(t)
            statistics[key] = self.normalizers[key].state_dict()

        return statistics


class ScalarTargetTransform(torch.nn.Module):
    """
    Forward and inverse normalization of scalar target.

    Forward is intended to be used as `pre_transform` of dataset and inverse is
    intended to be used before metrics and prediction function to transform the
    hessian back to the original space.

    Args:
        dataset_statistics_path: path to the dataset statistics file. Will be delayed to
            load when the forward or inverse function is called.
        target_names: names of the target

    """

    def __init__(
        self,
        target_names: List[str],
        dataset_statistics_path: Union[str, Path] = None,
    ):
        super().__init__()
        self.dataset_statistics_path = dataset_statistics_path
        self.dataset_statistics_loaded = False

        self.target_names = target_names

        self.normalizers = torch.nn.ModuleDict(
            {name: ScalarNormalize(num_features=1) for name in self.target_names}
        )

    def forward(self, struct: Crystal) -> Crystal:
        """
        Update target of Crystal.

        Note, should return it even we modify it in place.
        """

        # Instead of providing mean and norm of normalizer at instantiation, we delay
        # the loading here because this method will typically be called after dataset
        # statistics has been generated.
        self._fill_state_dict(struct.y[self.target_names[0]].device)

        for name in self.target_names:
            target = struct.y[name]
            struct.y[name] = self.normalizers[name](target)

        return struct

    def inverse(self, data: TensorType["batch", "D"], target_name: str):
        """
        Inverse transform model predictions/targets.

        This is supposed to be called in batch mode.
        """
        self._fill_state_dict(data.device)
        data = self.normalizers[target_name].inverse(data)

        return data

    def _fill_state_dict(self, device):
        """
        Because this is delayed be to called when actual forward or inverse is
        happening, need to move the tensor to the correct device.
        """
        if not self.dataset_statistics_loaded:
            statistics = torch.load(self.dataset_statistics_path)
            for name in self.target_names:
                self.normalizers[name].load_state_dict(statistics[name])
            self.to(device)

            self.dataset_statistics_loaded = True

    def compute_statistics(self, data: List[Crystal]) -> Dict[str, Any]:
        """
        Compute statistics of datasets.
        """

        targets = defaultdict(list)
        atomic_numbers = set()
        num_neigh = []

        for struct in data:
            for key in self.target_names:
                t = struct.y[key]
                targets[key].append(t)
            atomic_numbers.update(struct.atomic_numbers.tolist())
            num_neigh.append(struct.num_neigh)

        statistics = {
            "allowed_species": tuple(atomic_numbers),
            "average_num_neigh": torch.cat(num_neigh).mean(),
        }

        for key in self.target_names:
            t = torch.cat(targets[key])
            assert t.ndim == 2
            self.normalizers[key].compute_statistics(t)
            statistics[key] = self.normalizers[key].state_dict()

        return statistics


class TensorTargetTransform(torch.nn.Module):
    """
    Forward and inverse normalization of tensors.

    Forward is intended to be used as `pre_transform` of dataset and inverse is
    intended to be used before metrics and prediction function to transform the
    hessian back to the original space.

    Args:
        dataset_statistics_path: path to the dataset statistics file. Will be delayed to
            load when the forward or inverse function is called.
    """

    def __init__(
        self,
        target_name: str = "elastic_tensor_full",
        dataset_statistics_path: Union[str, Path] = None,
        scale: float = 1.0,
        irreps: str = "2x0e+2x2e+4e",
    ):
        super().__init__()
        self.dataset_statistics_path = dataset_statistics_path
        self.dataset_statistics_loaded = False

        self.target_name = target_name

        self.normalizer = MeanNormNormalize(irreps=irreps, scale=scale)

    def forward(self, struct: Crystal) -> Crystal:
        """
        Update target of Crystal.

        Note, should return it even we modify it in place.
        """

        # Instead of providing mean and norm of normalizer at instantiation, we delay
        # the loading here because this method will typically be called after dataset
        # statistics has been generated.

        self._fill_state_dict(struct.y[self.target_name].device)

        target = struct.y[self.target_name]  # shape (21,)
        struct.y[self.target_name] = self.normalizer(target)

        return struct

    def inverse(self, data: TensorType["batch", "D"]):
        """
        Inverse transform model predictions/targets.

        This is supposed to be called in batch mode.
        """
        self._fill_state_dict(data.device)
        data = self.normalizer.inverse(data)

        return data

    def _fill_state_dict(self, device):
        """
        Because this is delayed be to called when actual forward or inverse is
        happening, need to move the tensor to the correct device.
        """
        if not self.dataset_statistics_loaded:
            if self.dataset_statistics_path is None:
                raise ValueError("Cannot load dataset statistics from file `None`")
            statistics = torch.load(self.dataset_statistics_path)
            self.normalizer.load_state_dict(statistics[self.target_name])
            self.to(device)

            self.dataset_statistics_loaded = True

    def compute_statistics(self, data: List[Crystal]) -> Dict[str, Any]:
        """
        Compute statistics of datasets.
        """

        elastic_tensors = []
        atomic_numbers = set()
        num_neigh = []

        for struct in data:
            elastic_tensors.append(struct.y[self.target_name])
            atomic_numbers.update(struct.atomic_numbers.tolist())
            num_neigh.append(struct.num_neigh)

        elastic_tensors = torch.cat(elastic_tensors)
        atomic_numbers = tuple(atomic_numbers)
        average_num_neigh = torch.cat(num_neigh).mean()

        self.normalizer.compute_statistics(elastic_tensors)

        statistics = {
            self.target_name: self.normalizer.state_dict(),
            "allowed_species": atomic_numbers,
            "average_num_neigh": average_num_neigh,
        }

        return statistics


class TensorScalarTargetTransform(torch.nn.Module):
    """
    A Wrapper for forward and inverse normalization of tensors and scalars.
    """

    def __init__(
        self,
        *,
        tensor_target_name: Optional[str] = None,
        tensor_irreps: str = None,
        scalar_target_names: Optional[List[str]] = None,
        dataset_statistics_path: Union[str, Path] = None,
    ):
        super().__init__()
        if tensor_target_name is not None:
            self.tensor_normalizer = TensorTargetTransform(
                tensor_target_name, dataset_statistics_path, irreps=tensor_irreps
            )
        else:
            self.tensor_normalizer = None

        if scalar_target_names is not None:
            self.scalar_normalizers = ScalarTargetTransform(
                scalar_target_names, dataset_statistics_path
            )
        else:
            self.scalar_normalizers = None

    def forward(self, struct: Crystal) -> Crystal:
        """
        Update target of Crystal.

        Note, should return it even we modify it in place.
        """
        # these will modify struct in place
        if self.tensor_normalizer is not None:
            self.tensor_normalizer(struct)
        if self.scalar_normalizers is not None:
            self.scalar_normalizers(struct)

        return struct

    def inverse(self, data: TensorType["batch", "D"], target_name: str):
        """
        Inverse transform model predictions/targets.

        This is supposed to be called in batch mode.
        """
        if self.tensor_normalizer is not None:
            data = self.tensor_normalizer.inverse(data)
        if self.scalar_normalizers is not None:
            data = self.scalar_normalizers.inverse(data, target_name)

        return data

    def compute_statistics(self, data: List[Crystal]) -> Dict[str, Any]:
        """
        Compute statistics of datasets.
        """
        if self.tensor_normalizer is not None:
            tensor_statistics = self.tensor_normalizer.compute_statistics(data)
        else:
            tensor_statistics = {}
        if self.scalar_normalizers is not None:
            scalar_statistics = self.scalar_normalizers.compute_statistics(data)
        else:
            scalar_statistics = {}

        statistics = {**tensor_statistics, **scalar_statistics}

        return statistics


class FeatureTensorScalarTargetTransform(torch.nn.Module):
    """
    A Wrapper for forward and inverse normalization of feats, as well as tensor and
    scalar targets.
    """

    def __init__(
        self,
        *,
        feature_names: Optional[List[str]] = None,
        feature_sizes: Optional[List[int]] = None,
        tensor_target_name: Optional[str] = None,
        tensor_irreps: str = None,
        scalar_target_names: Optional[List[str]] = None,
        dataset_statistics_path: Union[str, Path] = None,
    ):
        super().__init__()

        if feature_names is not None:
            if feature_sizes is None:
                raise ValueError("Expect feats_size to be non-zero")
            self.feat_normalizer = ScalarFeatureTransform(
                feature_names=feature_names,
                feature_sizes=feature_sizes,
                dataset_statistics_path=dataset_statistics_path,
            )
        else:
            self.feat_normalizer = None

        if tensor_target_name is not None:
            assert tensor_irreps is not None, (
                "`tensor_irreps` needs to be provided when `tensor_target_name` is "
                " is used to normalized tensors"
            )
            self.tensor_normalizer = TensorTargetTransform(
                tensor_target_name,
                dataset_statistics_path=dataset_statistics_path,
                irreps=tensor_irreps,
            )
        else:
            self.tensor_normalizer = None

        if scalar_target_names is not None:
            self.scalar_normalizers = ScalarTargetTransform(
                scalar_target_names, dataset_statistics_path=dataset_statistics_path
            )
        else:
            self.scalar_normalizers = None

    def forward(self, struct: Crystal) -> Crystal:
        """
        Update target of Crystal.

        Note, should return it even we modify it in place.
        """
        # these will modify struct in place
        if self.feat_normalizer is not None:
            self.feat_normalizer(struct)
        if self.tensor_normalizer is not None:
            self.tensor_normalizer(struct)
        if self.scalar_normalizers is not None:
            self.scalar_normalizers(struct)

        return struct

    def inverse(self, data: TensorType["batch", "D"], target_name: str):
        """
        Inverse transform model predictions/targets.

        This is supposed to be called in batch mode.
        """
        # NOTE, no need to inverse transform feats
        if self.tensor_normalizer is not None:
            data = self.tensor_normalizer.inverse(data)
        if self.scalar_normalizers is not None:
            data = self.scalar_normalizers.inverse(data, target_name)

        return data

    def compute_statistics(self, data: List[Crystal]) -> Dict[str, Any]:
        """Compute statistics of datasets."""
        if self.feat_normalizer is not None:
            feat_statistics = self.feat_normalizer.compute_statistics(data)
        else:
            feat_statistics = {}

        if self.tensor_normalizer is not None:
            tensor_statistics = self.tensor_normalizer.compute_statistics(data)
        else:
            tensor_statistics = {}

        if self.scalar_normalizers is not None:
            scalar_statistics = self.scalar_normalizers.compute_statistics(data)
        else:
            scalar_statistics = {}

        statistics = {**feat_statistics, **tensor_statistics, **scalar_statistics}

        return statistics
