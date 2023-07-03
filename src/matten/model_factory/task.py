from pathlib import Path
from typing import Any, Dict, Union

from torch import Tensor

from matten.data.transform import ScalarTargetTransform, TensorTargetTransform
from matten.model.task import CanonicalRegressionTask


class ScalarRegressionTask(CanonicalRegressionTask):
    """
    Inverse transform prediction and target in metric.

    Note, in ScalarTargetTransform, the target are forward transformed.

    Args:
        name: name of the task. Values with this key in model prediction dict and
            target dict will be used for loss and metrics computation.
    """

    def __init__(
        self,
        name: str,
        loss_weight: float = 1.0,
        dataset_statistics_path: Union[str, Path] = "dataset_statistics.pt",
        normalize_target: bool = False,
        normalizer_kwargs: Dict[str, Any] = None,
    ):
        super().__init__(name, loss_weight=loss_weight)

        if normalizer_kwargs is None:
            normalizer_kwargs = {}
        if normalize_target:
            self.normalizer = ScalarTargetTransform(
                target_names=[name],
                dataset_statistics_path=dataset_statistics_path,
                **normalizer_kwargs,
            )
        else:
            self.normalizer = None

    def transform_target_loss(self, t: Tensor) -> Tensor:
        return t

    def transform_pred_loss(self, t: Tensor) -> Tensor:
        return t

    def transform_target_metric(self, t: Tensor) -> Tensor:
        if self.normalizer is not None:
            return self.normalizer.inverse(t, self.name)
        else:
            return t

    def transform_pred_metric(self, t: Tensor) -> Tensor:
        if self.normalizer is not None:
            return self.normalizer.inverse(t, self.name)
        else:
            return t


class TensorRegressionTask(CanonicalRegressionTask):
    """
    Inverse transform prediction and target in metric.

    Note, in TensorTargetTransform, the target are forward transformed.

    Args:
        name: name of the task. Values with this key in model prediction dict and
            target dict will be used for loss and metrics computation.
    """

    def __init__(
        self,
        name: str,
        loss_weight: float = 1.0,
        dataset_statistics_path: Union[str, Path] = "dataset_statistics.pt",
        normalize_target: bool = False,
        normalizer_kwargs: Dict[str, Any] = None,
    ):
        super().__init__(name, loss_weight=loss_weight)

        if normalizer_kwargs is None:
            normalizer_kwargs = {}
        if normalize_target:
            self.normalizer = TensorTargetTransform(
                target_name=name,
                dataset_statistics_path=dataset_statistics_path,
                **normalizer_kwargs,
            )
        else:
            self.normalizer = None

    def transform_target_loss(self, t: Tensor) -> Tensor:
        return t

    def transform_pred_loss(self, t: Tensor) -> Tensor:
        return t

    def transform_target_metric(self, t: Tensor) -> Tensor:
        if self.normalizer is not None:
            return self.normalizer.inverse(t)
        else:
            return t

    def transform_pred_metric(self, t: Tensor) -> Tensor:
        if self.normalizer is not None:
            return self.normalizer.inverse(t)
        else:
            return t
