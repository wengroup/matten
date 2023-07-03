"""
Base Lightning model for regression and classification.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.cli import instantiate_class
from torch import Tensor

from matten.data.data import DataPoint
from matten.model.task import Task, TaskType
from matten.model.utils import TimeMeter


class BaseModel(pl.LightningModule):
    """
    Base matten model for regression and classification tasks.

    This class accepts any type of data as batch. Subclass determines how the batch
    should be dealt with.

    Args:
        tasks: tasks that define the loss and metric. See matten.model.task
        backbone_hparams: hparams for the backbone model
        dataset_hparams: info from the dataset to initialize the model or tasks
        optimizer_hparams: hparams for the optimizer (e.g. Adam)
        lr_scheduler_hparams: hparams for the learning rate scheduler (e.g.
            ReduceLROnPlateau)
        trainer_hparams: trainer config params. These should not be used by the model,
            but to let the wandb logger log them. Then we can filter info on the wandb
            web interface.
        data_hparams: data config params. Similar to trainer_hparams, for the purpose
            of wandb filtering.

    Subclass must implement:
        - init_backbone(): create the underlying torch model
        - init_tasks(): create tasks that initialize the loss function and metrics
        - preprocess_batch(): preprocess the batched data to get input for the model
          and labels
        - decode(): compute model prediction using the torch model

    subclass may implement:
        - compute_loss(): compute the loss using model prediction and the target
    """

    # TODO, for `tasks`, instance of Task will be passed.
    #  When saving checkpoint, the instantiated object will be saved, and when
    #  loading it back, the object will be loaded (although at a different address).
    #  This can work without any problem, but we may want to pass hyperparams for
    #  tasks in and instantiate it in the model (just as optimizer_hparams).
    def __init__(
        self,
        tasks: Union[Task, List[Task], Dict[str, Task]] = None,
        backbone_hparams: Dict[str, Any] = None,
        dataset_hparams: Dict[str, Any] = None,
        optimizer_hparams: Dict[str, Any] = None,
        lr_scheduler_hparams: Dict[str, Any] = None,
        trainer_hparams: Dict[str, Any] = None,
        data_hparams: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.optimizer_hparams = optimizer_hparams
        self.lr_scheduler_hparams = lr_scheduler_hparams

        # backbone model
        self.backbone, extra_layers_dict = self.init_backbone(
            backbone_hparams, dataset_hparams
        )
        if extra_layers_dict is not None:
            self.extra_layers_dict = nn.ModuleDict(extra_layers_dict)

        # tasks
        self.tasks = self.init_tasks(tasks)

        # losses
        self.loss_fns = {name: task.init_loss() for name, task in self.tasks.items()}

        # metrics
        # dict of dict: {mode: {task_name: metric_object}}
        self.metrics = nn.ModuleDict()
        for mode in ["train", "val", "test"]:
            # cannot use `train` directly (already a submodule of the class)
            mode = "metric_" + mode
            self.metrics[mode] = nn.ModuleDict()
            for name, task in self.tasks.items():
                mc = task.init_metric_as_collection()
                self.metrics[mode][name] = mc

        # timer
        self.timer = TimeMeter()

        # callback monitor key. Should set argument `monitor` of the ModelCheckpoint to
        # this. Same for other callbacks
        self.monitor_key = "val/score"

    def init_backbone(
        self,
        backbone_hparams: Dict[str, Any],
        dataset_hparams: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Create a backbone torch model.

        A pytorch or lightning model that can be called like:
        `model(graphs, *args, **kwargs)`
        The model should return a dictionary of {task_name: task_prediction}, where the
        task_name should the name of one task defined in `init_tasks()` and
        task_prediction should be a tensor.

        This will be called in the `decode()` function.
        Oftentimes, the underlying model may not return a dictionary (e.g. when using
        existing models). In this case, the model prediction should be converted to a
        dictionary in the `decode()` function.
        """
        raise NotImplementedError

    def init_tasks(
        self, tasks: Union[Task, List[Task], Dict[str, Task]]
    ) -> Dict[str, Task]:
        """
        Convert tasks to a dict, keyed by task name and valued by task object.
        """

        if isinstance(tasks, dict):
            for name, t in tasks.items():
                assert (
                    name == t.name
                ), f"Task name not consistent; got {name} and {t.name}"
        elif isinstance(tasks, Task):
            tasks = {tasks.name: tasks}
        elif isinstance(tasks, list):
            tasks = {t.name: t for t in tasks}
        else:
            raise ValueError(f"Unsupported tasks type {type(tasks)}")

        return tasks

    def forward(
        self,
        batch,
        mode: Optional[str] = None,
        task_name: str = "elastic_tensor_full",
        **kwargs,
    ) -> Tuple[Dict, Dict]:
        """
        Forward pass step for prediction.

        Intended for prediction use and will not be called at training time. Instead,
        train_step/validation_step will be called at training time.

        Args:
            batch:
            mode: select what to return. See `Returns` below.
            kwargs: extra arguments needed by the model.

        The functionality here is largely the same as `self.shared_step()`.

        Returns:
            A tuple of (predictions, labels), each is a dictionary. The content of
            predictions depends on the value of mode:
                If None, returns the model predictions: backbone + decoder.
                If `backbone`, returns the backbone prediction.
        """

        # ========== preprocess batch ==========
        graphs, labels = self.preprocess_batch(batch)

        # ========== compute predictions ==========
        if mode is None or mode.lower() == "none":
            preds = self.decode(graphs, **kwargs)
            preds = self.transform_prediction(preds, task_name=task_name)
            labels = self.transform_target(labels, task_name=task_name)
        elif mode == "backbone":
            preds = self.backbone(graphs, **kwargs)
        else:
            supported = (None, "backbone")
            raise ValueError(f"Expect mode to be one of {supported}; got {mode}")

        return preds, labels

    def transform_prediction(self, prediction: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Transform the predictions before returning for use.

        This is different from `transform_{pred,target}_{loss,metric}` in task,
        which is used internally when training the model. This transform is supposed
        to be used in `self.forward()`, which provides the final prediction of the
        model.

        Args:
            prediction: predictions of the decoder

        Returns:
            Transformed predictions.
        """
        return prediction

    def transform_target(self, target: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Similar to `self.transform_prediction()`, but for target.
        """
        return target

    def preprocess_batch(self, batch) -> Tuple[Any, Dict[str, Tensor]]:
        """
        preprocess the batch data to get model input and labels.

        Args:
            batch: batched data

        Returns:
            A tuple of (model_input, labels), where labels should be a dict of tensors,
            i.e. {task_name: task_label}
        """
        raise NotImplementedError

    def decode(self, model_input, *args, **kwargs) -> Dict[str, Tensor]:
        """
        Compute prediction for each task using the backbone model.

        Args:
            model_input: input for the model to make predictions, e.g. batched graphs

        Returns:
            {task_name: task_prediction}
        """
        raise NotImplementedError

    def compute_loss(
        self,
        preds: Dict[str, Tensor],
        labels: Dict[str, Tensor],
        weight: Tensor = None,
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Compute the loss for each task.

        Args:
            preds: {task_name, prediction} prediction for each task
            labels: {task_name, label} labels for each task
            weight: weight factor to be multiplied by predictions and labels

        Returns:
            individual_loss: {task_name: loss} loss of individual task
            total_loss: total loss, weighted sum of individual loss
        """
        individual_losses = {}
        total_loss = 0.0

        for task_name, task in self.tasks.items():
            p = preds[task_name]
            l = labels[task_name]
            p = task.transform_pred_loss(p)
            l = task.transform_target_loss(l)
            if weight is not None:
                p = p * weight
                l = l * weight

            if task.task_type == TaskType.CLASSIFICATION and task.is_binary():
                # this will use BCEWithLogitsLoss, which requires label be of float
                p = p.reshape(-1)
                l = l.reshape(-1).to(torch.get_default_dtype())

            loss_fn = self.loss_fns[task_name]
            loss = loss_fn(p, l)
            individual_losses[task_name] = loss
            total_loss = total_loss + task.loss_weight * loss

        return individual_losses, total_loss

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, "train")
        self.update_metrics(preds, labels, "train")

        return {"loss": loss}

    def on_training_epoch_end(self):
        self.compute_metrics("train")

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, "val")
        self.update_metrics(preds, labels, "val")

        return {"loss": loss}

    def on_validation_epoch_end(self):
        _, score = self.compute_metrics("val")

        # val/score used for early stopping and learning rate scheduler
        if score is not None:
            self.log(
                self.monitor_key, score, on_step=False, on_epoch=True, prog_bar=True
            )

        # time it
        delta_t, cumulative_t = self.timer.update()
        self.log("epoch time", delta_t, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "cumulative time", cumulative_t, on_step=False, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, "test")
        self.update_metrics(preds, labels, "test")

        return {"loss": loss}

    def on_test_epoch_end(self):
        self.compute_metrics("test")

    def shared_step(self, batch, mode: str):
        """
        Shared computation step.

        Args:
            batch: data batch, obtained from dataloader
            mode: train, val, or test
        """

        # `batch_size` needed to ensure the values logged via self.log are correctly
        # averaged across batch, because batch is PyG graph, and thus lightning
        # cannot automatically detect the batch size.
        # NOTE, using batch_size or not does not affect the results (i.e.
        # loss, metrics, and monitor_key) since we compute them directly. It only
        # affects the logged value via self.log (which is averaged).
        # TODO 1. move this to the below ModelForPyGData; 2. this only works when
        #  each graph has 1 target, if multiple need update.
        batch_size = batch.num_graphs

        # ========== preprocess batch ==========
        graphs, labels = self.preprocess_batch(batch)

        # ========== compute predictions ==========
        preds = self.decode(graphs)

        # ========== compute losses ==========
        target_weight = graphs.get("target_weight", None)
        individual_loss, total_loss = self.compute_loss(
            preds, labels, weight=target_weight
        )

        self.log_dict(
            {
                f"{mode}/loss/{task_name}": loss
                for task_name, loss in individual_loss.items()
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )

        self.log(
            f"{mode}/total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return total_loss, preds, labels

    def update_metrics(self, preds: Dict, labels: Dict, mode: str):
        """
        Update metric values at each step, i.e. keep record of values of each step that
        will be used to compute the epoch metrics.

        Args:
            mode: train, val, or test
        """
        mode = "metric_" + mode

        for task_name, metric in self.metrics[mode].items():
            task = self.tasks[task_name]

            p = task.transform_pred_metric(preds[task_name])
            l = task.transform_target_metric(labels[task_name])

            if task.task_type == TaskType.CLASSIFICATION:
                if task.is_binary():
                    p = torch.sigmoid(p.reshape(-1))
                else:
                    p = torch.argmax(p, dim=1)

            metric(p, l)

    def compute_metrics(
        self, mode, log: bool = True
    ) -> Tuple[Dict[str, Tensor], Union[Tensor, None]]:
        """
        Compute metric and logger it at each epoch.

        Args:
            mode: `train`, `val`, or `test`
            log: whether to log the metrics

        Returns:
            individual_score: individual metric scores, {task_name: scores},
                where scores is a dict.
            score: aggregated score. `None` if metric_aggregation() of task is not set.
        """

        mode = "metric_" + mode

        total_score = None
        individual_score = {}

        for task_name, metric_coll in self.metrics[mode].items():
            # metric collection output, a dict: {metric_name: metric_value}
            score = metric_coll.compute()
            individual_score[task_name] = score

            if log:
                for metric_name, metric_value in score.items():
                    self.log(
                        f"{mode}/{metric_name}/{task_name}",
                        metric_value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )

            # compute score for model checkpoint and early stopping
            task = self.tasks[task_name]
            metric_agg_dict = task.metric_aggregation()
            if metric_agg_dict:
                total_score = 0 if total_score is None else total_score
                for metric_name, weight in metric_agg_dict.items():
                    total_score = total_score + score[metric_name] * weight

            # reset to initial state for next epoch
            metric_coll.reset()

        return individual_score, total_score

    def configure_optimizers(self):
        # optimizer
        model_params = (filter(lambda p: p.requires_grad, self.parameters()),)
        optimizer = instantiate_class(model_params, self.optimizer_hparams)

        # lr scheduler
        scheduler = self._config_lr_scheduler(optimizer)

        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.monitor_key,
            }

    def _config_lr_scheduler(self, optimizer):
        """
        Configure lr scheduler.

        This allows not to use a lr scheduler. To achieve this, set `class_path` under
        `lr_scheduler` to `none` or `null`.

        Return:
            lr scheduler or None
        """
        class_path = self.lr_scheduler_hparams.get("class_path")
        if class_path is None or class_path == "none":
            scheduler = None
        else:
            scheduler = instantiate_class(optimizer, self.lr_scheduler_hparams)

        return scheduler


class ModelForPyGData(BaseModel):
    """
    A lightning model working with data provided as PyG batched data.

    Subclass must implement:
        - init_backbone(): create the underlying torch model
        - init_tasks(): create tasks that defines initialize the loss function and metrics

    """

    def preprocess_batch(self, batch: DataPoint) -> Tuple[DataPoint, Dict[str, Tensor]]:
        """
        Preprocess the batch data to get model input and labels.

        Note, this requires all the labels be stored as a dict in `y` of PyG data.

        Args:
            batch: PyG batched data

        Returns:
            (model_input, labels), where model_input is batched graph and labels
                is a dict of tensors, i.e. {task_name: task_label}
        """

        graphs = batch
        graphs = graphs.to(self.device)  # lightning cannot move graphs to gpu

        # task labels
        labels = {name: graphs.y[name] for name in self.tasks}

        # convert graphs to a dict to use NequIP stuff
        graphs = graphs.tensor_property_to_dict()

        return graphs, labels

    def decode(self, model_input: DataPoint, *args, **kwargs) -> Dict[str, Tensor]:
        """
        Compute prediction for each task using the backbone model.

        Args:
            model_input: (batched) PyG graph

        Returns:
            {task_name: task_prediction}
        """

        preds = self.backbone(model_input)

        return preds
