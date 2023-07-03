import torch

from matten.model.task import CanonicalRegressionTask


def test_regression_task():
    name = "some_test"
    loss_weight = 2.0

    task = CanonicalRegressionTask(name=name, loss_weight=loss_weight)

    # property
    assert task.name == name
    assert task.loss_weight == loss_weight

    # metrics
    metric = task.init_metric_as_collection()
    preds = torch.FloatTensor([0, 0, 1, 2])
    labels = torch.FloatTensor([1, 0, 1, 2])
    metric(preds, labels)
    out = metric.compute()
    assert out["MeanAbsoluteError"] == 0.25

    # transform
    preds_transform = task.transform_pred_loss(preds)
    labels_transform = task.transform_target_metric(labels)

    assert torch.allclose(preds_transform, preds)
    assert torch.allclose(labels_transform, labels)
