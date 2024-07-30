"""Script to train the materials tensor model."""

from pathlib import Path
from typing import Dict, List, Union

import yaml
from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.cli import instantiate_class as lit_instantiate_class

from matten.dataset.structure_scalar_tensor import TensorDataModule
from matten.log import set_logger
from matten.model_factory.task import TensorRegressionTask
from matten.model_factory.tfn_atomic_tensor import AtomicTensorModel


def instantiate_class(d: Union[Dict, List]):
    args = tuple()  # no positional args
    if isinstance(d, dict):
        return lit_instantiate_class(args, d)
    elif isinstance(d, list):
        return [lit_instantiate_class(args, x) for x in d]
    else:
        raise ValueError(f"Cannot instantiate class from {d}")


def get_args(path: Path):
    """Get the arguments from the config file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main(config: Dict):
    dm = TensorDataModule(**config["data"])
    dm.prepare_data()
    dm.setup()

    model = AtomicTensorModel(
        tasks=TensorRegressionTask(name=config["data"]["tensor_target_name"]),
        backbone_hparams=config["model"],
        dataset_hparams=dm.get_to_model_info(),
        optimizer_hparams=config["optimizer"],
        lr_scheduler_hparams=config["lr_scheduler"],
    )

    try:
        callbacks = instantiate_class(config["trainer"].pop("callbacks"))
        lit_logger = instantiate_class(config["trainer"].pop("logger"))
    except KeyError:
        callbacks = None
        lit_logger = None

    trainer = Trainer(
        callbacks=callbacks,
        logger=lit_logger,
        **config["trainer"],
    )

    logger.info("Start training!")
    trainer.fit(model, datamodule=dm)

    # test
    logger.info("Start testing!")
    trainer.test(ckpt_path="best", datamodule=dm)

    # print path of best checkpoint
    logger.info(f"Best checkpoint path: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    config_file = Path(__file__).parent / "configs" / "atomic_tensor.yaml"
    config = get_args(config_file)

    seed = config.get("seed_everything", 1)
    seed_everything(seed)

    log_level = config.get("log_level", "INFO")
    set_logger(log_level)

    main(config)
