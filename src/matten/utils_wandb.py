import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from loguru import logger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from matten.utils import to_path, yaml_dump


def get_git_repo_commit(repo_path: Path) -> str:
    """
    Get the latest git commit info of a github repository.

    Args:
        repo_path: path to the repo

    Returns:
        latest commit info
    """
    output = subprocess.check_output(["git", "log"], cwd=to_path(repo_path))
    output = output.decode("utf-8").split("\n")[:6]
    latest_commit = "\n".join(output)

    return latest_commit


def get_hostname() -> str:
    """
    Get the hostname of the machine.

    Returns:
        hostname
    """
    output = subprocess.check_output("hostname")
    hostname = output.decode("utf-8").strip()

    return hostname


def write_running_metadata(
    git_repo: Optional[Path] = None, filename: str = "running_metadata.yaml"
):
    """
    Write additional running metadata to a file and then copy it to wandb.

    Currently, we write:
    - the running dir, i.e. cwd
    - git repo commit, optional

    Args:
        git_repo: path to the git repo, if None, will try to find it automatically.
        filename: name of the file to write
    """
    if git_repo is None:
        import matten

        git_repo = Path(matten.__file__).parents[1]

    d = {
        "running_dir": Path.cwd().as_posix(),
        "hostname": get_hostname(),
        "git_commit": get_git_repo_commit(git_repo),
    }

    yaml_dump(d, filename)


@rank_zero_only
def save_files_to_wandb(wandb, paths: List[Union[str, Path]]):
    """
    Save files to wandb.

    Args:
        wandb: a wandb run that has been initialized. For example, if using lightning
            wandb logger, this could be: trainer.logger.experiment
        paths: path of the files.
    """
    directory = wandb.dir  # something like: /wandb/run-20210921_113428-3t119xym/files
    for p in paths:
        p = to_path(p)
        if p.exists():
            # make a copy to avoid creating a symlink to the original file
            target = to_path(directory).joinpath(p.name)
            shutil.copy(p, target)

            # If we do not call the below line, wandb will save the files
            # automatically since we moved them to its `files` directory.
            # Here, we explicitly call it in case something wrong happens and wandb
            # did not finish successfully.
            wandb.save(target.as_posix(), policy="now")
        else:
            logger.warning(f"File `{str(p)}` does not exist. Won't save it to wandb.")


def get_wandb_run_path(identifier: str, path: Union[str, Path] = "."):
    """
    Find the wandb run path given its experiment identifier.

    Args:
        identifier: wandb unique identifier of experiment, e.g. 2i3rocdl
        path: root path to search

    Returns:
        path to the wandb run directory:
        e.g. running_dir/job_0/wandb/wandb/run-20201210_160100-3kypdqsw
    """
    for root, dirs, files in os.walk(path):
        if "wandb" not in root:
            continue
        for d in dirs:
            if d.startswith("run-") or d.startswith("offline-run-"):
                if d.split("-")[-1] == identifier:
                    return os.path.abspath(os.path.join(root, d))

    raise RuntimeError(f"Cannot found job {identifier} in {path}")


def get_wandb_checkpoint_path(identifier: str, path: Union[str, Path] = "."):
    """
    Get path to the checkpoint directory.

    Args:
        identifier: wandb unique identifier of experiment, e.g. 2i3rocdl
        path: root path to search
    Returns:
        path to the wandb checkpoint directory:
        e.g. running_dir/job_0/wandb/<project_name>/<identifier>/checkpoints
    """
    for root, dirs, files in os.walk(path):
        if root.endswith(f"{identifier}/checkpoints"):
            return os.path.abspath(root)

    return None


def get_wandb_logger(loggers):
    """
    Given a logger instance or a sequence of logger instances, return the wandb logger
    instance.

    Return  `None` if no wandb logger is used.
    """
    if isinstance(loggers, Sequence):
        for lg in loggers:
            if isinstance(lg, WandbLogger):
                return lg
    else:
        if isinstance(loggers, WandbLogger):
            return loggers

    return None


def get_wandb_identifier(save_dir: Union[str, Path], run_directory: str = "latest-run"):
    """
    Get the identifier of a wandb run.

    Args:
        save_dir: name of the directory to save wandb log, e.g. /path/to/wandb_log/
        run_directory: the directory for the run that stores files, logs, and run info,

    Returns:
        identifier: identifier of the wandb run
    """

    save_dir = to_path(save_dir)
    run_dir = save_dir.joinpath("wandb", run_directory).resolve()
    if run_dir.exists():
        identifier = str(run_dir).split("-")[-1]
        return identifier
    else:
        return None


def get_wandb_checkpoint_and_identifier_latest(
    save_dir: Union[str, Path], run_directory: str = "latest-run"
) -> Tuple[Union[str, None], Union[str, None]]:
    """
    Get the latest checkpoint path and the identifier of the wandb logger from wandb logs.

    Args:
        save_dir: name of the directory to save wandb log, e.g. /path/to/wandb_log/
        run_directory: the directory for the run that stores files, logs, and run info,
            e.g. run-20210203_142512-6eooscnj
    Returns:
        ckpt_path: path to the latest run
        identifier: identifier of the wandb run
    """
    identifier = get_wandb_identifier(save_dir, run_directory)

    if identifier:
        # checkpoint path of latest_run
        ckpt_dir = get_wandb_checkpoint_path(identifier, save_dir)
        if ckpt_dir is not None:
            ckpt_path = str(to_path(ckpt_dir).joinpath("last.ckpt").resolve())
        else:
            ckpt_path = None
            identifier = None
    else:
        ckpt_path = None
        identifier = None

    return ckpt_path, identifier
