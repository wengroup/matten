import inspect
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import torch
import yaml
from e3nn.io import CartesianTensor


def to_path(path: Union[str, Path]) -> Path:
    """
    Convert a str to pathlib.Path.
    """
    return Path(path).expanduser().resolve()


def create_directory(path: Union[str, Path], is_directory: bool = False):
    """
    Create the directory for a file.

    Args:
        path: path to the file
        is_directory: whether the file itself is a directory? If yes, will create it;
            if not, will create a directory that is the parent of the file.
    """
    p = to_path(path)

    if is_directory:
        dirname = p
    else:
        dirname = p.parent

    if not dirname.exists():
        os.makedirs(dirname)


def to_list(value: Any) -> Sequence:
    """
    Convert a non-list to a list.
    """
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def yaml_dump(obj, filename: Union[str, Path], sort_keys: bool = False):
    """
    Dump an object as yaml.
    """
    create_directory(filename)
    with open(to_path(filename), "w") as f:
        yaml.dump(obj, f, default_flow_style=False, sort_keys=sort_keys)


def yaml_load(filename: Union[str, Path]):
    """
    Load an object from yaml.
    """
    with open(to_path(filename), "r") as f:
        obj = yaml.safe_load(f)

    return obj


def detect_nan_and_inf(
    x: torch.Tensor,
    file: Union[str, Path] = None,
    name: str = None,
    level: int = 1,
    filename: str = None,
):
    """
    Detect whether a tensor is nan or inf.

    Args:
        x: the tensor
        file: file where this function is called, can be `__file__`.
        name: name of the tensor.
        level: to show stack info of which level. 1 means where this function is
            called; 2 means the function that calls this function...
        filename: yaml filename to write the tensor
    """

    def get_line():
        # credit: https://stackoverflow.com/questions/6810999/how-to-determine-file-function-and-line-number
        #
        # 0 represents this line, 1 represents line at caller, 2 represents line at
        # caller of caller...
        frame_record = inspect.stack()[level + 1]  # +1 because we put this in get_line
        frame = frame_record[0]
        info = inspect.getframeinfo(frame)
        return info.lineno

    if torch.isnan(x).any():
        if filename:
            x = x.detach().cpu().numpy().tolist()
            yaml_dump(x, filename)
        raise ValueError(f"Tensor is nan at line {get_line()} of {file}, name={name}")

    elif torch.isinf(x).any():
        if filename:
            x = x.detach().cpu().numpy().tolist()
            yaml_dump(x, filename)
        raise ValueError(f"Tensor is inf at line {get_line()} of {file}, name={name}")


class CartesianTensorWrapper:
    """
    A wrapper of CartesianTensor that keeps a copy of reduced tensor product to
    avoid memory leak.
    """

    def __init__(self, formula):
        self.converter = CartesianTensor(formula=formula)
        self.rtp = self.converter.reduced_tensor_products()

    def from_cartesian(self, data):
        return self.converter.from_cartesian(data, self.rtp.to(data.device))

    def to_cartesian(self, data):
        return self.converter.to_cartesian(data, self.rtp.to(data.device))


class ToCartesian(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.ct = CartesianTensorWrapper(formula)

    def forward(self, data):
        return self.ct.to_cartesian(data)
