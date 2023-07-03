from typing import Tuple, Union

import numpy.typing as npt

PBC = Tuple[bool, bool, bool]

Vector = Union[Tuple[float, float, float], npt.ArrayLike]
IntVector = Union[Tuple[int, int, int], npt.ArrayLike]

Lattice = Union[Tuple[Vector, Vector, Vector], npt.ArrayLike]
