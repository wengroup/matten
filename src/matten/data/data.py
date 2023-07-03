import warnings
from typing import Dict, List, Optional, Union

import ase.geometry
import ase.neighborlist
import numpy as np
import numpy.typing as npt
import torch
from pymatgen.core import Structure
from torch import Tensor
from torch_geometric.data import Data

from matten._typing import PBC, IntVector, Vector
from matten.data._dtype import DTYPE, DTYPE_BOOL, DTYPE_INT, TORCH_FLOATS, TORCH_INTS


class DataPoint(Data):
    """
    Base class of a graph data point (be it a molecule or a crystal).

    Args:
        pos: coords of the atoms
        edge_index: 2D array (2, num_edges). edge index.
        x: input to the model, i.e. initial node features
        y: reference value for the output of the model, e.g. DFT energy, forces
        edge_cell_shift: 2D array (num_edges, 3). which periodic image of the target
            point each edge goes to, relative to the source point. Used when periodic
            boundary conditions is effective (e.g. for crystals). Optional.
        cell: 3D array (1, 3, 3). The periodic cell for ``edge_cell_shift`` as the three
            triclinic cell vectors. Necessary only when ``edge_cell_shift`` is not
            None. Optional.
        num_neigh: 1D array (num_atoms). number of neighbors for each atom.
        kwargs: extra property of a data point, e.g. prediction output such as energy
            and forces.

    Notes:
        To use PyG ``InMemoryDataset`` and ``Batch``, all arguments of `Data` should
        be keyword arguments. This is because in collate()
        (see https://github.com/pyg-team/pytorch_geometric/blob/74245f3a680c1f6fd1944623e47d9e677b43e827/torch_geometric/data/collate.py#L14)
         something like `out = cls()` (cls is the subclassed Data class) is used.
        We see that a `Data` class is instantiated without passing arguments.
        For this class, we use ``*`` to distinguish them in the initializer. Arguments
        before ``*`` are necessary, after it are optional.
    """

    def __init__(
        self,
        pos: List[Vector] = None,
        edge_index: npt.ArrayLike = None,
        x: Dict[str, npt.ArrayLike] = None,
        y: Dict[str, npt.ArrayLike] = None,
        *,
        edge_cell_shift: Optional[List[IntVector]] = None,
        cell: Optional[npt.ArrayLike] = None,
        num_neigh: Optional[npt.ArrayLike] = None,
        **kwargs,
    ):
        # convert to tensors
        if pos is not None:
            pos = torch.as_tensor(pos, dtype=DTYPE)
            assert pos.shape[1] == 3
            num_nodes = pos.shape[0]
        else:
            num_nodes = None

        if edge_index is not None:
            edge_index = torch.as_tensor(edge_index, dtype=DTYPE_INT)
            assert edge_index.shape[0] == 2
            num_edges = edge_index.shape[1]
        else:
            num_edges = None

        if edge_cell_shift is not None:
            # the dtype can be int, but we use float here because it's going to
            # be multiplied with `cell`, which is float
            edge_cell_shift = torch.as_tensor(edge_cell_shift, dtype=DTYPE)
            assert edge_cell_shift.shape == (num_edges, 3)
            assert cell is not None, (
                "both `edge_cell_shift` and `cell` should be " "provided"
            )

        if cell is not None:
            cell = torch.as_tensor(cell, dtype=DTYPE)
            assert cell.shape == (3, 3)
            assert edge_cell_shift is not None, (
                "both `edge_cell_shift` and `cell` should " "be provided"
            )

        if num_neigh is not None:
            num_neigh = torch.as_tensor(num_neigh, dtype=DTYPE)
            assert len(num_neigh) == len(pos)

        # convert input and output to tensors
        if x is not None:
            tensor_x = {}
            for k, v in x.items():
                v_out = self._convert_to_tensor(v)
                if v_out is None:
                    raise ValueError(
                        f"Only accepts np.ndarray or torch.Tensor. `{k}` of x is of "
                        f"type `{type(v)}`."
                    )
                tensor_x[k] = v_out
            self._check_tensor_dict(tensor_x, dict_name="x")
        else:
            tensor_x = None

        if y is not None:
            tensor_y = {}
            for k, v in y.items():
                v_out = self._convert_to_tensor(v)
                if v_out is None:
                    raise ValueError(
                        f"Only accepts np.ndarray or torch.Tensor. `{k}` of y is of "
                        f"type `{type(v)}`."
                    )
                tensor_y[k] = v_out
            self._check_tensor_dict(tensor_y, dict_name="y")
        else:
            tensor_y = None

        # convert kwargs to tensor
        tensor_kwargs = {}
        for k, v in kwargs.items():
            v_out = self._convert_to_tensor(v)
            if v_out is None:
                raise ValueError(
                    f"Only accepts np.ndarray or torch.Tensor. kwarg `{k}` is of type "
                    f" `{type(v)}`."
                )
            tensor_kwargs[k] = v_out
        self._check_tensor_dict(tensor_kwargs, dict_name="kwargs")

        super().__init__(
            pos=pos,
            edge_index=edge_index,
            edge_cell_shift=edge_cell_shift,
            cell=cell,
            num_nodes=num_nodes,
            x=tensor_x,
            y=tensor_y,
            num_neigh=num_neigh,
            **tensor_kwargs,
        )

    def tensor_property_to_dict(self):
        """
        Convert all tensor properties to a dict.
        """
        d = self.to_dict()

        out = {}
        for k, v in d.items():
            if isinstance(v, Tensor):
                out[k] = v
            elif isinstance(v, dict):
                out.update(v)

        return out

    @staticmethod
    def _check_tensor_dict(d: Dict[str, Tensor], dict_name: str = "name_unknown"):
        """
        Check the values of a dict are at least 1D tensors.

        Args:
            d: the dict to check
            dict_name: name of the dictionary
        """

        for k, v in d.items():
            assert isinstance(
                v, Tensor
            ), f"Expect `{k}` in dict `{dict_name}` to be a tensor, got `{type(v)}`"

            assert len(v.shape) >= 1, (
                f"Expect `{k}` in dict `{dict_name}` to be a tensor at least 1D, "
                f"but its shape is `{v.shape}`."
            )

    @staticmethod
    def _convert_to_tensor(x):
        """
        Convert a np.ndarray or a torch.tensor to tensor.

        Return None, if cannot deal with it.
        """
        if isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, np.floating):
                return torch.as_tensor(x, dtype=DTYPE)
            elif np.issubdtype(x.dtype, np.integer):
                return torch.as_tensor(x, dtype=DTYPE_INT)
            elif x.dtype == bool:
                return torch.as_tensor(x, dtype=DTYPE_BOOL)
            else:
                return None

        elif isinstance(x, Tensor):
            if x.dtype in TORCH_FLOATS:
                return torch.as_tensor(x, dtype=DTYPE)
            elif x.dtype in TORCH_INTS:
                return torch.as_tensor(x, dtype=DTYPE_INT)
            elif x.dtype == torch.bool:
                return torch.as_tensor(x, dtype=DTYPE_BOOL)
            else:
                return None

        else:
            return None


class Crystal(DataPoint):
    """
    Crystal graph data point, with the notation of supercell.
    """

    @classmethod
    def from_points(
        cls,
        pos: List[Vector],
        cell: npt.ArrayLike,
        pbc: PBC,
        r_cut: float,
        x: Dict[str, npt.ArrayLike],
        y: Dict[str, npt.ArrayLike],
        **kwargs,
    ):
        """
        Create a crystal from a set of points.

        Args:
            pos:
            cell:
            pbc:
            r_cut: neighbor cutoff distance
            x:
            y:
            **kwargs:
        """

        # deal with periodic boundary conditions
        edge_index, edge_cell_shift, cell, num_neigh = neighbor_list_and_relative_vec(
            pos=pos,
            r_max=r_cut,
            self_interaction=False,
            strict_self_interaction=True,
            cell=cell,
            pbc=pbc,
        )

        return cls(
            pos=pos,
            edge_index=edge_index,
            x=x,
            y=y,
            edge_cell_shift=edge_cell_shift,
            cell=cell,
            num_neigh=num_neigh,
            **kwargs,
        )

    @classmethod
    def from_pymatgen(
        cls,
        struct: Structure,
        r_cut: float,
        x: Dict[str, npt.ArrayLike],
        y: Dict[str, npt.ArrayLike],
        **kwargs,
    ):
        return cls.from_points(
            pos=struct.cart_coords,
            cell=struct.lattice.matrix,
            pbc=(True, True, True),
            r_cut=r_cut,
            x=x,
            y=y,
            **kwargs,
        )


# TODO, we can directly return the shift vector using `ijD`, instead of edge_cell_shift
#  to speed up the code a bit. At training, this is not a problem since the dataset
#  is in memory and the dataloader prefetches it.
def neighbor_list_and_relative_vec(
    pos: np.ndarray,
    r_max: float,
    self_interaction: bool = False,
    strict_self_interaction: bool = True,
    cell: np.ndarray = None,
    pbc: Union[bool, List[bool]] = False,
):
    """
    Create neighbor list (``edge_index``) and relative vectors (``edge_attr``) based on
    radial cutoff.

    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).
    Thus, ``edge_index`` has the same convention as the relative vectors:
    :math:`\\vec{r}_{source, target}`
    If the input positions are a tensor with ``requires_grad == True``,
    the output displacement vectors will be correctly attached to the inputs
    for autograd.
    All outputs are Tensors on the same device as ``pos``; this allows future
    optimization of the neighbor list on the GPU.

    Args:
        pos (shape [N, 3]): Positional coordinate; Tensor or numpy array. If Tensor,
            must be on CPU.
        r_max (float): Radial cutoff distance for neighbor finding.
        self_interaction (bool): Whether to include same periodic image self-edges in
            the neighbor list. Should be False for most applications.
        strict_self_interaction (bool): Whether to include *any* self interaction edges
            in the graph, even if the two instances of the atom are in different
            periodic images. Should be True for most applications.
        pbc (bool or 3-tuple of bool): Whether the system is periodic in each of the
            three cell dimensions.
        cell (shape [3, 3]): Cell for periodic boundary conditions.
            Ignored if ``pbc == False``.

    Returns:
        edge_index (torch.tensor shape [2, num_edges]): List of edges.
        edge_cell_shift (torch.tensor shape [num_edges, 3]): Relative cell shift
            vectors. Returned only if cell is not None.
        cell (torch.Tensor [3, 3]): the cell as a tensor on the correct device.
            Returned only if cell is not None.
        num_neigh (torch.Tensor [N]) number of neighbors for each atom.
    """
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3

    # Either the position or the cell may be on the GPU as tensors
    if isinstance(pos, torch.Tensor):
        temp_pos = pos.detach().cpu().numpy()
        out_device = pos.device
        out_dtype = pos.dtype
    else:
        temp_pos = np.asarray(pos)
        out_device = torch.device("cpu")
        out_dtype = torch.get_default_dtype()

    # Right now, GPU tensors require a round trip
    if out_device.type != "cpu":
        warnings.warn(
            "Currently, neighborlists require a round trip to the CPU. "
            "Please pass CPU tensors if possible."
        )

    # Get a cell on the CPU no matter what
    if isinstance(cell, torch.Tensor):
        temp_cell = cell.detach().cpu().numpy()
        cell_tensor = cell.to(device=out_device, dtype=out_dtype)
    elif cell is not None:
        temp_cell = np.array(cell)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)
    else:
        # ASE will "complete" this correctly.
        temp_cell = np.zeros((3, 3), dtype=temp_pos.dtype)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)

    # ASE dependent part
    temp_cell = ase.geometry.complete_cell(temp_cell)

    first_idex, second_idex, shifts = ase.neighborlist.primitive_neighbor_list(
        "ijS",
        pbc,
        temp_cell,
        temp_pos,
        cutoff=float(r_max),
        # we want edges from atom to itself in different periodic images!
        self_interaction=strict_self_interaction,
        use_scaled_positions=False,
    )

    # Eliminate true self-edges that don't cross periodic boundaries
    if not self_interaction:
        bad_edge = first_idex == second_idex
        bad_edge &= np.all(shifts == 0, axis=1)
        keep_edge = ~bad_edge
        if not np.any(keep_edge):
            raise ValueError(
                "After eliminating self edges, no edges remain in this system."
            )
        first_idex = first_idex[keep_edge]
        second_idex = second_idex[keep_edge]
        shifts = shifts[keep_edge]

    # Build output:
    edge_index = torch.vstack(
        (torch.LongTensor(first_idex), torch.LongTensor(second_idex))
    ).to(device=out_device)

    shifts = torch.as_tensor(
        shifts,
        dtype=out_dtype,
        device=out_device,
    )

    # Number of neighbors for each atoms
    num_neigh = torch.as_tensor(np.bincount(first_idex), device=out_device)

    # Some atoms with large atom index may not have neighbors due to the use of bincount
    # As a concrete example, suppose we have 5 atoms and first_idex is [0,1,1,3,3,3,3],
    # then bincount will be [1, 2, 0, 4], which means atoms 0,1,2,3 have 1,2,0,4
    # neighbors respectively. Although atom 2 is handled by bincount, atom 4 cannot.
    # The below part is to make this work.
    if len(num_neigh) != len(pos):
        tmp_num_neigh = torch.zeros(len(pos), dtype=num_neigh.dtype, device=out_device)
        tmp_num_neigh[list(range(len(num_neigh)))] = num_neigh
        num_neigh = tmp_num_neigh

    return edge_index, shifts, cell_tensor, num_neigh


class DataError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.msg = msg
