"""
These are utilities copied from nequip
"""
import math
from typing import Union

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from matten.data.irreps import DataKey, ModuleIrreps


# source: nequip.nn.nonlinearities
class ShiftedSoftPlus(torch.nn.Module):
    """
    Shifted softplus as defined in SchNet, NeurIPS 2017.

    :param beta: value for the a more general softplus, default = 1
    :param threshold: values above are linear function, default = 20
    """

    _log2: float

    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.softplus = torch.nn.Softplus(beta=beta, threshold=threshold)
        self._log2 = math.log(2.0)

    def forward(self, x):
        """
        Evaluate shifted softplus

        :param x: torch.Tensor, input
        :return: torch.Tensor, ssp(x)
        """
        return self.softplus(x) - self._log2


import torch
from torch import nn


# source nequip.nn.cutoffs
class PolynomialCutoff(nn.Module):
    def __init__(self, r_max, p=6):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        p : int
            Power used in envelope function
        """
        super(PolynomialCutoff, self).__init__()

        self.register_buffer("p", torch.Tensor([p]))
        self.register_buffer("r_max", torch.Tensor([r_max]))

    def forward(self, x):
        """
        Evaluate cutoff function.

        x: torch.Tensor, input distance
        """
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0)
            * torch.pow(x / self.r_max, self.p)
            + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1.0)
            - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2.0)
        )

        envelope *= (x < self.r_max).float()
        return envelope


# source: eequip.nn.radial_basis
class BesselBasis(nn.Module):
    r_max: float
    prefactor: float

    def __init__(self, r_max, num_basis=8, trainable=True):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x):
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)

        return self.prefactor * (numerator / x.unsqueeze(-1))


# source: nequip.nn.embedding._edge
@compile_mode("script")
class SphericalHarmonicEdgeAttrs(ModuleIrreps, torch.nn.Module):
    """Construct edge attrs as spherical harmonic projections of edge vectors.

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        irreps_edge_sh: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        out_field: str = DataKey.EDGE_ATTRS,
    ):
        super().__init__()
        self.out_field = out_field

        if isinstance(irreps_edge_sh, int):
            self.irreps_edge_sh = o3.Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)

        self.init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
        )
        self.sh = o3.SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        data = with_edge_vectors(data, with_lengths=False)
        edge_vec = data[DataKey.EDGE_VECTORS]
        edge_sh = self.sh(edge_vec)
        data[self.out_field] = edge_sh
        return data


# source: nequip.nn.embedding._edge
@compile_mode("script")
class RadialBasisEdgeEncoding(ModuleIrreps, torch.nn.Module):
    out_field: str

    def __init__(
        self,
        basis=BesselBasis,
        cutoff=PolynomialCutoff,
        basis_kwargs={},
        cutoff_kwargs={},
        out_field: str = DataKey.EDGE_EMBEDDING,
        irreps_in=None,
    ):
        super().__init__()
        self.basis = basis(**basis_kwargs)
        self.cutoff = cutoff(**cutoff_kwargs)
        self.out_field = out_field

        self.init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps([(self.basis.num_basis, (0, 1))])},
        )

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        data = with_edge_vectors(data, with_lengths=True)
        edge_length = data[DataKey.EDGE_LENGTH]
        edge_length_embedded = (
            self.basis(edge_length) * self.cutoff(edge_length)[:, None]
        )
        data[self.out_field] = edge_length_embedded
        return data


# source nequip.data.AtomicDataDict.py
@torch.jit.script
def with_edge_vectors(data: DataKey.Type, with_lengths: bool = True) -> DataKey.Type:
    """Compute the edge displacement vectors for a graph.

    If ``data.pos.requires_grad`` and/or ``data.cell.requires_grad``, this
    method will return edge vectors correctly connected in the autograd graph.

    Returns:
        Tensor [n_edges, 3] edge displacement vectors
    """
    if DataKey.EDGE_VECTORS in data:
        if with_lengths and DataKey.EDGE_LENGTH not in data:
            data[DataKey.EDGE_LENGTH] = torch.linalg.norm(
                data[DataKey.EDGE_VECTORS], dim=-1
            )
        return data
    else:
        # Build it dynamically
        # Note that this is
        # (1) backwardable, because everything (pos, cell, shifts)
        #     is Tensors.
        # (2) works on a Batch constructed from AtomicData
        pos = data[DataKey.POSITIONS]
        edge_index = data[DataKey.EDGE_INDEX]
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        if DataKey.CELL in data:
            # ^ note that to save time we don't check that the edge_cell_shifts are trivial if no cell is provided; we just assume they are either not present or all zero.
            # -1 gives a batch dim no matter what
            cell = data[DataKey.CELL].view(-1, 3, 3)
            edge_cell_shift = data[DataKey.EDGE_CELL_SHIFT]
            if cell.shape[0] > 1:
                batch = data[DataKey.BATCH]
                # Cell has a batch dimension
                # note the ASE cell vectors as rows convention
                edge_vec = edge_vec + torch.einsum(
                    "ni,nij->nj", edge_cell_shift, cell[batch[edge_index[0]]]
                )
                # TODO: is there a more efficient way to do the above without
                # creating an [n_edge] and [n_edge, 3, 3] tensor?
            else:
                # Cell has either no batch dimension, or a useless one,
                # so we can avoid creating the large intermediate cell tensor.
                # Note that we do NOT check that the batch array, if it is present,
                # is trivial — but this does need to be consistent.
                edge_vec = edge_vec + torch.einsum(
                    "ni,ij->nj",
                    edge_cell_shift,
                    cell.squeeze(0),  # remove batch dimension
                )
        data[DataKey.EDGE_VECTORS] = edge_vec
        if with_lengths:
            data[DataKey.EDGE_LENGTH] = torch.linalg.norm(edge_vec, dim=-1)
        return data


# source nequip.data.AtomicDataDict.py
@torch.jit.script
def with_batch(data: DataKey.Type) -> DataKey.Type:
    """Get batch Tensor.

    If this AtomicDataPrimitive has no ``batch``, one of all zeros will be
    allocated and returned.
    """
    if DataKey.BATCH in data:
        return data
    else:
        pos = data[DataKey.POSITIONS]
        batch = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
        data[DataKey.BATCH] = batch
        return data
