from typing import Dict, List, Tuple

import torch
from e3nn.math import soft_one_hot_linspace
from e3nn.o3 import Irreps
from torch_scatter import scatter

from matten.data.irreps import DataKey, ModuleIrreps
from matten.nn._nequip import with_edge_vectors


class SpeciesEmbedding(ModuleIrreps, torch.nn.Module):
    """
    Embed atomic species (number) to  with fixed-size lookup table using
    torch.nn.Embedding.

    Args:
        irreps_in:
        embedding_dim: output dim of the species embedding
        num_species: number of uniques species for embedding. If this is provided,
            the `data` for forward should contain DataKey.SPECIES_INDEX, which are
            the index of the atom species (from 0 to num_species-1).
        allowed_species: allowed atomic number of the species. This serves the same
            purpose as `num_species`, and is exclusive with it. The difference is that
            allowed_species allows non-consecutive integers as input, and it will be
            mapped to consecutive species_index internally. If this is used,
            the `data` for forward should contain DAtaKey.ATOMIC_NUMBERS.
        out_fields: the generated embedding will be assigned to the output data dict
            with keys in out_fields
        use_atom_feats: whether to use atom features as species embedding in addition
            to embedding of atom number. If `True`, this requires data keyed by
            `atom_feats` in the data dict.
        atom_feats_dim: the dimension of atom features. If `use_atom_feats` is
            `True`, this needs to be provided.
    """

    def __init__(
        self,
        irreps_in: Dict[str, Irreps] = None,
        embedding_dim: int = 16,
        num_species: int = None,
        allowed_species: List[int] = None,
        out_fields: Tuple[str] = (DataKey.NODE_ATTRS, DataKey.NODE_FEATURES),
        use_atom_feats: bool = False,
        atom_feats_dim: int = None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.out_fields = out_fields
        self.use_atom_feats = use_atom_feats

        if allowed_species is not None and num_species is not None:
            raise ValueError("allowed_species and num_species cannot both be provided.")

        if allowed_species is not None:
            self.atomic_number_to_index = _AtomicNumberToIndex(allowed_species)
            self.num_species = self.atomic_number_to_index.num_species

        if self.use_atom_feats:
            if atom_feats_dim is None:
                raise ValueError(
                    "`atom_feats_dim` must be provided if `use_atom_feats` is True."
                )
            else:
                feats_dim = embedding_dim + atom_feats_dim
        else:
            feats_dim = embedding_dim

        attrs_dim = self.num_species

        # species as a scalar with even parity, with multiplicity out_dim
        irreps_out = {
            DataKey.NODE_ATTRS: Irreps(f"{attrs_dim}x0e"),
            DataKey.NODE_FEATURES: Irreps(f"{feats_dim}x0e"),
        }

        self.init_irreps(irreps_in, irreps_out)

        # learnable embedding layer
        # self.embedding = torch.nn.Embedding(num_species, embedding_dim)

        self.linear = torch.nn.Linear(attrs_dim, embedding_dim)

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        if DataKey.SPECIES_INDEX in data:
            type_numbers = data[DataKey.SPECIES_INDEX]
        elif DataKey.ATOMIC_NUMBERS in data:
            type_numbers = self.atomic_number_to_index(data[DataKey.ATOMIC_NUMBERS])
            data[DataKey.SPECIES_INDEX] = type_numbers
        else:
            raise ValueError(
                "Nothing in `data` to encode. Need either species_index or "
                "atomic_numbers"
            )

        attrs = torch.nn.functional.one_hot(
            type_numbers, num_classes=self.num_species
        ).float()

        embed = self.linear(attrs)

        # add atom feats (name hard coded)
        if self.use_atom_feats:
            embed = torch.hstack((embed, data["atom_feats"]))

        data[DataKey.NODE_ATTRS] = attrs
        data[DataKey.NODE_FEATURES] = embed

        return data


class NodeAttrsFromEdgeAttrs(ModuleIrreps, torch.nn.Module):
    REQUIRED_KEYS_IRREPS_IN = [DataKey.EDGE_ATTRS, DataKey.EDGE_INDEX]

    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        field: str = DataKey.EDGE_ATTRS,
        out_field: str = DataKey.NODE_ATTRS,
        reduce: str = "mean",
    ):
        """
        Compute node attributes from edge attributes, e.g. mean of the edge attributes.

        This modifies data[out_filed] from data[field].

        Args:
            irreps_in: input irreps
            field: field from which to obtain the data
            out_field: field to which to write the data
            reduce: reduction method, `mean` or `sum`
        """
        super().__init__()

        self.init_irreps(irreps_in, irreps_out={out_field: irreps_in[field]})

        self.field = field
        self.out_field = out_field
        self.reduce = reduce

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        edge_src, edge_dst = data[DataKey.EDGE_INDEX]

        x = scatter(
            data[self.field],
            edge_dst,
            dim=0,
            dim_size=len(data[DataKey.NODE_ATTRS]),
            reduce=self.reduce,
        )

        data[self.out_field] = x

        return data


class EdgeLengthEmbedding(ModuleIrreps, torch.nn.Module):
    REQUIRED_KEYS_IRREPS_IN = [DataKey.POSITIONS, DataKey.EDGE_INDEX]

    def __init__(
        self,
        irreps_in: Dict[str, Irreps] = None,
        out_field: str = DataKey.EDGE_EMBEDDING,
        num_basis: int = 10,
        start: float = 0.0,
        end: float = 5.0,
        basis: str = "bessel",
        cutoff: bool = True,
    ):
        """
        Embed edge length using basis functions.
        """
        super().__init__()

        self.num_basis = num_basis
        self.start = start
        self.end = end
        self.basis = basis
        self.cutoff = cutoff

        irreps_out = Irreps(f"{num_basis}x0e")
        self.init_irreps(irreps_in, irreps_out={out_field: irreps_out})

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        data = with_edge_vectors(data, with_lengths=True)
        edge_length = data[DataKey.EDGE_LENGTH]

        length_embedding = soft_one_hot_linspace(
            edge_length,
            start=self.start,
            end=self.end,
            number=self.num_basis,
            basis=self.basis,
            cutoff=self.cutoff,
        )
        # normalize it to ensure second moment close to 1, see
        # https://docs.e3nn.org/en/stable/guide/convolution.html
        length_embedding = length_embedding.mul(self.num_basis**0.5)

        data[DataKey.EDGE_EMBEDDING] = length_embedding

        return data


class _AtomicNumberToIndex(torch.nn.Module):
    """
    Map non-consecutive atomic numbers to consecutive atomic index.

    For example, suppose we have C and O with atomic numbers 6 and 8, we can map them
    to atomic index 0, and 1.
    """

    def __init__(self, allowed_atomic_numbers: List[int]):
        super().__init__()

        allowed = torch.as_tensor(sorted(allowed_atomic_numbers), dtype=torch.long)
        num_species = len(allowed)

        self.register_buffer("_min_Z", allowed.min())
        self.register_buffer("_max_Z", allowed.max())
        self.register_buffer("_num_species", torch.as_tensor(num_species))

        # initialize all map to -1
        Z_to_index = torch.full((1 + self._max_Z - self._min_Z,), -1, dtype=torch.long)

        Z_to_index[allowed - self._min_Z] = torch.arange(num_species).to(torch.long)
        self.register_buffer("_Z_to_index", Z_to_index)

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atomic_numbers: 1D tensor of atomic numbers

        Returns:
            1D tensor of atomic index
        """
        if atomic_numbers.min() < self._min_Z or atomic_numbers.max() > self._max_Z:
            raise RuntimeError(
                "Invalid atomic numbers. Expect atomic numbers to be in the range "
                f"[{self._min_Z}, {self._max_Z}], but got min {atomic_numbers.min()} "
                f"and max {atomic_numbers.max()}"
            )

        index = self._Z_to_index[atomic_numbers - self._min_Z]

        if index.min() < 0:
            supported = [
                Z + self._min_Z for Z, idx in enumerate(self._Z_to_index) if idx != -1
            ]
            for i, val in enumerate(index):
                if val == -1:
                    n = atomic_numbers[i]
                    raise RuntimeError(
                        f"Expect atomic numbers to be in {supported}, "
                        f"got invalid atomic numbers `{n}` for data point `{i}`."
                    )

        return index

    @property
    def num_species(self):
        return self._num_species
