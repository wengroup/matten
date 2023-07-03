"""
Operations on node features/attrs.

These are based on nequip.nn._atomwise.py
"""

from typing import Dict, Optional

import e3nn
import torch
from e3nn.o3 import Irreps
from torch_scatter import scatter

from matten.data.irreps import DataKey, ModuleIrreps
from matten.nn._nequip import with_batch


class NodewiseSelect(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        field: str = DataKey.NODE_FEATURES,
        out_field: Optional[str] = None,
        mask_field: Optional[str] = None,
    ):

        """
        Select the atom features (attrs) in a structure by boolean masks.

        For example, for a structure has 4 atoms, `data[mask_field] = [True,False,True,
        False]` will select the features (attrs) of atoms 0 and 2, ignoring atoms
        1 and 3.

        Args:
            irreps_in:
            field: the field from which to select the features/attrs
            out_field: the output field for the selected features/attrs. Defaults to
                field.
            mask_field: field of the masks. If `None`, all atomic sites will be
                selected, corresponding to set `data[mask_field]` to `[True, True,
                True, True]` in the above example.

        Note:
            This module does not necessarily need to be a subclass of ModuleIrreps,
            since no operations on irreps are conduced. However, we use it as a proxy
            to check the existence of the fields. The idea is that if a filed is in the
            irreps_in dict, it should be in the data dict for `forward`.
            # TODO we still need to enable the check of the mask_field. For this
            #  purpose, we can add a native module update the irreps_in dict at the
            #  beginning of a model and set its irreps to `None`. Note, we do this just
            #  for consistence purpose, it is not really needed. So, we may ignore it.
        """
        super().__init__()

        self.field = field
        self.out_field = out_field if out_field is not None else field
        self.mask_field = mask_field

        self.init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.field]},
            required_keys_irreps_in=[self.field],
        )

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        # shallow copy so input `data` is not modified
        data = data.copy()
        value = data[self.field]

        if self.mask_field is None:
            # simply copy it over
            selected = value
        else:
            masks = data[self.mask_field]
            selected = value[masks]

        data[self.out_field] = selected

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  field: {self.field}, out_field: {self.out_field}, out field irreps: "
            f"{self.irreps_out[self.out_field]}\n"
            ")"
        )


class NodewiseLinear(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        irreps_out: Irreps = None,
        field: str = DataKey.NODE_FEATURES,
        out_field: Optional[str] = None,
    ):
        super().__init__()

        self.field = field
        self.out_field = out_field if out_field is not None else field

        if irreps_out is None:
            irreps_out = irreps_in[self.field]

        self.init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_out},
            required_keys_irreps_in=[self.field],
        )

        self.linear = e3nn.o3.Linear(
            irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[self.out_field]
        )

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        data[self.out_field] = self.linear(data[self.field])
        return data


class NodewiseReduce(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        field: str,
        out_field: Optional[str] = None,
        reduce: str = "sum",
    ):
        super().__init__()

        assert reduce in ("sum", "mean", "min", "max")

        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field

        self.init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.field]},
            required_keys_irreps_in=[self.field],
        )

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        with_batch(data)
        data[self.out_field] = scatter(
            data[self.field], data[DataKey.BATCH], dim=0, reduce=self.reduce
        )

        return data
