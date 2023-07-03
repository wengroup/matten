from typing import Dict, Optional

import torch
from e3nn.io import CartesianTensor
from e3nn.o3 import Irreps

from matten.data.irreps import DataKey, ModuleIrreps


class IrrepsToCartesianTensor(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        formula: str = "ij=ji",
        field: str = DataKey.NODE_FEATURES,
        out_field: Optional[str] = None,
    ):
        """
        Convert irreps tensor to cartesian tensor.

        Args:
            irreps_in:
            formula: formula to indicate the symmetry of the cartesian tensor,
                e.g. `ij=ij` means a general 2D tensor. See the docs of
                `CartesianTensor` for more.
            field:
            out_field:
        """
        super().__init__()

        self.formula = formula
        self.field = field
        self.out_field = field if out_field is None else out_field

        # NOTE, should not add output to irreps_out, since it is a cartesian tensor,
        # no longer an irreps
        self.init_irreps(irreps_in, required_keys_irreps_in=[field])

        self.ct = CartesianTensor(formula=formula)
        self.ct_rtp = self.ct.reduced_tensor_products()

        assert self.irreps_in[self.field] == self.ct, (
            f"input irreps of {self.field} is {self.irreps_in[self.field]}, not equal "
            f"to the irreps of the target irreps {self.ct}"
        )

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        value = data[self.field]
        cartesian_tensor = self.ct.to_cartesian(value, self.ct_rtp.to(value.device))
        data[self.out_field] = cartesian_tensor

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  field: {self.field}, out_field: {self.out_field}, formula: {self.formula}\n"
            ")"
        )
