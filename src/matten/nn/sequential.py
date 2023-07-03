from collections import OrderedDict
from typing import overload

import torch

from matten.data.irreps import ModuleIrreps, _check_irreps_compatible


class Sequential(torch.nn.Sequential, ModuleIrreps):
    """
    This is the same as torch.nn.Sequential, with additional check on irreps
    compatibility between consecutive modules.
    """

    @overload
    def __init__(self, *args: ModuleIrreps) -> None:
        ...

    @overload
    def __init__(self, arg: "OrderedDict[str, ModuleIrreps]") -> None:
        ...

    def __init__(self, *args):

        # dict input
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            module_dict = args[0]
            module_list = list(module_dict.values())
        # sequence input
        else:
            module_list = list(args)
            module_dict = OrderedDict(
                (f"{m.__class__.__name__}_{i}", m) for i, m in enumerate(module_list)
            )

        # check in/out irreps compatibility
        for i, (m1, m2) in enumerate(zip(module_list, module_list[1:])):
            if not _check_irreps_compatible(m1.irreps_out, m2.irreps_in):
                raise ValueError(
                    f"Output irreps of module {i} `{m1.__class__.__name__}`: "
                    f"{m1.irreps_out}` is incompatible with input irreps of module {i+1} "
                    f"`{m2.__class__.__name__}`: {m2.irreps_in}."
                )

        self.init_irreps(
            irreps_in=module_list[0].irreps_in, irreps_out=module_list[-1].irreps_out
        )

        super().__init__(module_dict)
