from collections import OrderedDict
from typing import Dict, Optional, Tuple

from e3nn.o3 import Irreps
from loguru import logger

from matten.data.irreps import ModuleIrreps
from matten.log import get_log_level
from matten.nn.sequential import Sequential
from matten.nn.utils import DetectAnomaly


def create_sequential_module(
    modules: "OrderedDict[str, Tuple[ModuleIrreps, Dict]]",
    irreps_in: Optional[Dict[str, Irreps]] = None,
    use_kwargs_irreps_in: bool = False,
) -> Sequential:
    """
    Create a sequential module from a sequence of modules.

    The main purpose of this function is to automatically determine the irreps for
    consecutive modules, i.e. using the output irreps of module i as the input irreps
    of module i+1.

    Args:
        modules: modules to create the sequential module. It should be look like
            {name: (ModuleClass, kwargs)}. ``kwargs`` should be a dictionary and it will
            be used to instantiate the class via ``ModuleClass(**kwargs)``.
            Note, the output irreps of the previous module will be added to kwargs with
            key ``irreps_in``.
        irreps_in: input irreps for the first module.
        use_kwargs_irreps_in: whether to use the irreps_in if it is provided as
            kwargs for a module. If `True`, the irreps_out of module i-1 and the
            irreps_in in kwargs of module i (higher priority) are combined as the
            irreps_in for module i.

    Returns:
        a sequential module
    """

    module_names = []
    module_instances = []
    for name, (cls_type, kwargs) in modules.items():
        if len(module_instances) == 0:
            ir = irreps_in
        else:
            ir = module_instances[-1].irreps_out

        if "irreps_in" in kwargs:
            if use_kwargs_irreps_in:
                if ir is not None:
                    # order matters: irreps_in in kwargs takes higher priority
                    ir.update(kwargs["irreps_in"])
                else:
                    ir = kwargs["irreps_in"]
            else:
                raise ValueError(
                    f"Trying to automatically determine irreps_in for module {name} "
                    f"But it is provided as kwargs. Set `use_kwargs_irrpes_in=True` to "
                    f"force it."
                )

        kwargs = kwargs.copy()
        kwargs["irreps_in"] = ir

        cls_name = cls_type.__name__
        try:
            m = cls_type(**kwargs)
        except Exception:
            raise RuntimeError(
                f"Failed instantiate module `{cls_name}` with kwargs: `{kwargs}`"
            )
        logger.info(f"Instantiate module `{cls_name}` with kwargs: `{kwargs}`")
        logger.debug(
            f"Module `{cls_name}` has irreps_in = `{m.irreps_in}`, and has "
            f"irreps_out = `{m.irreps_out}`"
        )

        module_names.append(name)
        module_instances.append(m)

        #
        # add anomaly detecting when in debug mode
        #
        if get_log_level() == "DEBUG":
            module_names.append(DetectAnomaly.__name__ + "_" + name)
            module_instances.append(DetectAnomaly(irreps_in=m.irreps_out, name=name))

    modules_dict = OrderedDict(zip(module_names, module_instances))

    return Sequential(modules_dict)
