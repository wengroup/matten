"""
In and out irreps of a module.

This is a recreation of class GraphModuleMixin
https://github.com/mir-group/nequip/blob/main/nequip/nn/_graph_mixin.py
to make it general for different data.
"""
from typing import Dict, Sequence

from e3nn.o3 import Irreps

from matten.data import _key

DataKey = _key


class ModuleIrreps:
    """
    Expected input and output irreps of a module.

    subclass can implement:
      - REQUIRED_KEY_IRREPS_IN
      - REQUIRED_TYPE_IRREPS_IN
      - OPTIONAL_MUL_TYPE_IRREPS_IN
      - fix_irreps_in

    ``None`` is a valid irreps in the context for anything that is invariant but not
    well described by an ``e3nn.o3.Irreps``. An example are edge indexes in a graph,
    which are invariant but are integers, not ``0e`` scalars.
    """

    REQUIRED_KEYS_IRREPS_IN = None
    REQUIRED_TYPE_IRREPS_IN = None
    OPTIONAL_MUL_TYPE_IRREPS_IN = None

    def init_irreps(
        self,
        irreps_in: Dict[str, Irreps] = None,
        irreps_out: Dict[str, Irreps] = None,
        *,
        required_keys_irreps_in: Sequence[str] = None,
        required_type_irreps_in: Dict[str, Irreps] = None,
        optional_mul_type_irreps_in: Dict[str, Irreps] = None,
    ):
        """
        Args:
            irreps_in: input irreps, availables keys in `DataKey`
            irreps_out: a dict of output irreps, available keys in `DataKey`. If a
                string, it should be a key in irreps_in, and then irreps_out will be
                set to {key: irreps_in[key]}
            required_keys_irreps_in: the required keys should be present in irreps_in.
                This only requires the irreps is given in `irreps_in`; does not specify
                what the irreps should be.
            required_type_irreps_in: for irreps in this dict, it should be present in
                irreps_in. In addition, their type (i.e. degree and parity) should
                match; multiplicity is not needed to match.
            optional_mul_type_irreps_in: for irreps in this dict, if it is given in
                `irreps_in`, they should match (all multiplicity, degree, and parity).
                If it is not given in `irreps_in`, no check will be made.
        """

        # input irreps
        irreps_in = {} if irreps_in is None else irreps_in
        irreps_in = _fix_irreps_dict(irreps_in)
        irreps_in = self.fix_irreps_in(irreps_in)

        # output irreps
        if irreps_out is None:
            irreps_out = {}
        elif isinstance(irreps_out, str):
            assert (
                irreps_out in irreps_in
            ), f"`irreps_in` does not contain key for `irreps_out = {irreps_out}`"
            irreps_out = {irreps_out, irreps_in[irreps_out]}
        irreps_out = _fix_irreps_dict(irreps_out)

        # required keys of input irreps
        required_keys = (
            [] if self.REQUIRED_KEYS_IRREPS_IN is None else self.REQUIRED_KEYS_IRREPS_IN
        )
        if required_keys_irreps_in is not None:
            required_keys += list(required_keys_irreps_in)

        # required type of input irreps
        required_type = (
            {} if self.REQUIRED_TYPE_IRREPS_IN is None else self.REQUIRED_TYPE_IRREPS_IN
        )
        if required_type_irreps_in is not None:
            required_type.update(required_type_irreps_in)
        required_type = _fix_irreps_dict(required_type)

        # optional exact input irreps
        optional = (
            {}
            if self.OPTIONAL_MUL_TYPE_IRREPS_IN is None
            else self.OPTIONAL_MUL_TYPE_IRREPS_IN
        )
        if optional_mul_type_irreps_in is not None:
            optional.update(optional_mul_type_irreps_in)
        optional = _fix_irreps_dict(optional)

        # Check compatibility

        # check required keys
        for k in required_keys + list(required_type.keys()):
            if k not in irreps_in:
                raise ValueError(
                    f"This module {type(self)} requires `{k}` in `irreps_in`."
                )

        # check required type
        for k, v in required_type.items():
            ok = _check_irreps_type(irreps_in[k], v)
            if not ok:
                raise ValueError(
                    f"This module {type(self)} expects irreps_in['{k}'] be of type {v}, "
                    f"instead got {irreps_in[k]}. Note, type means degree and parity, "
                    f"not multiplicity."
                )

        # check optional exact
        for k, v in optional.items():
            if k in irreps_in and irreps_in[k] != optional[k]:
                raise ValueError(
                    f"This module {type(self)} expects irreps_in['{k}'] to be {v}, "
                    f"instead got {irreps_in[k]}."
                )

        # Save stuff
        self._irreps_in = irreps_in

        # The output irreps of any graph module are whatever inputs it has,
        # overwritten with whatever outputs it has.
        self._irreps_out = irreps_in.copy()
        self._irreps_out.update(irreps_out)

    @property
    def irreps_in(self):
        return self._irreps_in

    @property
    def irreps_out(self):
        return self._irreps_out

    def fix_irreps_in(self, irreps_in: Dict[str, Irreps]) -> Dict[str, Irreps]:
        """
        Fix the input irreps.
        """
        irreps_in = irreps_in.copy()

        # positions are always 1o and should always be present in
        pos = DataKey.POSITIONS
        if pos in irreps_in and irreps_in[pos] != Irreps("1x1o"):
            raise ValueError(f"Positions must have irreps 1o, got `{irreps_in[pos]}`")
        irreps_in[pos] = Irreps("1o")

        # edges are always None and should always be present
        edge_index = DataKey.EDGE_INDEX
        if edge_index in irreps_in and irreps_in[edge_index] is not None:
            raise ValueError(
                f"Edge indexes must have irreps `None`, got `{irreps_in[edge_index]}`"
            )
        irreps_in[edge_index] = None

        return irreps_in


# copied from nequip:
# https://github.com/mir-group/nequip/blob/main/nequip/data/AtomicData.py
def _fix_irreps_dict(irreps: Dict[str, Irreps]) -> Dict[str, Irreps]:
    """
    Fix irreps.

      - convert string representation to object
      - deal with None. ``None`` is a valid irreps in the context for anything that
        is invariant but not well described by an ``e3nn.o3.Irreps``. An example are
        edge indexes in a graph, which are invariant but are integers, not ``0e``
        scalars.
    """
    special_irreps = [None]
    irreps = {k: (i if i in special_irreps else Irreps(i)) for k, i in irreps.items()}

    return irreps


# copied from nequip:
# https://github.com/mir-group/nequip/blob/main/nequip/data/AtomicData.py
def _check_irreps_compatible(ir1: Dict[str, Irreps], ir2: Dict[str, Irreps]):
    return all(ir1[k] == ir2[k] for k in ir1 if k in ir2)


def _check_irreps_type(irreps1: Irreps, irreps2: Irreps) -> bool:
    """
    Check the type of the two irreps are the same.

    This only checks the type (degree and parity), not the multiplicity.

    Args:
        irreps1: the first irreps
        irreps2: the second irreps
    """
    irreps1 = Irreps(irreps1)
    irreps2 = Irreps(irreps2)

    # ls gives all the l of an irreps
    if set(irreps1.ls) == set(irreps2.ls):
        return True
    else:
        return False
