from e3nn.o3 import Irreps

from matten.data.irreps import DataKey
from matten.nn.transformer_conv import TransformerConv


def test_Transformer():

    conv = TransformerConv(
        irreps_in={
            DataKey.NODE_FEATURES: Irreps("4x1e+2x0e"),
            DataKey.NODE_ATTRS: Irreps("4x1e+2x0e"),
            DataKey.EDGE_ATTRS: Irreps("4x1e+2x0e"),
            DataKey.EDGE_EMBEDDING: Irreps("4x0e"),
        },
        irreps_out={
            DataKey.NODE_FEATURES: Irreps("8x1e+4x0e"),
        },
        irreps_query_and_key=Irreps("8x1e+4x0e"),
        r_max=5.0,
    )
