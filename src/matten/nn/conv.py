"""
Tensor field network based on the implementation at:
https://github.com/e3nn/e3nn/blob/bd873a1cfcbad36c2ac233e9e48d7b7c3c5892f4/e3nn/nn/models/v2106/gate_points_networks.py#L1

With one difference:
- remove the `alpha` part for self connection, which we found to tend to be zeros all
  the time.
"""


from typing import Dict

import torch
from e3nn.o3 import FullyConnectedTensorProduct, Irreps
from torch_scatter import scatter

from matten.data.irreps import DataKey, ModuleIrreps
from matten.nn.utils import (
    ACTIVATION,
    ActivationLayer,
    NormalizationLayer,
    UVUTensorProduct,
)


class PointConv(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        conv_layer_irreps: Irreps,
        fc_num_hidden_layers: int = 1,
        fc_hidden_size: int = 8,
        avg_num_neighbors: int = None,
    ):
        """
        Args:
            irreps_in:
            conv_layer_irreps:
            fc_num_hidden_layers:
            fc_hidden_size:
            avg_num_neighbors: average number of neighbors of each node used to
                normalize the aggregated message. If not `None`, the provided value
                will be used to normalize all the node features. If `None`, separate
                individual values will be used.
        """
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors

        self.init_irreps(irreps_in)

        node_feats_irreps_in = self.irreps_in[DataKey.NODE_FEATURES]
        node_attrs_irreps = self.irreps_in[DataKey.NODE_ATTRS]
        edge_attrs_irreps = self.irreps_in[DataKey.EDGE_ATTRS]
        conv_layer_irreps = Irreps(conv_layer_irreps)

        #
        # convolution
        #
        self.lin1 = FullyConnectedTensorProduct(
            node_feats_irreps_in, node_attrs_irreps, node_feats_irreps_in
        )

        self.tp = UVUTensorProduct(
            node_feats_irreps_in,
            edge_attrs_irreps,
            conv_layer_irreps,
            mlp_input_size=self.irreps_in[DataKey.EDGE_EMBEDDING].dim,
            mlp_hidden_size=fc_hidden_size,
            mlp_num_hidden_layers=fc_num_hidden_layers,
            mlp_activation=ACTIVATION["e"]["silu"],
        )

        # tp_irreps_out may not be the same as the requested irreps_out of the tp
        # (i.e. conv_layer_irreps) since UVU only products possible paths
        tp_irreps_out = self.tp.irreps_out

        self.lin2 = FullyConnectedTensorProduct(
            tp_irreps_out, node_attrs_irreps, conv_layer_irreps
        )

        #
        # self connection
        #
        self.sc = FullyConnectedTensorProduct(
            node_feats_irreps_in, node_attrs_irreps, conv_layer_irreps
        )

        # # inspired by https://arxiv.org/pdf/2002.10444.pdf
        # self.alpha = FullyConnectedTensorProduct(
        #     tp_irreps_out, node_attrs_irreps, Irreps("0e")
        # )
        # # with torch.no_grad():
        # #     self.alpha.weight.zero_()
        # assert self.alpha.output_mask[0] == 1.0, (
        #     f"tp_irreps_out={tp_irreps_out} and node_attrs_irreps"
        #     f"={self.irreps_node_attr} are not able to generate scalars"
        # )

        # add output irreps
        self.irreps_out[DataKey.NODE_FEATURES] = conv_layer_irreps

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        node_feats = data[DataKey.NODE_FEATURES]
        node_attrs = data[DataKey.NODE_ATTRS]
        edge_attrs = data[DataKey.EDGE_ATTRS]
        edge_embedding = data[DataKey.EDGE_EMBEDDING]
        edge_src, edge_dst = data[DataKey.EDGE_INDEX]

        node_self_connection = self.sc(node_feats, node_attrs)

        # message
        node_feats = self.lin1(node_feats, node_attrs)
        msg = self.tp(node_feats[edge_src], edge_attrs, edge_embedding)
        aggregated_msg = scatter(msg, edge_dst, dim_size=len(node_feats), dim=0)

        if self.avg_num_neighbors is not None:
            aggregated_msg = aggregated_msg.div(self.avg_num_neighbors**0.5)
        else:
            num_neigh = data[DataKey.NUM_NEIGH].reshape(-1, 1)
            aggregated_msg = aggregated_msg.div(num_neigh**0.5)

        # update
        node_conv_out = self.lin2(aggregated_msg, node_attrs)

        # # m=1: tp has path to generate result, m=0: has not
        # # So, the below snippet means:
        # # When m = 0 (i.e. no self connection path exists), alpha = 1; node feats
        # # will take full update from node_conv_out.
        # # When m = 1 (i.e. self connection path exists), alpha = alpha; node feats
        # # will only take alpha `amount` of node_conv_out. Actually, alpha is
        # # initialized to zero, so this means taking no node_conv_out at all.
        # # As learning progress, since alpha is learnable, this will gradually take
        # # effect.
        # alpha = self.alpha(aggregated_msg, node_attrs)
        # m = self.sc.output_mask
        # alpha = (1 - m) + alpha * m
        # node_feats = node_self_connection + alpha * node_conv_out

        node_feats = node_self_connection + node_conv_out

        data[DataKey.NODE_FEATURES] = node_feats

        return data


class PointConvWithActivation(ModuleIrreps, torch.nn.Module):
    def __init__(
        self,
        irreps_in: Dict[str, Irreps],
        conv_layer_irreps: Irreps,
        fc_num_hidden_layers: int = 1,
        fc_hidden_size: int = 8,
        avg_num_neighbors: int = None,
        activation_type: str = "gate",
        activation_scalars: Dict[str, str] = {"e": "silu", "o": "tanh"},
        activation_gates: Dict[str, str] = {"e": "sigmoid", "o": "tanh"},
        normalization: str = None,
    ):
        """
        Args:
            irreps_in:
            conv_layer_irreps:
            fc_num_hidden_layers:
            fc_hidden_size:
            avg_num_neighbors:
            activation_type:
            activation_scalars:
            activation_gates:
            normalization:
        """
        super().__init__()

        self.init_irreps(irreps_in)

        node_feats_irreps_in = self.irreps_in[DataKey.NODE_FEATURES]
        edge_attrs_irreps = self.irreps_in[DataKey.EDGE_ATTRS]
        conv_layer_irreps = Irreps(conv_layer_irreps)

        self.act = ActivationLayer(
            node_feats_irreps_in,
            edge_attrs_irreps,
            conv_layer_irreps,
            activation_type=activation_type,
            activation_scalars=activation_scalars,
            activation_gates=activation_gates,
        )
        act_irreps_in = self.act.irreps_in
        act_irreps_out = self.act.irreps_out

        self.conv = PointConv(
            irreps_in=self.irreps_in,
            conv_layer_irreps=act_irreps_in,
            fc_num_hidden_layers=fc_num_hidden_layers,
            fc_hidden_size=fc_hidden_size,
            avg_num_neighbors=avg_num_neighbors,
        )

        self.norm = NormalizationLayer(act_irreps_out, method=normalization)

        # add output irreps
        self.irreps_out[DataKey.NODE_FEATURES] = act_irreps_out

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        batch = data[DataKey.BATCH]

        data = self.conv(data)

        x = data[DataKey.NODE_FEATURES]
        x = self.act(x)

        x = self.norm(x, batch)

        data[DataKey.NODE_FEATURES] = x

        return data
