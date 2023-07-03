import torch

from matten.data.irreps import DataKey
from matten.nn.nodewise import NodewiseSelect


def test_atomwise_select():

    node_feats = DataKey.NODE_FEATURES
    mask_field = "node_masks"
    out_field = "selected_node_features"
    aws = NodewiseSelect(
        irreps_in={node_feats: None, mask_field: None},
        field=node_feats,
        out_field=out_field,
        mask_field=mask_field,
    )

    n_atoms = 5
    data = {
        node_feats: torch.arange(n_atoms * 2).reshape(n_atoms, 2),
        mask_field: torch.tensor([True, False, True, True, False]),
    }

    out = aws(data)

    assert torch.allclose(out[out_field], data[node_feats][data[mask_field]])
