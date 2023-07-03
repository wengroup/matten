from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as fn
from e3nn.nn import BatchNorm, FullyConnectedNet, Gate, NormActivation
from e3nn.o3 import Irrep, Irreps, TensorProduct
from torch import Tensor
from torch_scatter import scatter

from matten.data.irreps import DataKey, ModuleIrreps
from matten.nn._nequip import ShiftedSoftPlus
from matten.utils import detect_nan_and_inf

ACTIVATION = {
    # for even irreps
    "e": {
        "ssp": ShiftedSoftPlus(),
        "silu": fn.silu,
        "sigmoid": torch.sigmoid,
    },
    # for odd irreps
    "o": {
        "abs": torch.abs,
        "tanh": torch.tanh,
    },
}


class ActivationLayer(torch.nn.Module):
    def __init__(
        self,
        tp_irreps_in1: Irreps,
        tp_irreps_in2: Irreps,
        tp_irreps_out: Irreps,
        *,
        activation_type: str = "gate",
        activation_scalars: Dict[str, str] = None,
        activation_gates: Dict[str, str] = None,
    ):
        """
        Nonlinear equivariant activation function layer.

        This is intended to be applied after a tensor product convolution layer.

        Args:
            tp_irreps_in1: first irreps for the tensor product layer
            tp_irreps_in2: second irreps for the tensor product layer
            tp_irreps_out: intended output irreps for the tensor product layer.
                Note, typically this is not the actual irreps out we will use for the
                tensor product. The actual one is determined here, i.e. the `irreps_in`
                attribute of this class.
            activation_type: `gate` or `norm`
            activation_scalars: activation function for scalar irreps (i.e. l=0).
                Should be something like {'e':act_e, 'o':act_o}, where `act_e` is the
                name of the activation function ('ssp' or 'silu') for even irreps;
                `act_o` is the name of the activation function ('abs' or 'tanh') for
                odd irreps.
            activation_gates: activation function for tensor irreps (i.e. l>0) when
                using the `Gate` activation. Ignored for `NormActivation`.
                Should be something like {'e':act_e, 'o':act_o}, where `act_e` is the
                name of the activation function ('ssp' or 'silu') for even irreps;
                `act_o` is the name of the activation function ('abs' or 'tanh') for
                odd irreps.
        """
        super().__init__()

        # set defaults

        # activation function for even (i.e. 1) and odd (i.e. -1) scalars
        if activation_scalars is None:
            activation_scalars = {
                1: ACTIVATION["e"]["ssp"],
                # odd scalars requires either an even or odd activation,
                # not an arbitrary one like relu
                -1: ACTIVATION["o"]["tanh"],
            }
        else:
            # change key from e or v to 1 or -1
            key_mapping = {"e": 1, "o": -1}
            activation_scalars = {
                key_mapping[k]: ACTIVATION[k][v] for k, v in activation_scalars.items()
            }

        # activation function for even (i.e. 1) and odd (i.e. -1) high-order tensors
        if activation_gates is None:
            activation_gates = {1: ACTIVATION["e"]["ssp"], -1: ACTIVATION["o"]["abs"]}
        else:
            # change key from e or v to 1 or -1
            key_mapping = {"e": 1, "o": -1}
            activation_gates = {
                key_mapping[k]: ACTIVATION[k][v] for k, v in activation_gates.items()
            }

        # in and out irreps of activation

        ir_tmp, _, _ = Irreps(tp_irreps_out).sort()
        tp_irreps_out = ir_tmp.simplify()

        irreps_scalars = Irreps(
            [
                (mul, ir)
                for mul, ir in tp_irreps_out
                if ir.l == 0 and tp_path_exists(tp_irreps_in1, tp_irreps_in2, ir)
            ]
        )
        irreps_gated = Irreps(
            [
                (mul, ir)
                for mul, ir in tp_irreps_out
                if ir.l > 0 and tp_path_exists(tp_irreps_in1, tp_irreps_in2, ir)
            ]
        )

        if activation_type == "gate":
            # Setting all ir to 0e if there is path exist, since 0e gates will not
            # change the parity of the output, and we want to keep its parity.
            # If there is no 0e, we use 0o to change the party of the high-order irreps.
            if irreps_gated.dim > 0:
                if tp_path_exists(tp_irreps_in1, tp_irreps_in2, "0e"):
                    ir = "0e"
                elif tp_path_exists(tp_irreps_in1, tp_irreps_in2, "0o"):
                    ir = "0o"
                else:
                    raise ValueError(
                        f"tp_irreps_in1={tp_irreps_in1} times "
                        f"tp_irreps_in2={tp_irreps_in2} is unable to produce gates "
                        f"needed for irreps_gated={irreps_gated}"
                    )
            else:
                ir = None
            # irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            self.activation = Gate(
                irreps_scalars=irreps_scalars,  # scalars
                act_scalars=[activation_scalars[ir.p] for _, ir in irreps_scalars],
                irreps_gates=irreps_gates,  # gates (scalars)
                act_gates=[activation_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,  # gated tensors
            )

        elif activation_type == "norm":

            self.activation = NormActivation(
                irreps_in=(irreps_scalars + irreps_gated).simplify(),
                # norm is an even scalar, so activation_scalars[1]
                scalar_nonlinearity=activation_scalars[1],
                normalize=True,
                epsilon=1e-8,
                bias=False,
            )

        else:
            supported = ("gate", "norm")
            raise ValueError(
                f"Support `activation_type` includes {supported}, got {activation_type}"
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x)

    @property
    def irreps_in(self):
        return self.activation.irreps_in

    @property
    def irreps_out(self):
        return self.activation.irreps_out


class UVUTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        *,
        internal_and_share_weights: bool = False,
        mlp_input_size: int = None,
        mlp_hidden_size: int = 8,
        mlp_num_hidden_layers: int = 1,
        mlp_activation: Callable = ACTIVATION["e"]["ssp"],
    ):
        """
        UVU tensor product.

        Args:
            irreps_in1: irreps of first input, with available keys in `DataKey`
            irreps_in2: input of second input, with available keys in `DataKey`
            irreps_out: output irreps, with available keys in `DataKey`
            internal_and_share_weights: whether to create weights for the tensor
                product, if `True` all `mlp_*` params are ignored and if `False`,
                they should be provided to create an MLP to transform some data to be
                used as the weight of the tensor product.
            mlp_input_size: size of the input data used as the weight for the tensor
                product transformation via an MLP
            mlp_hidden_size: hidden layer size for the MLP
            mlp_num_hidden_layers: number of hidden layers for the radial MLP, excluding
                input and output layers
            mlp_activation: activation function for the MLP.
        """

        super().__init__()

        # uvu instructions for tensor product
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(irreps_in1):
            for j, (_, ir_in2) in enumerate(irreps_in2):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in irreps_out or ir_out == Irreps("0e"):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = Irreps(irreps_mid)

        assert irreps_mid.dim > 0, (
            f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} produces no "
            f"instructions in irreps_out={irreps_out}"
        )

        # sort irreps_mid to let irreps of the same type be adjacent to each other
        self.irreps_mid, permutation, _ = irreps_mid.sort()

        # sort instructions accordingly
        instructions = [
            (i_1, i_2, permutation[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        self.tp = TensorProduct(
            irreps_in1,
            irreps_in2,
            self.irreps_mid,
            instructions,
            internal_weights=internal_and_share_weights,
            shared_weights=internal_and_share_weights,
        )

        if not internal_and_share_weights:
            assert mlp_input_size is not None, (
                "Expect `mlp_input_size` be provided when "
                "`internal_and_share_weights` is set to `False`, got `None`"
            )

            # radial network on scalar edge embedding (e.g. edge distance)
            layer_sizes = (
                [mlp_input_size]
                + mlp_num_hidden_layers * [mlp_hidden_size]
                + [self.tp.weight_numel]
            )
            self.weight_nn = FullyConnectedNet(layer_sizes, act=mlp_activation)
        else:
            self.weight_nn = None

    def forward(
        self, data1: Tensor, data2: Tensor, data_weight: Optional[Tensor] = None
    ) -> Tensor:

        if self.weight_nn is not None:
            assert data_weight is not None, "data for weight not provided"
            weight = self.weight_nn(data_weight)
        else:
            weight = None
        x = self.tp(data1, data2, weight)

        return x

    @property
    def irreps_out(self):
        """
        Output irreps of the layer.

        This is different from the input `irreps_out`, since we use the UVU tensor
        product with given instructions.
        """
        # should be fine to simplify, which will affect the normalization of the next
        # layer that uses this as irreps_in
        return self.irreps_mid.simplify()


class ScalarMLP(torch.nn.Module):
    """
    Multilayer perceptron for scalars.

    By default, activation is applied to each hidden layer. For hidden layers:
    Linear -> BN (default to False) -> Activation

    Optionally, one can add an output layer by setting `out_size`. For output layer:
    Linear with the option to use bias or not, but activation is not applied.

    Args:
        in_size: input feature size
        hidden_sizes: sizes for hidden layers
        batch_norm: whether to add 1D batch norm
        activation: activation function for hidden layers
        out_size: size of output layer
        out_bias: bias for output layer, this use set to False internally if
            out_batch_norm is used.
    """

    def __init__(
        self,
        in_size: int,
        hidden_sizes: List[int],
        *,
        batch_norm: bool = False,
        activation: Callable = ACTIVATION["e"]["ssp"],
        out_size: Optional[int] = None,
        out_bias: bool = True,
    ):
        super().__init__()
        self.num_hidden_layers = len(hidden_sizes)
        self.has_out_layer = out_size is not None

        layers = []

        # hidden layers
        if batch_norm:
            bias = False
        else:
            bias = True

        for size in hidden_sizes:
            layers.append(torch.nn.Linear(in_size, size, bias=bias))

            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(size))

            if activation is not None:
                layers.append(activation)

            in_size = size

        # output layer
        if out_size is not None:
            layers.append(torch.nn.Linear(in_size, out_size, bias=out_bias))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

    def __repr__(self):
        s = f"ScalarMLP, num hidden layers: {self.num_hidden_layers}"
        if self.has_out_layer:
            s += "; with output layer"
        return s


def scatter_add(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """
    Special case of torch_scatter.scatter with dim=0
    """
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class DetectAnomaly(ModuleIrreps, torch.nn.Module):
    """
    Check anomaly in a dict of tensor.

    This is supposed to be used as layer when constructing the model, and it will check
    the values of the data dict of its previous layer.
    """

    def __init__(self, irreps_in: Dict[str, Irreps], name: str):
        super().__init__()
        self.init_irreps(irreps_in=irreps_in)

        self.name = name

    def forward(self, data: DataKey.Type) -> DataKey.Type:
        for k, v in data.items():
            if v is None:
                continue

            try:
                detect_nan_and_inf(v)
            except ValueError:
                raise ValueError(f"Anomaly detected for {k} of {self.name}")

        return data


class NormalizationLayer(torch.nn.Module):
    """
    A wrapper class to do method.

    Args:
        irreps: irreps of the tensor
        method: normalization method; should be one of `batch`, `instance`, and `none`.
    """

    def __init__(self, irreps: Irreps, method: str = None):
        super().__init__()

        self.method = method

        supported = ("batch", "instance", "none", None)
        assert method in supported, f"Unsupported normalization {method}"

        if method is not None and method != "none":
            if method == "instance":
                self.n = InstanceNorm(irreps)
            else:
                self.n = BatchNorm(irreps)
        else:
            self.n = None

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """

        Args:
            x:
            batch: index of the graph the nodes belong to

        Returns:

        """
        if self.method == "batch":
            x = self.n(x)
        elif self.method == "instance":
            x = self.n(x, batch)

        return x


# TODO there is bug in the below module, as it does not distinguish training and
#  predicting mode when there is affine transformation.
# copied for the repo of segnn paper
# The e3nn InstanceNorm does not work, because it treats each node as an instance.
# However, for each node, there is only one channel, and thus instance normalization is
# impossible.
# This implementation treat each graph as an instance, and treat each node as a
# channel. Then instance is possible. This is more like a graph normalization.
class InstanceNorm(torch.nn.Module):
    """
    Instance normalization for orthonormal representations
    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.
    Parameters
    ----------
    irreps : `Irreps`
        representation
    eps : float
        avoid division by zero when we normalize by the variance
    affine : bool
        do we have weight and bias parameters
    reduce : {'mean', 'max'}
        method used to reduce
    """

    def __init__(
        self, irreps, eps=1e-5, affine=True, reduce="mean", normalization="component"
    ):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0)
        num_features = self.irreps.num_irreps

        if affine:
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # TODO, this does not work for saving data.
        # mean and stdev, and norm should be buffer such that we save the model and
        # load back, they will be recovered, Take a look at the e3nn BatchNorm
        # implementation

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, input, batch):
        """evaluate
        Parameters
        ----------
        input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        # batch, *size, dim = input.shape  # TODO: deal with batch
        # input = input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # input has shape [batch * nodes, dim], but with variable nr of nodes.
        # the input batch slices this into separate graphs
        dim = input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for (mul, ir) in self.irreps:
            # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = input[:, ix : ix + mul * d]  # [batch * sample, mul * repr]
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0:
                # Compute the mean
                field_mean = global_mean_pool(field, batch).reshape(
                    -1, mul, 1
                )  # [batch, mul, 1]]
                # Subtract the mean
                field = field - field_mean[batch]

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError(
                    "Invalid normalization option {}".format(self.normalization)
                )
            # Reduction method
            if self.reduce == "mean":
                field_norm = global_mean_pool(field_norm, batch)  # [batch, mul]
            elif self.reduce == "max":
                field_norm = global_max_pool(field_norm, batch)  # [batch, mul]
            else:
                raise ValueError("Invalid reduce option {}".format(self.reduce))

            # Then apply the rescaling (divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.weight[None, iw : iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm[batch].reshape(
                -1, mul, 1
            )  # [batch * sample, mul, repr]

            if self.affine and d == 1:  # scalars
                bias = self.bias[ib : ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output


def global_mean_pool(x, batch, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce="mean")


def global_max_pool(x, batch, size: Optional[int] = None):
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    maximum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by

    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce="max")
