from typing import Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import ones, zeros
from torch_geometric.utils import (add_self_loops, degree, dropout_adj,
                                   remove_self_loops)
try:
    from torch_sparse import SparseTensor, matmul, set_diag, sum
except ModuleNotFoundError:
    SparseTensor = type(None)

    def matmul(*args, **kwargs):
        raise ImportError("torch_sparse is not installed, but this branch should not be used in this project.")

    def set_diag(*args, **kwargs):
        raise ImportError("torch_sparse is not installed, but this branch should not be used in this project.")

    def sum(*args, **kwargs):
        raise ImportError("torch_sparse is not installed, but this branch should not be used in this project.")

from layers.types import Adj, OptPairTensor, OptTensor


class ScaleGConv(MessagePassing):
    def __init__(self, channels: int, dropedge: float = 0.,
                 add_self_loops: bool = True,
                 cached: bool = False, normalize: bool = True,
                 bias: bool = False, aggr="add", **kwargs):
        super(ScaleGConv, self).__init__(aggr=aggr, **kwargs)

        self.channels = channels
        self.dropedge = dropedge
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.weight = Parameter(torch.Tensor(channels))

        if bias:
            self.bias = Parameter(torch.Tensor(channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, x_0: Tensor,
                alpha: float, beta: float, edge_weight: OptTensor = None):
        edge_index, edge_weight = self.get_norm(x, edge_index, edge_weight)

        support = (1 - alpha) * (1 - beta) * x + beta * x * self.weight
        initial = alpha * (1 - beta) * x_0 + beta * x_0 * self.weight
        out = self.propagate(edge_index, x=support, edge_weight=edge_weight, size=None) + initial

        if self.bias is not None:
            out = out + self.bias
        return out

    def get_norm(self, x, edge_index: Adj, edge_weight: OptTensor = None):
        edge_index, edge_weight = dropout_adj(edge_index, edge_weight,
                                              p=self.dropedge, force_undirected=True,
                                              num_nodes=x.size(0), training=self.training)
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        True, self.add_self_loops, dtype=x.dtype)
                    # print(edge_weight)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        True, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        return edge_index, edge_weight

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        assert edge_weight is not None
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.channels)


class ScaleStarGConv(MessagePassing):
    def __init__(self, channels: int, dropedge: float = 0.,
                 add_self_loops: bool = True,
                 cached: bool = False, normalize: bool = True,
                 bias: bool = False, aggr="add", **kwargs):
        super(ScaleStarGConv, self).__init__(aggr=aggr, **kwargs)

        self.channels = channels
        self.dropedge = dropedge
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.weight1 = Parameter(torch.Tensor(channels))
        self.weight2 = Parameter(torch.Tensor(channels))

        if bias:
            self.bias = Parameter(torch.Tensor(channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight1)
        ones(self.weight2)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, x_0: Tensor,
                alpha: float, beta: float, edge_weight: OptTensor = None):
        edge_index, edge_weight = self.get_norm(x, edge_index, edge_weight)

        support = (1 - alpha) * (1 - beta) * x + beta * x * self.weight1
        initial = alpha * (1 - beta) * x_0 + beta * x_0 * self.weight2
        out = self.propagate(edge_index, x=support, edge_weight=edge_weight, size=None) + initial

        if self.bias is not None:
            out = out + self.bias
        return out

    def get_norm(self, x, edge_index: Adj, edge_weight: OptTensor = None):
        edge_index, edge_weight = dropout_adj(edge_index, edge_weight,
                                              p=self.dropedge, force_undirected=True,
                                              num_nodes=x.size(0), training=self.training)
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        True, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        True, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        return edge_index, edge_weight

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        assert edge_weight is not None
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.channels)


class BiScaleStarGConv(MessagePassing):
    def __init__(self, channels: int, dropout: float = 0.,
                 layernorm: bool = True, bias: bool = True, aggr: str = "add",
                 **kwargs):
        super(BiScaleStarGConv, self).__init__(aggr=aggr, **kwargs)

        self.channels = channels
        self.add_self_loops = add_self_loops

        self.weight_src = Parameter(torch.Tensor(channels))
        self.weight_dst = Parameter(torch.Tensor(channels))

        self.dropout = torch.nn.Dropout(p=dropout)
        self.layernorm = torch.nn.LayerNorm(channels, elementwise_affine=True) if layernorm else None

        if bias:
            self.bias = Parameter(torch.Tensor(channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight_src)
        ones(self.weight_dst)
        zeros(self.bias)
        if self.layernorm is not None:
            self.layernorm.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size=None, res_connect=False):
        if isinstance(x, Tensor):
            x = (x, x)

        edge_index, edge_weight = self.get_norm(x, edge_index, size)

        x_l = x[0] * self.weight_src
        x_r = x[1]
        out = self.propagate(edge_index, x=(x_l, x_r), edge_weight=edge_weight, size=size)
        if x_r is not None:
            out = out + x_r * self.weight_dst

        if self.bias is not None:
            out = out + self.bias

        out = self.dropout(out)
        if self.layernorm is not None:
            out = self.layernorm(out)
        if res_connect:
            out = out + x_r if x_r is not None else x[0]
        return out

    def message(self, x_j: Tensor, edge_weight=None):
        return edge_weight.view(-1, 1) * x_j if edge_weight is not None else x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def get_norm(self, x, edge_index, size):
        if isinstance(edge_index, Tensor):
            num_nodes = size[1] if size is not None else x[0].size(self.node_dim)
            if x[1] is not None: num_nodes = x[1].size(self.node_dim)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

            _, col = edge_index[0], edge_index[1]
            deg_inv = 1. / degree(col, num_nodes=num_nodes).clamp_(1.)

            edge_weight = deg_inv[col]

        elif isinstance(edge_index, SparseTensor):
            edge_index = set_diag(edge_index)

            col, _, _ = edge_index.coo()  # Transposed.
            deg_inv = 1. / sum(edge_index, dim=1).clamp_(1.)

            edge_weight = deg_inv[col]
            edge_index = edge_index.set_value(edge_weight, layout='coo')

        return edge_index, edge_weight
