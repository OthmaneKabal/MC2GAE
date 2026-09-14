from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data

try:
    from torch_geometric import EdgeIndex
    from torch_geometric.nn import CuGraphGATConv
except ImportError:
    EdgeIndex = None
    CuGraphGATConv = None


def as_cugraph_edge_index(edge_index, num_nodes: int):
    """Return a column-sorted EdgeIndex for PyG's cuGraph operators."""
    if EdgeIndex is None:
        raise ImportError(
            "CuGraphGATConv requires a PyG build exposing EdgeIndex."
        )
    if isinstance(edge_index, EdgeIndex):
        return edge_index

    source, target = edge_index[0], edge_index[1]
    order = torch.argsort(target)
    sorted_edges = torch.stack((source[order], target[order]), dim=0)
    return EdgeIndex(
        sorted_edges,
        sparse_size=(num_nodes, num_nodes),
        sort_order="col",
    )


class CuGraphGATEncoder(nn.Module):
    """GAT encoder using PyG's fused cuGraph GAT operator."""

    def __init__(self, data: Data, out_channels, num_layers=2, heads=4,
                 dropout=0.5):
        super().__init__()
        if CuGraphGATConv is None:
            raise ImportError(
                "CuGraphGATConv is unavailable. Install a compatible PyG "
                "cuGraph/cugraph-ops build on the CUDA server."
            )
        if len(out_channels) != num_layers:
            raise ValueError("len(out_channels) must equal num_layers")

        self.out_channels = out_channels[-1]
        self.heads = heads
        in_channels = int(data.x.shape[1])
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for layer_index in range(num_layers):
            input_dim = in_channels if layer_index == 0 else out_channels[layer_index - 1]
            self.convs.append(
                CuGraphGATConv(
                    input_dim,
                    out_channels[layer_index],
                    heads=heads,
                    concat=False,
                )
            )
            self.bns.append(nn.BatchNorm1d(out_channels[layer_index]))

        self.dropout = nn.Dropout(p=dropout)
        self.final_layer = nn.Linear(out_channels[-1], self.out_channels)
        self.relu = nn.ReLU()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_layer.reset_parameters()

    def forward(self, data: Data):
        x = data.x
        edge_index = as_cugraph_edge_index(data.edge_index, x.size(0))
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
        return self.final_layer(x)
