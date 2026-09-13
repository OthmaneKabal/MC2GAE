from torch import nn
import torch
from torch_geometric.data import Data

try:
    from torch_geometric import EdgeIndex
    from torch_geometric import __version__ as pyg_version
    from torch_geometric.nn import CuGraphRGCNConv
except ImportError:
    EdgeIndex = None
    pyg_version = "0.0.0"
    CuGraphRGCNConv = None


def _version_at_least(version, major, minor):
    parts = version.split(".")
    try:
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (IndexError, ValueError):
        return False


class CuGraphRGCNEncoder(nn.Module):
    """RGCN encoder backed by PyG's CUDA/cuGraph operator.

    The cuGraph operator consumes CSC adjacency data instead of edge_index.
    CSC is built per sampled PyG batch because NeighborLoader changes the
    local node and edge numbering at every batch.
    """

    def __init__(self, data: Data, out_channels, num_layers=2, num_bases=30,
                 dropout=0.5, message_sens="source_to_target"):
        super().__init__()
        if CuGraphRGCNConv is None:
            raise ImportError(
                "CuGraphRGCNConv is unavailable. Install a compatible PyG "
                "cuGraph/cugraph-ops build on the CUDA server."
            )
        if message_sens not in ("source_to_target", "target_to_source"):
            raise ValueError(f"Unsupported message direction: {message_sens}")
        if len(out_channels) != num_layers:
            raise ValueError("len(out_channels) must equal num_layers")

        self.out_channels = out_channels[-1]
        self.message_sens = message_sens
        in_channels = int(data.x.shape[1])
        num_relations = int(data.edge_type.max().item()) + 1
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for layer_index in range(num_layers):
            input_dim = in_channels if layer_index == 0 else out_channels[layer_index - 1]
            self.convs.append(
                CuGraphRGCNConv(
                    input_dim,
                    out_channels[layer_index],
                    num_relations,
                    num_bases=num_bases,
                    aggr="mean",
                )
            )
            self.bns.append(nn.BatchNorm1d(out_channels[layer_index]))

        self.dropout = nn.Dropout(p=dropout)
        self.final_layer = nn.Linear(out_channels[-1], self.out_channels)
        self.relu = nn.ReLU()

    @staticmethod
    def _prepare_adjacency(edge_index, edge_type, num_nodes, message_sens):
        if message_sens == "source_to_target":
            source, target = edge_index[0], edge_index[1]
        else:
            source, target = edge_index[1], edge_index[0]

        order = torch.argsort(target)
        row = source[order].contiguous()
        sorted_target = target[order]
        sorted_edge_type = edge_type[order].contiguous()

        # PyG >= 2.7 exposes the cuGraph operator with EdgeIndex input.
        # Older PyG releases used the (row, colptr) CSC tuple instead.
        # PyG 2.6 introduced the EdgeIndex input expected by the current
        # cuGraph operator. Older releases expect the explicit CSC tuple.
        if EdgeIndex is not None and _version_at_least(pyg_version, 2, 6):
            sorted_edge_index = torch.stack((source[order], target[order]), dim=0)
            return (
                EdgeIndex(
                    sorted_edge_index,
                    sparse_size=(num_nodes, num_nodes),
                    sort_order="col",
                ),
                sorted_edge_type,
            )

        counts = torch.bincount(sorted_target, minlength=num_nodes)
        colptr = torch.cat((
            torch.zeros(1, dtype=torch.long, device=edge_index.device),
            counts.cumsum(0),
        )).contiguous()
        # PyG 2.5.x expects the source-node count as the third CSC item.
        return (row, colptr, num_nodes), sorted_edge_type

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_layer.reset_parameters()

    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        # NeighborLoader may preserve edge attributes as [E, 1], while the
        # cuGraph operator requires a flat integer relation vector [E].
        edge_type = data.edge_type.reshape(-1).to(dtype=torch.long).contiguous()
        if edge_type.numel() != edge_index.size(1):
            raise ValueError(
                "CuGraphRGCNEncoder received mismatched edge metadata: "
                f"edge_index has {edge_index.size(1)} edges but edge_type has "
                f"{edge_type.numel()} values."
            )
        adjacency, edge_type = self._prepare_adjacency(
            edge_index, edge_type, x.size(0), self.message_sens
        )

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, adjacency, edge_type)
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
        return self.final_layer(x)
