from __future__ import annotations

from torch import nn

from CuGraphGATEncoder import CuGraphGATConv, as_cugraph_edge_index


class CuGraphGATDecoder(nn.Module):
    """Symmetric feature decoder using fused cuGraph GAT layers."""

    def __init__(self, encoder: nn.Module, data, heads=4, alpha=0.01,
                 dropout=0.5):
        super().__init__()
        if CuGraphGATConv is None:
            raise ImportError(
                "CuGraphGATConv is unavailable. Install a compatible PyG "
                "cuGraph/cugraph-ops build on the CUDA server."
            )

        encoder_out_channels = [conv.out_channels for conv in encoder.convs]
        encoder_in_channels = encoder.convs[0].in_channels
        decoder_out_channels = list(reversed(encoder_out_channels)) + [encoder_in_channels]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for index in range(len(decoder_out_channels) - 1):
            self.convs.append(
                CuGraphGATConv(
                    decoder_out_channels[index],
                    decoder_out_channels[index + 1],
                    heads=heads,
                    concat=False,
                )
            )
            self.bns.append(nn.BatchNorm1d(decoder_out_channels[index + 1]))

        self.dropout = nn.Dropout(p=dropout)
        self.final_layer = nn.Linear(decoder_out_channels[-1], data.num_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_layer.reset_parameters()

    def forward(self, data, embeddings):
        x = embeddings
        edge_index = as_cugraph_edge_index(data.edge_index, x.size(0))
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
        return self.final_layer(x)
