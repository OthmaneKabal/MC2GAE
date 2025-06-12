from torch import nn
from torch_geometric.nn import GATConv


class GATDecoder(nn.Module):
    def __init__(self, encoder: nn.Module, data, heads=1, alpha=0.01, dropout=0.5):
        super(GATDecoder, self).__init__()

        # Get encoder configuration
        if isinstance(encoder.convs[0], GATConv):
            # Since encoder uses concat=False, output channels are just out_channels
            encoder_out_channels = [conv.out_channels for conv in encoder.convs]
        else:
            encoder_out_channels = [conv.out_channels for conv in encoder.convs]

        encoder_in_channels = encoder.convs[0].in_channels

        # Create decoder architecture (reverse of encoder)
        decoder_out_channels = list(reversed(encoder_out_channels)) + [encoder_in_channels]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(len(decoder_out_channels) - 1):
            self.convs.append(GATConv(decoder_out_channels[i], decoder_out_channels[i + 1],
                                      heads=heads, concat=False, dropout=dropout))
            # Since concat=False, output dimension is decoder_out_channels[i + 1]
            self.bns.append(nn.BatchNorm1d(decoder_out_channels[i + 1]))

        self.dropout = nn.Dropout(p=dropout)

        # Final layer to reconstruct original features
        self.final_layer = nn.Linear(decoder_out_channels[-1], data.num_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)

    def forward(self, data, embeddings):
        x, edge_index = embeddings, data.edge_index

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)

        x = self.final_layer(x)
        return x
