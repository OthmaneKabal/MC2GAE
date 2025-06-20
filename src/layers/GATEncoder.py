from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    def __init__(self, data: Data, out_channels, num_layers=2, heads=4, dropout=0.5):
        super(GATEncoder, self).__init__()
        self.out_channels = out_channels[-1]
        in_channels = data.x.shape[1]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.heads = heads

        for i in range(num_layers):
            input_dim = in_channels if i == 0 else out_channels[i - 1]
            self.convs.append(GATConv(input_dim, out_channels[i], heads=heads, concat=False, dropout=dropout))
            # Since concat=False, output dimension is out_channels[i], not out_channels[i] * heads
            self.bns.append(nn.BatchNorm1d(out_channels[i]))

        self.dropout = nn.Dropout(p=dropout)

        # Final layer - input is out_channels[-1] since concat=False
        self.final_layer = nn.Linear(out_channels[-1], out_channels[-1])
        self.relu = nn.ReLU()

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.final_layer(x)
        return x