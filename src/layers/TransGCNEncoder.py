from torch import nn
from torch_geometric.data import Data
from typing import List
import sys
sys.path.append('/trans_gcn_layer')
from trans_gcn_layer.trans_gnn import TransGNN


class TransGCNEncoder(nn.Module):
    def __init__(self, data: Data, out_channels: List[int], num_layers=2,
                 dropout=0.5, kg_score_fn='TransE', variant='conv' ,use_edges_info=True, activation='relu', bias=False):
        """
        Encodeur basé sur TransGCN utilisant plusieurs couches TransGNN avec ReLU, Dropout, etc.

        :param data: objet Data de PyG
        :param out_channels: liste des dimensions de sortie par couche
        :param num_layers: nombre de couches TransGNN
        :param dropout: probabilité de dropout
        :param kg_score_fn: nom de la fonction de transformation ('TransE' ou autre)
        :param use_edges_info: si True, utilise les embeddings des relations
        :param activation: fonction d'activation
        :param bias: inclut ou non un biais dans les couches
        """
        super().__init__()
        assert len(out_channels) == num_layers, "out_channels doit avoir autant d'éléments que num_layers"
        self.out_channels = out_channels[-1]
        in_channels = data.x.shape[1]

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        print("transgcn0..**********************************************************************************.")
        for i in range(num_layers):
            layer_in = in_channels if i == 0 else out_channels[i - 1]
            layer_out = out_channels[i]

            self.layers.append(TransGNN(
                in_channels=layer_in,
                out_channels=layer_out,
                use_edges_info=use_edges_info,
                kg_score_fn=kg_score_fn,
                variant=variant,  # comme dans l'article (pas d'attention)
                activation=activation,
                bias=bias
            ))
        self.convs = self.layers

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
            x = self.dropout(x)
        return x, edge_attr
