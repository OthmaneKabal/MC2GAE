from torch import nn
from torch_geometric.data import Data
from typing import List
import sys
sys.path.append('/trans_gcn_layer')
from trans_gcn_layer.trans_gnn import TransGNN



class TransGCNDecoder(nn.Module):
    def __init__(self, encoder: nn.Module, data: Data,
                 num_layers=2, alpha=0.01, dropout=0.5,
                 kg_score_fn='TransE',variant='conv', use_edges_info=True, activation='relu', bias=False):
        """
        Décodeur basé sur TransGCN, symétrique de l'encodeur TransGCNEncoder.

        :param encoder: modèle encodeur TransGCNEncoder
        :param data: objet Data contenant les attributs edge_index et edge_attr
        :param num_layers: nombre de couches (doit correspondre à l'encodeur)
        :param alpha: coefficient de la LeakyReLU
        :param dropout: probabilité de dropout
        :param kg_score_fn: fonction de transformation (TransE, etc.)
        :param use_edges_info: utiliser les features des relations ou non
        :param activation: fonction d'activation dans TransGNN
        :param bias: ajouter un biais dans les couches ou non
        """
        super().__init__()

        # Extraire les dimensions de l'encodeur pour les inverser
        encoder_out_channels = [layer.out_channels for layer in encoder.layers]
        encoder_in_channels = encoder.layers[0].in_channels
        decoder_channels = list(reversed(encoder_out_channels)) + [encoder_in_channels]

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)

        for i in range(len(decoder_channels) - 1):
            in_dim = decoder_channels[i]
            out_dim = decoder_channels[i + 1]
            self.layers.append(TransGNN(
                in_channels=in_dim,
                out_channels=out_dim,
                use_edges_info=use_edges_info,
                kg_score_fn=kg_score_fn,
                variant=variant,
                activation=activation,
                bias=bias
            ))

        self.final_layer = nn.Linear(decoder_channels[-1], data.num_features)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.final_layer.reset_parameters()

    def forward(self, data: Data, embeddings, new_edge_attr = None):
        """
        Passe avant du décodeur pour reconstruire les features à partir des embeddings.

        :param data: objet Data (doit contenir edge_index et edge_attr)
        :param embeddings: embeddings des entités issus de l'encodeur
        :return: reconstruction des features d'origine
        """
        x = embeddings
        edge_index = data.edge_index
        if new_edge_attr is None:
            edge_attr = data.edge_attr
        else:
            edge_attr = new_edge_attr
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
            x = self.dropout(x)

        return self.final_layer(x)