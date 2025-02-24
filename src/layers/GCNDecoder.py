

from torch import nn
from torch_geometric.nn import GCNConv


class GCNDecoder(nn.Module):
    def __init__(self, encoder: nn.Module, data, alpha=0.01, dropout=0.5, message_sens="source_to_target"):
        """
        Initialize the symmetric GCN decoder based on the given encoder.

        Parameters:
        - encoder: The GCN encoder used to obtain embeddings,
                   from which we extract the layer dimensions.
        - data: Data object to access the original features (data.x).
        - alpha: Leaky ReLU coefficient to preserve negative values.
        - dropout: Dropout probability for regularization.
        """
        super(GCNDecoder, self).__init__()

        # Get dimensions of the encoder layers to reverse them for the decoder
        encoder_out_channels = [layer.out_channels for layer in encoder.convs]
        encoder_in_channels = encoder.convs[0].in_channels  # Initial node feature dimension

        # Reverse encoder dimensions for the decoder
        decoder_out_channels = list(reversed(encoder_out_channels)) + [encoder_in_channels]

        # Create GCN layers for the decoder with reversed dimensions
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()  # Batch normalization layers
        for i in range(len(decoder_out_channels) - 1):
            input_dim = decoder_out_channels[i]
            output_dim = decoder_out_channels[i + 1]
            # Add a GCN layer
            self.convs.append(GCNConv(input_dim, output_dim, flow=message_sens))
            # Add a batch normalization layer
            self.bns.append(nn.BatchNorm1d(output_dim))

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Final linear layer to reconstruct node features
        self.final_layer = nn.Linear(decoder_out_channels[-1], data.num_features)

        # Activation function
        self.relu = nn.ReLU()

        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
    def reset_parameters(self):
        """Reset the parameters of the decoder layers."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_layer.reset_parameters()

    def forward(self, data, embeddings):
        """
        Forward pass in the decoder to reconstruct node features.

        Parameters:
        - data: Data object containing original features (data.x) and edge_index (edge indices).
        - embeddings: Embeddings produced by the encoder (input to the decoder).

        Returns:
        - Reconstructed node features.
        """
        x = embeddings
        edge_index = data.edge_index

        # Apply each GCN layer with BatchNorm, ReLU, and Dropout in between
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)

        # Apply the final linear layer
        x = self.final_layer(x)

        return x


# from torch import nn
# from torch_geometric.nn import GCNConv
#
# class GCNDecoder(nn.Module):
#     def __init__(self, encoder: nn.Module, data, alpha=0.01):
#         """
#         Initialise le décodeur GCN symétrique basé sur l'encodeur fourni.
#
#         Paramètres:
#         - encoder : l'encodeur GCN utilisé pour obtenir les embeddings,
#                     à partir duquel nous extrayons les dimensions de chaque couche
#         - data : objet Data pour accéder aux caractéristiques d'origine (data.x)
#         - alpha : coefficient de Leaky ReLU pour conserver les valeurs négatives
#         """
#         super(GCNDecoder, self).__init__()
#
#         # Récupérer les dimensions des couches de l'encodeur pour les inverser dans le décodeur
#         encoder_out_channels = [layer.out_channels for layer in encoder.convs]
#         encoder_in_channels = encoder.convs[0].in_channels  # Dimension initiale des caractéristiques des nœuds
#
#         # Inverser les dimensions de l'encodeur pour le décodeur
#         decoder_out_channels = list(reversed(encoder_out_channels)) + [encoder_in_channels]
#
#         # Créer les couches GCN du décodeur avec les dimensions inversées
#         self.convs = nn.ModuleList()
#         for i in range(len(decoder_out_channels) - 1):
#             input_dim = decoder_out_channels[i]
#             output_dim = decoder_out_channels[i + 1]
#             # Ajouter une couche GCN
#             self.convs.append(GCNConv(input_dim, output_dim))
#
#         # Instancier Leaky ReLU avec le coefficient alpha
#         self.relu = nn.ReLU()
#
#     def reset_parameters(self):
#         """Réinitialise les paramètres des couches de décodeur."""
#         for conv in self.convs:
#             conv.reset_parameters()
#
#     def forward(self, data, embeddings):
#         """
#         Passe en avant dans le décodeur pour reconstruire les caractéristiques des nœuds.
#
#         Paramètres:
#         - data : objet de type Data contenant les caractéristiques d'origine (data.x) et edge_index (indices des arêtes)
#         - embeddings : les embeddings produits par l'encodeur (entrée pour le décodeur)
#
#         Retourne:
#         - Reconstruction des caractéristiques des nœuds
#         """
#         x = embeddings
#         edge_index = data.edge_index
#
#         # Appliquer chaque couche GCN avec une activation ReLU entre chaque couche
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x = self.self.relu(x)
#
#         return x


