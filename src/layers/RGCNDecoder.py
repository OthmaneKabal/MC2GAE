from torch import nn
from torch_geometric.nn import RGCNConv

class RGCNDecoder(nn.Module):
    def __init__(self, encoder: nn.Module, data, num_bases=30, alpha=0.01):
        """
        Initialise le décodeur RGCN symétrique basé sur l'encodeur fourni.

        Paramètres:
        - encoder : l'encodeur RGCN utilisé pour obtenir les embeddings,
                    à partir duquel nous extrayons les dimensions de chaque couche
        - data : objet Data pour accéder aux caractéristiques d'origine (data.x) et relations
        - num_bases : nombre de bases à utiliser dans chaque couche RGCN pour réduire le nombre de paramètres
        - alpha : coefficient de Leaky ReLU pour conserver les valeurs négatives
        """
        super(RGCNDecoder, self).__init__()

        # Récupérer les dimensions des couches de l'encodeur pour les inverser dans le décodeur
        encoder_out_channels = [layer.out_channels for layer in encoder.convs]
        encoder_in_channels = encoder.convs[0].in_channels  # Dimension initiale des caractéristiques des nœuds

        # Inverser les dimensions de l'encodeur pour le décodeur
        decoder_out_channels = list(reversed(encoder_out_channels)) + [encoder_in_channels]

        # Créer les couches RGCN du décodeur avec les dimensions inversées
        self.convs = nn.ModuleList()
        for i in range(len(decoder_out_channels) - 1):
            input_dim = decoder_out_channels[i]
            output_dim = decoder_out_channels[i + 1]
            # Ajouter le nombre de bases spécifié dans chaque couche RGCN
            self.convs.append(RGCNConv(input_dim, output_dim, data.edge_type.max().item() + 1, num_bases=num_bases))

        # Instancier Leaky ReLU avec le coefficient alpha
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)

    def reset_parameters(self):
        """Réinitialise les paramètres des couches de décodeur."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, embeddings):
        """
        Passe en avant dans le décodeur pour reconstruire les caractéristiques des nœuds.

        Paramètres:
        - data : objet de type Data contenant les caractéristiques d'origine (data.x),
                 edge_index (indices des arêtes) et edge_type (types d'arêtes)
        - embeddings : les embeddings produits par l'encodeur (entrée pour le décodeur)

        Retourne:
        - Reconstruction des caractéristiques des nœuds
        """
        x = embeddings
        edge_index, edge_type = data.edge_index, data.edge_type

        # Appliquer chaque couche RGCN avec une activation ReLU entre chaque couche
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = self.leaky_relu(x)

        return x


