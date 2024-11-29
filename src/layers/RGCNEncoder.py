from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv

class RGCNEncoder(nn.Module):
    def __init__(self, data: Data, out_channels, num_layers=2, num_bases=30):
        """
        Initialise l'encodeur RGCN avec ReLU entre chaque couche et un nombre de bases spécifié.

        Paramètres:
        - data : objet de type Data pour extraire les caractéristiques d'entrée et les relations
        - out_channels : liste contenant le nombre de caractéristiques de sortie pour chaque couche
        - num_layers : nombre de couches RGCN à empiler (doit correspondre à la longueur de out_channels)
        - num_bases : nombre de bases à utiliser dans chaque couche RGCN pour réduire le nombre de paramètres
        """
        super(RGCNEncoder, self).__init__()

        # Assurer que le nombre de couches correspond à la taille de out_channels
        assert len(out_channels) == num_layers, "La longueur de out_channels doit être égale à num_layers"
        self.out_channels = out_channels[1]
        # Extraire les dimensions d'entrée et le nombre de relations depuis l'objet data
        in_channels = data.x.shape[1]
        num_relations = data.edge_type.max().item() + 1

        # Créer une liste de couches RGCN avec des tailles de sortie différentes
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            input_dim = in_channels if i == 0 else out_channels[i - 1]
            # Ajouter le nombre de bases spécifié dans chaque couche RGCN
            self.convs.append(RGCNConv(input_dim, out_channels[i], num_relations, num_bases=num_bases))
        # Instancier ReLU pour l'activation
        self.relu = nn.ReLU()

    def reset_parameters(self):
        """Réinitialise les paramètres des couches de l'encodeur."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data: Data):
        """
        Passe en avant dans le réseau en prenant un objet Data comme entrée.

        Paramètres:
        - data : objet de type Data contenant x (caractéristiques des nœuds),
                 edge_index (indices des arêtes) et edge_type (types d'arêtes)

        Retourne:
        - Embeddings des nœuds après passage dans l'encodeur RGCN
        """
        # Extraire les attributs de l'objet Data
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        # Appliquer chaque couche RGCN avec une activation ReLU entre chaque couche
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = self.relu(x)

        return x
