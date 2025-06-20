import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'utils')))
from src.layers.TransGCNEncoder import TransGCNEncoder


class GNNClassifier(nn.Module):
    def __init__(self, base_encoder: nn.Module, mlp_layers_sizes: list, num_classes: int):
        """
        :param base_encoder: modèle GNN (doit retourner des embeddings de nœuds)
        :param mlp_layers_sizes: liste des tailles des couches du MLP, [0] pour désactiver le MLP
        :param num_classes: nombre de classes de sortie
        """
        super().__init__()
        self.encoder = base_encoder
        self.use_mlp = not (len(mlp_layers_sizes) == 1 and mlp_layers_sizes[0] == 0)

        if self.use_mlp:
            layers = []
            input_dim = self.encoder.out_channels  # doit être défini dans l’encodeur
            for hidden_dim in mlp_layers_sizes:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, num_classes))
            self.classifier = nn.Sequential(*layers)
        else:
            self.classifier = nn.Linear(self.encoder.out_channels, num_classes)

    def forward(self, data):
        x = self.encode(data)
        return self.classifier(x)

    def encode(self, data):
        if isinstance(self.encoder, TransGCNEncoder):
            x, _ = self.encoder(data)
        else:
            x = self.encoder(data)
        return x
