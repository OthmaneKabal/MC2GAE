import torch
from torch_geometric.loader import NeighborLoader
import numpy as np
import random
class GraphDataLoader:
    def __init__(self, data, num_neighbors=[100, 100], batch_size=256, shuffle=True, seed = 42):
        """
        Initialise le DataLoader pour un graphe avec NeighborLoader.

        Args:
            data (Data): L'objet Data de torch_geometric.
            num_neighbors (list): Nombre de voisins à échantillonner à chaque saut.
            batch_size (int): Taille de batch.
            shuffle (bool): Si True, mélange les nœuds pour obtenir des batches différents à chaque époque.
        """
        self.data = data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def get_loader(self):
        """
        Retourne le NeighborLoader configuré pour l'échantillonnage.

        Returns:
            NeighborLoader: DataLoader pour l'entraînement avec l'échantillonnage de voisins.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        loader = NeighborLoader(
        self.data,
        num_neighbors=self.num_neighbors,
        batch_size=self.batch_size,
        input_nodes=torch.arange(self.data.x.size(0), dtype=torch.long),  # Entiers pour input_nodes
        shuffle=self.shuffle
        )
        return loader
