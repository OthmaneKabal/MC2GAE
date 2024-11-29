from torch.utils.data import Dataset, DataLoader
import torch

class TripletDataset(Dataset):
    """
    Custom dataset for positive and negative triplets to train the ConvE model.
    """

    def __init__(self, positive_triplets, negative_triplets, entity_embeddings, relation_embeddings):
        """
        Args:
            positive_triplets (torch.Tensor): Tensor of positive triplets (num_triplets, 3).
            negative_triplets (torch.Tensor): Tensor of negative triplets (num_triplets, 3).
            entity_embeddings (torch.Tensor): Embedding matrix for entities (num_entities, emb_dim).
            relation_embeddings (torch.Tensor): Embedding matrix for relations (num_relations, emb_dim).
        """
        self.positive_triplets = positive_triplets
        self.negative_triplets = negative_triplets
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.triplets = torch.cat([positive_triplets, negative_triplets], dim=0)
        self.labels = torch.cat([torch.ones(len(positive_triplets)), torch.zeros(len(negative_triplets))], dim=0)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        label = self.labels[idx]

        head_emb = self.entity_embeddings[triplet[0]]
        rel_emb = self.relation_embeddings[triplet[1]]
        tail_emb = self.entity_embeddings[triplet[2]]

        return head_emb, rel_emb, tail_emb, label


def create_data_loader(positive_triplets, negative_triplets, entity_embeddings, relation_embeddings, batch_size,
                       shuffle=True):
    """
    Creates a DataLoader for the ConvE model.

    Args:
        positive_triplets (torch.Tensor): Tensor of positive triplets (num_triplets, 3).
        negative_triplets (torch.Tensor): Tensor of negative triplets (num_triplets, 3).
        entity_embeddings (torch.Tensor): Embedding matrix for entities (num_entities, emb_dim).
        relation_embeddings (torch.Tensor): Embedding matrix for relations (num_relations, emb_dim).
        batch_size (int): Batch size for the data loader.

    Returns:
        DataLoader: DataLoader for training the ConvE model.
    """
    dataset = TripletDataset(positive_triplets, negative_triplets, entity_embeddings, relation_embeddings)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader