import torch
from torch import nn


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, relation_embedding_dim):
        super(DistMultDecoder, self).__init__()
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, relation_embedding_dim))
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

    def forward(self, z, edge_index, edge_type):
        s = z[edge_index[0, :]]
        r = self.relation_embedding[edge_type]
        o = z[edge_index[1, :]]
        score = torch.sum(s * r * o, dim=1)
        return score

