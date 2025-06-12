import torch
from torch import nn


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, relation_embedding_dim, relations_embedding="random", initi_embd = None):
        super(DistMultDecoder, self).__init__()
        if relations_embedding == "random":
            self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, relation_embedding_dim))
            nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))
        elif relations_embedding == "bert" and initi_embd != None:
            self.relation_embedding = initi_embd

    def forward(self, z, edge_index, edge_type):
        s = z[edge_index[0, :]]
        r = self.relation_embedding[edge_type]
        o = z[edge_index[1, :]]
        score = torch.sum(s * r * o, dim=1)
        return score

