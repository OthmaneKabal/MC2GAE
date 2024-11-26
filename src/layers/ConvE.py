import torch
from torch import nn
import torch.nn.functional as F


class ConvE(nn.Module):

    def __init__(self, config, entities_initial_emb, relations_embeddings):
        super(ConvE, self).__init__()
        self.entities_initial_emb = {k: v.to(config['device']) for k, v in entities_initial_emb.items()}
        self.relations_embeddings = {k: v.to(config['device']) for k, v in relations_embeddings.items()}

        self.inp_drop = nn.Dropout(config['input_drop'])
        self.hidden_drop = nn.Dropout(config['hidden_drop'])
        self.feature_map_drop = nn.Dropout2d(config['feat_drop'])
        self.emb_dim1 = config['embedding_shape1']
        self.emb_dim2 = config['embedding_dim'] // self.emb_dim1

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=config['use_bias'])
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(config['embedding_dim'])
        self.fc = nn.Linear(config['hidden_size'], config['embedding_dim'])
        self.register_parameter('b', nn.Parameter(torch.zeros(config['num_entities'])))
        self.all_entities = torch.stack(list(self.entities_initial_emb.values())).to(config['device'])

    def get_embeddings_from_keys(self, keys, is_entity=True):
        if is_entity:
            return torch.stack([self.entities_initial_emb[key] for key in keys])
        else:
            return torch.stack([self.relations_embeddings[key] for key in keys])

    def forward(self, e1, rel, e2):
        e1_embedded = self.get_embeddings_from_keys(e1, is_entity=True).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.get_embeddings_from_keys(rel, is_entity=False).view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        x = self.bn0(stacked_inputs)
        x = self.inp_drop(x)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = F.relu(self.bn2(x))
        e2_embedded = self.get_embeddings_from_keys(e2, is_entity=True)
        scores = torch.sum(x * e2_embedded, dim=1)
        return torch.sigmoid(scores)