import torch
from torch_geometric.data import Data
import random


def create_triplet_lookup(data):
    triplet_set = set()
    for i in range(data.edge_index.size(1)):
        src = data.edge_index[0, i].item()
        dest = data.edge_index[1, i].item()
        relation = data.edge_type[i].item()
        triplet_set.add((src, relation, dest))
    return triplet_set


def is_triplet_in_data(triplet_set, triplet):
    return triplet in triplet_set

def generate_negatives(data, batch, negative_ratio=1, relation_weight=None):
    """
    Generate negative triplets dynamically for a given batch of triplets with validation
    and resampling for existing triplets in the graph.
    """
    edge_index = batch.edge_index  # (2, num_edges)
    edge_type = batch.edge_type    # (num_edges,)
    num_nodes = batch.num_nodes
    data_edge_type = data.edge_type
    num_relations = batch.num_relations if hasattr(batch, "num_relations") else torch.max(data_edge_type).item() + 1

    positives = []
    negatives = []

    # Precompute triplet lookup
    triplet_set = create_triplet_lookup(data)

    # Collect all positive triplets
    for i in range(edge_index.size(1)):
        h, t = edge_index[:, i]
        r = edge_type[i]
        positives.append((h.item(), r.item(), t.item()))

    # Generate negatives
    for h, r, t in positives:
        for _ in range(negative_ratio):
            while True:  # Keep sampling until a valid negative is found
                if random.random() < 0.33:
                    # Corrupt head
                    h_neg = random.randint(0, num_nodes - 1)
                    global_h_neg = batch.n_id[h_neg] if hasattr(batch, 'n_id') else h_neg
                    if not is_triplet_in_data(triplet_set, (global_h_neg, r, batch.n_id[t])):
                        negatives.append((h_neg, r, t))
                        break  # Valid negative found, exit loop
                elif random.random() < 0.66:
                    # Corrupt tail
                    t_neg = random.randint(0, num_nodes - 1)
                    global_t_neg = batch.n_id[t_neg] if hasattr(batch, 'n_id') else t_neg
                    if not is_triplet_in_data(triplet_set, (batch.n_id[h], r, global_t_neg)):
                        negatives.append((h, r, t_neg))
                        break  # Valid negative found, exit loop
                else:
                    # Corrupt relation
                    if relation_weight:
                        r_neg = random.choices(
                            list(relation_weight.keys()),
                            weights=list(relation_weight.values()),
                            k=1
                        )[0]
                    else:
                        r_neg = random.randint(0, num_relations - 1)
                    if not is_triplet_in_data(triplet_set, (batch.n_id[h], r_neg, batch.n_id[t])):
                        negatives.append((h, r_neg, t))
                        break  # Valid negative found, exit loop

    # Convert negatives to tensor
    negative_tensor = torch.tensor(negatives, dtype=torch.long)
    return negative_tensor

def get_positives(batch):
    """
    Generate a tensor of all positive triplets (head, relation, tail) from the given batch.
    """
    edge_index = batch.edge_index  # (2, num_edges)
    edge_type = batch.edge_type    # (num_edges,)

    positives = []

    # Collect all positive triplets
    for i in range(edge_index.size(1)):
        h, t = edge_index[:, i]
        r = edge_type[i]
        positives.append((h.item(), r.item(), t.item()))

    # Convert positives to tensor
    positive_tensor = torch.tensor(positives, dtype=torch.long)
    return positive_tensor