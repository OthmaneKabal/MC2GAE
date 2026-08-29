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


def _rng_random(rng):
    return rng.random() if rng is not None else random.random()


def _rng_randint(rng, low, high):
    return rng.randint(low, high) if rng is not None else random.randint(low, high)


def _rng_choices(rng, population, weights):
    if rng is not None:
        return rng.choices(population, weights=weights, k=1)[0]
    return random.choices(population, weights=weights, k=1)[0]


def _sample_entity_candidate(candidate_node_ids, node_position_by_global_id, rng=None, max_attempts=100):
    num_candidates = len(candidate_node_ids)
    for _ in range(max_attempts):
        candidate_global = int(candidate_node_ids[_rng_randint(rng, 0, num_candidates - 1)])
        candidate_local = node_position_by_global_id.get(candidate_global)
        if candidate_local is not None:
            return candidate_local
    return None


def _sample_soft_type_candidate(candidates_by_relation, relation, node_ids, node_position_by_global_id,
                                triplet_set, fixed_node_global, corrupt_head=True, max_attempts=30, rng=None):
    if candidates_by_relation is None or relation not in candidates_by_relation:
        return None

    candidates = candidates_by_relation[relation]
    if candidates is None or len(candidates) == 0:
        return None

    num_candidates = len(candidates)
    for _ in range(max_attempts):
        candidate_global = int(candidates[_rng_randint(rng, 0, num_candidates - 1)])
        candidate_local = node_position_by_global_id.get(candidate_global)
        if candidate_local is None:
            continue
        if corrupt_head:
            triplet = (candidate_global, relation, fixed_node_global)
        else:
            triplet = (fixed_node_global, relation, candidate_global)
        if not is_triplet_in_data(triplet_set, triplet):
            return candidate_local
    return None


def generate_negatives(data, batch, negative_ratio=1, relation_weight=None, triplet_set=None,
                       negative_sampling_mode="uniform", soft_type_candidates=None,
                       soft_type_negative_ratio=0.7, negative_corruption_mode="mixed",
                       negative_entity_sampling_scope="batch", rng=None):

    """
    Generate negative triplets dynamically for a given batch of triplets with validation
    and resampling for existing triplets in the graph.
    """
    
    edge_index = batch.edge_index  # (2, num_edges)
    edge_type = batch.edge_type    # (num_edges,)
    num_nodes = batch.num_nodes
    data_edge_type = data.edge_type
    num_relations = batch.num_relations if hasattr(batch, "num_relations") else torch.max(data_edge_type).item() + 1
    node_ids = batch.n_id if hasattr(batch, "n_id") else torch.arange(num_nodes, device=edge_index.device)
    node_position_by_global_id = {int(global_id): idx for idx, global_id in enumerate(node_ids.detach().cpu().tolist())}
    if negative_entity_sampling_scope not in ("batch", "global"):
        raise ValueError("negative_entity_sampling_scope must be one of: batch, global")
    if negative_entity_sampling_scope == "global":
        candidate_node_ids = torch.arange(data.num_nodes, device=edge_index.device)
    else:
        candidate_node_ids = node_ids
    use_soft_type_sampling = (
        negative_sampling_mode == "soft_type_aware" and
        soft_type_candidates is not None and
        soft_type_negative_ratio > 0
    )
    if negative_corruption_mode not in ("mixed", "relation_only", "entity_only"):
        raise ValueError(
            "negative_corruption_mode must be one of: mixed, relation_only, entity_only"
        )

    positives = []
    negatives = []

    if triplet_set is None:
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
                if negative_corruption_mode == "relation_only":
                    corruption = "relation"
                elif negative_corruption_mode == "entity_only":
                    corruption = "head" if _rng_random(rng) < 0.5 else "tail"
                elif _rng_random(rng) < 0.33:
                    corruption = "head"
                elif _rng_random(rng) < 0.66:
                    corruption = "tail"
                else:
                    corruption = "relation"

                if corruption == "head":
                    # Corrupt head
                    global_t = node_ids[t].item()
                    h_neg = None
                    if use_soft_type_sampling and _rng_random(rng) < soft_type_negative_ratio:
                        h_neg = _sample_soft_type_candidate(
                            soft_type_candidates.get("domain"),
                            r,
                            node_ids,
                            node_position_by_global_id,
                            triplet_set,
                            global_t,
                            corrupt_head=True,
                            rng=rng,
                        )
                    if h_neg is None:
                        h_neg = _sample_entity_candidate(candidate_node_ids, node_position_by_global_id, rng=rng)
                        if h_neg is None:
                            raise RuntimeError(
                                "Could not sample a negative head from the selected entity scope. "
                                "Use full-graph training for negative_entity_sampling_scope='global'."
                            )
                    global_h_neg = node_ids[h_neg].item()
                    if not is_triplet_in_data(triplet_set, (global_h_neg, r, global_t)):
                        negatives.append((h_neg, r, t))
                        break  # Valid negative found, exit loop
                elif corruption == "tail":
                    # Corrupt tail
                    global_h = node_ids[h].item()
                    t_neg = None
                    if use_soft_type_sampling and _rng_random(rng) < soft_type_negative_ratio:
                        t_neg = _sample_soft_type_candidate(
                            soft_type_candidates.get("range"),
                            r,
                            node_ids,
                            node_position_by_global_id,
                            triplet_set,
                            global_h,
                            corrupt_head=False,
                            rng=rng,
                        )
                    if t_neg is None:
                        t_neg = _sample_entity_candidate(candidate_node_ids, node_position_by_global_id, rng=rng)
                        if t_neg is None:
                            raise RuntimeError(
                                "Could not sample a negative tail from the selected entity scope. "
                                "Use full-graph training for negative_entity_sampling_scope='global'."
                            )
                    global_t_neg = node_ids[t_neg].item()
                    if not is_triplet_in_data(triplet_set, (global_h, r, global_t_neg)):
                        negatives.append((h, r, t_neg))
                        break  # Valid negative found, exit loop
                else:
                    # Corrupt relation
                    if relation_weight:
                        r_neg = _rng_choices(
                            rng,
                            list(relation_weight.keys()),
                            weights=list(relation_weight.values())
                        )
                    else:
                        r_neg = _rng_randint(rng, 0, num_relations - 1)
                    global_h = node_ids[h].item()
                    global_t = node_ids[t].item()
                    if not is_triplet_in_data(triplet_set, (global_h, r_neg, global_t)):
                        negatives.append((h, r_neg, t))
                        break  # Valid negative found, exit loop

    # Convert negatives to tensor
    negative_tensor = torch.tensor(negatives, dtype=torch.long, device=edge_index.device)
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
    positive_tensor = torch.tensor(positives, dtype=torch.long, device=edge_index.device)
    return positive_tensor
