# data_cache.py

from src.model.gnn_classifier.classifier_utils import instantiate_encoder
from src.model.gnn_classifier.GNNClassifier import GNNClassifier
from data.GraphDataLoader import GraphDataLoader
from torch_geometric.loader import NeighborLoader
from GraphDataPreparation import GraphDataPreparation
import torch

_cached = {}

import copy

def get_data_and_loaders(config):
    if "annotated_graph" in _cached:
        # return a deepcopy to isolate mutations
        return copy.deepcopy(_cached["annotated_graph"]), _cached["train_loader"], _cached["test_loader"]

    print("📦 Initializing data and loaders once...")

    gdp = GraphDataPreparation(config['entities_path'],
                               config['kg_path'],
                               config["edges_path"],
                               is_directed=True)

    data = gdp.prepare_graph_with_type()
    annotated_graph = gdp.annotate_with_labels(data,
                                               config["train_set_path"],
                                               config["GS"]).to(config['device'])

    train_loader = NeighborLoader(
        annotated_graph,
        input_nodes=annotated_graph.train_mask,
        num_neighbors=config['num_neighbors'],
        batch_size=config['train_batch_size'],
        shuffle=config['shuffle'],
    )

    test_loader = NeighborLoader(
        annotated_graph,
        input_nodes=annotated_graph.test_mask,
        num_neighbors=config['num_neighbors'],
        batch_size=config['test_batch_size'],
    )

    # Store only unmodified graph (before being touched)
    _cached["annotated_graph"] = annotated_graph.cpu()  # store on CPU to reduce GPU memory
    _cached["train_loader"] = train_loader
    _cached["test_loader"] = test_loader

    return copy.deepcopy(_cached["annotated_graph"].to(config['device'])), train_loader, test_loader
