import sys
import os

from src.model.gnn_classifier.GNNClassifier import GNNClassifier
from torch_geometric.loader import NeighborLoader

from src.model.gnn_classifier.train_classifier import training_loop_minibatch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'utils')))

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch_geometric.loader import NeighborLoader
from data.GraphDataLoader import GraphDataLoader
import torch.nn.functional as F
from src.layers.GATEncoder import GATEncoder
from src.model.utils.utils import  set_seed
from src.layers.GCNDecoder import GCNDecoder
from src.layers.GCNEncoder import GCNEncoder
from src.layers.MLPDecoder import MLPDecoder
import torch.optim as optim
from src.layers.RGCNEncoder import RGCNEncoder
from src.layers.RGCNDecoder import RGCNDecoder
from TransGCNEncoder import TransGCNEncoder
from TransGCNDecoder import TransGCNDecoder
import pandas as pd
from classifier_config import classifier_config
from tqdm import tqdm
import wandb
import torch
import random
import numpy as np
from GraphDataPreparation import GraphDataPreparation
from classifier_utils import instantiate_encoder
set_seed(classifier_config['seed'])
from data_cache import get_data_and_loaders



# def main():
#     print(classifier_config)
#     gdp = GraphDataPreparation(classifier_config['entities_path'],
#                                classifier_config['kg_path'],
#                                classifier_config["edges_path"],
#                                is_directed=True)
#     data = gdp.prepare_graph_with_type()
#     annotated_graph = gdp.annotate_with_labels(data,
#                                 classifier_config["train_set_path"],
#                                 classifier_config["GS"]).to(classifier_config['device'])
#     print(annotated_graph)
#     encoder = instantiate_encoder(classifier_config, annotated_graph)
#     model = GNNClassifier(encoder, classifier_config["MLP_layers"], 8).to(classifier_config['device'])
#     train_loader = NeighborLoader(
#         annotated_graph,
#         input_nodes=annotated_graph.train_mask,
#         num_neighbors=classifier_config['num_neighbors'],
#         batch_size=classifier_config['train_batch_size'],
#         shuffle=classifier_config['shuffle'],
#     )
#
#     test_loader = NeighborLoader(
#         annotated_graph,
#         input_nodes=annotated_graph.test_mask,
#         num_neighbors=classifier_config['num_neighbors'],
#         batch_size=classifier_config['test_batch_size'],
#     )
#     training_loop_minibatch(model, train_loader, test_loader, classifier_config, classifier_config['epochs'])

def main(custom_config):
    print(custom_config)
    set_seed(classifier_config['seed'])
    gdp = GraphDataPreparation(custom_config['entities_path'],
                               custom_config['kg_path'],
                               custom_config["edges_path"],
                               is_directed=True)

    # data = gdp.prepare_graph_with_type()
    # annotated_graph = gdp.annotate_with_labels(
    #     data,
    #     custom_config["train_set_path"],
    #     custom_config["GS"]
    # ).to(custom_config['device'])
    #
    # train_loader = NeighborLoader(
    #     annotated_graph,
    #     input_nodes=annotated_graph.train_mask,
    #     num_neighbors=custom_config['num_neighbors'],
    #     batch_size=custom_config['train_batch_size'],
    #     shuffle=custom_config['shuffle'],
    # )
    #
    # test_loader = NeighborLoader(
    #     annotated_graph,
    #     input_nodes=annotated_graph.test_mask,
    #     num_neighbors=custom_config['num_neighbors'],
    #     batch_size=custom_config['test_batch_size'],
    # )

    annotated_graph, train_loader, test_loader = get_data_and_loaders(custom_config)


    encoder = instantiate_encoder(custom_config, annotated_graph)
    model = GNNClassifier(encoder, custom_config["MLP_layers"], 8).to(custom_config['device'])



    training_loop_minibatch(model, train_loader, test_loader, custom_config, custom_config['epochs'])

if __name__ == '__main__':
    from classifier_config import classifier_config
    main(classifier_config)