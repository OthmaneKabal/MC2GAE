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
from src.model.utils.utils import set_seed, load_model_from_checkpoint
from src.layers.GCNDecoder import GCNDecoder
from src.layers.GCNEncoder import GCNEncoder
from src.layers.MLPDecoder import MLPDecoder
import torch.optim as optim
from src.layers.RGCNEncoder import RGCNEncoder
from src.layers.RGCNDecoder import RGCNDecoder
from TransGCNEncoder import TransGCNEncoder
from TransGCNDecoder import TransGCNDecoder
import pandas as pd
from config import classifier_config
from tqdm import tqdm
import wandb
import torch
import random
import numpy as np
from GraphDataPreparation import GraphDataPreparation
from src.model.gnn_classifier.classifier_utils import instantiate_encoder
set_seed(classifier_config['seed'])

from src.model.gnn_classifier.data_cache  import get_data_and_loaders



def main(custom_config):
    print(custom_config)
    set_seed(custom_config['seed'])
    gdp = GraphDataPreparation(custom_config['entities_path'],
                               custom_config['kg_path'],
                               custom_config["edges_path"],
                               is_directed=True)



    annotated_graph, train_loader, test_loader = get_data_and_loaders(custom_config)


    model, model_info, checkpoint = load_model_from_checkpoint(custom_config['model_path'], annotated_graph)

    encoder = model.encoder
    model = GNNClassifier(encoder, custom_config["MLP_layers"], 8).to(custom_config['device'])



    best_results = training_loop_minibatch(model, train_loader, test_loader, custom_config, custom_config['epochs'])
    print(best_results)
if __name__ == '__main__':
    main(classifier_config)