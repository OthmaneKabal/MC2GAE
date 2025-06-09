import sys
import os

from loss_func import recon_r_loss, sce_loss_fnc, similarity_pair_loss, mse_loss_fnc, contrastive_loss, \
    contrastive_loss_exclude_is, calculate_cluster_assignments, inter_cluster_loss, intra_cluster_loss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'utils')))
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch_geometric.loader import NeighborLoader
from evaluate import evaluate
from utils.ConvENegativeSampling import generate_negatives, get_positives
from utils.ConvEDataLoader import create_data_loader
from utils.utils import generate_relation_embeddings_tensor, removed_edges_train_test_split, \
    save_model_with_hyperparams, set_seed, save_model, calculate_metrics
from data_augmentation import relation_based_edge_dropping_balanced
from data_augmentation import view_partial_features_masking
from GraphDataLoader import GraphDataLoader
import torch.nn.functional as F

import pandas as pd
from config import config
from MRGAE import  MRGAE
from tqdm import tqdm
import wandb
import torch
import random
import numpy as np
seed = 42
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
import torch
import copy
set_seed(42)
torch.backends.cudnn.deterministic = True
import torch_geometric.transforms as T


print(5)

def generate_augmented_view(config):
