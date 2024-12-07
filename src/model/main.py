import sys
import os

from train_optimize_parms import train_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'utils')))
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch_geometric.loader import NeighborLoader
from src.model.evaluate import evaluate, evaluate_all
from ConvE import ConvE
from utils.ConvENegativeSampling import generate_negatives, get_positives
from utils.ConvEDataLoader import create_data_loader
from utils.utils import generate_relation_embeddings_tensor, removed_edges_train_test_split, \
    save_model_with_hyperparams, set_seed
from data_augmentation import relation_based_edge_dropping_balanced
from data_augmentation import view_partial_features_masking
from GraphDataLoader import GraphDataLoader
import torch.optim as optim
from RGCNEncoder import RGCNEncoder
from RGCNDecoder import RGCNDecoder
from torch.nn import MSELoss
from torch_geometric.data import Data
from config import config
from GraphDataPreparation import GraphDataPreparation
from MC2GEA import  MC2GEA
from tqdm import tqdm
import wandb
import torch
import random
import numpy as np
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
import torch
import copy
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
wandb.require("legacy-service")
wandb.login(key="c278e62d2025b60ff8b984a40f7b62b697f9b4fd", relogin=True)



wandb_project_name = config["wandb_project_name"]

save_dir = config["save_dir"]
Entities_path = config["Entities_path"]
KG_path = config["KG_path"]
Edges_path = config["Edges_path"]
device = config["device"]

gdp = GraphDataPreparation(Entities_path, KG_path, edges_embd_path=Edges_path, is_directed=True)
data = gdp.prepare_graph_with_type()
data = Data(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type).to(device)


for num_bases in config["hyperparams_grid"]["num_bases"]:
    for out_channels in config["hyperparams_grid"]["out_channels"]:
        config["convE_config"]["embedding_dim"] = out_channels[1]
        config["convE_config"]["hidden_size"] = config["coresp_hidden_sizes"][out_channels[1]]

        # Nom unique pour chaque run, incluant les hyperparamètres
        run_name = f"bases_{num_bases}_channels_{'-'.join(map(str, out_channels))}"
        # Initialisation de wandb pour chaque combinaison d'hyperparamètres
        wandb.init(
            project=wandb_project_name,
            name=run_name,
            config= config,
            settings=wandb.Settings(start_method="thread")
        )
        print(f"\nTraining with num_bases={num_bases} and out_channels={out_channels}...\n")
        # Initialiser le modèle avec les hyperparamètres actuels
        RGCN_encoder = RGCNEncoder(data, out_channels, config["num_layers"], num_bases).to(device)
        RGCN_decoder = RGCNDecoder(RGCN_encoder, data, num_bases, config["alpha"]).to(device)
        r_decoder = ConvE(config["convE_config"])
        autoencoder = MC2GEA(RGCN_encoder, RGCN_decoder, r_decoder = r_decoder).to(device)
        optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
        train_model(autoencoder, data, optimizer, config["num_epochs"], num_bases, out_channels, gdp, training_options= config["training_options"],save_dir=config["save_dir"], wandb = wandb)
        wandb.finish()