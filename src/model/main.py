import sys
import os

import pandas as pd
from torch_geometric.nn import GAE
from win32comext.shell.demos.servers.folder_view import tasks

from src.layers.GCNDecoder import GCNDecoder
from src.layers.GCNEncoder import GCNEncoder
from src.layers.MLPDecoder import MLPDecoder
from src.model.train_optimize_parms import train_GAE, train_X_reconstruction
from train_optimize_parms import train_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'utils')))
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch_geometric.loader import NeighborLoader
from src.model.evaluate import evaluate, evaluate_all
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
from MRGAE import  MRGAE
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


def main():
    results = []
    wandb_project_name = config["wandb_project_name"]
    # save_dir = config["save_dir"]
    Entities_path = config["Entities_path"]
    KG_path = config["KG_path"]
    Edges_path = config["Edges_path"]
    device = config["device"]
    gdp = GraphDataPreparation(Entities_path, KG_path, edges_embd_path=Edges_path, is_directed=True)
    data = gdp.prepare_graph_with_type()
    data = Data(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type).to(device)

    for task in config["training_task"]:
        save_dir = config["root_save_dir"] + f"/{task}"

        msg_sens = config["message_sens"][0]

        for out_channels in config["hyperparams_grid"]["out_channels"]:
            for encoder_ in config["encoders"]:
                for decoder_ in config["decoders"]:
                    use_num_bases = (encoder_ == "RGCN") or (decoder_ == "RGCN")
                    if use_num_bases:
                        for num_bases in config["hyperparams_grid"]["num_bases"]:
                            if encoder_ == "GCN":
                                encoder = GCNEncoder(data, out_channels, config["num_layers"],
                                                     message_sens=msg_sens).to(device)
                            elif encoder_ == "RGCN":
                                encoder = RGCNEncoder(data, out_channels, config["num_layers"], num_bases,
                                                      message_sens=msg_sens).to(device)

                            if decoder_ == "GCN":
                                decoder = GCNDecoder(encoder, data, config["alpha"], message_sens=msg_sens).to(device)
                            elif decoder_ == "RGCN":
                                decoder = RGCNDecoder(encoder, data, num_bases, config["alpha"],
                                                      message_sens=msg_sens).to(device)
                            elif decoder_ == "MLP":
                                decoder = MLPDecoder(encoder, data, config["alpha"]).to(device)


                            run_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_dec-{decoder_}"
                            file_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_dec-{decoder_}"
                            run_config = {
                                "device": config["device"],
                                "num_layers": 2,
                                "alpha": config["alpha"],
                                "max_masking_percentage": config["max_masking_percentage"],
                                "total_drop_rate": config["total_drop_rate"],
                                "learning_rate": config["learning_rate"],
                                "batch_size": config["batch_size"],
                                "num_neighbors": [500, 500],
                                "num_epochs": 100,
                                "bases": num_bases,
                                "out_channels": out_channels,
                                "training_task": config["training_task"],
                                "encoders": encoder_,
                                "decoders": decoder_,
                                "message_sens": msg_sens
                            }

                            wandb.init(
                                project=config["wandb_project_name"],
                                name=run_name,
                                config=run_config,
                                settings=wandb.Settings(start_method="thread")
                            )

                            autoencoder = MRGAE(encoder, decoder, projections=config["projections"]).to(device)
                            optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                            performances = train_X_reconstruction(autoencoder, data, optimizer, config["num_epochs"], num_bases, out_channels,
                                        gdp, file_name,device, loss_fct=["MSE"], save_dir = save_dir,
                                        wandb=wandb)
                            results.append(performances)
                            # df = pd.DataFrame(data)
                            #
                            # # Sauvegarde en fichier Excel
                            # df.to_excel("results.xlsx", index=False)

                            wandb.finish()



                    else:  # Si num_bases n'est pas utilisé
                        encoder = GCNEncoder(data, out_channels, config["num_layers"], message_sens=msg_sens).to(device)

                        if decoder_ == "GCN":
                            decoder = GCNDecoder(encoder, data, config["alpha"], message_sens=msg_sens).to(device)
                        elif decoder_ == "MLP":
                            decoder = MLPDecoder(encoder, data, config["alpha"]).to(device)
                        else:
                            print("Error: RGCN decoder requires num_bases but is not defined!")
                            exit(-1)

                        run_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_dec-{decoder_}"
                        file_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_dec-{decoder_}"
                        run_config = {
                            "device": config["device"],
                            "num_layers": 2,
                            "alpha": config["alpha"],
                            "max_masking_percentage": config["max_masking_percentage"],
                            "total_drop_rate": config["total_drop_rate"],
                            "learning_rate": config["learning_rate"],
                            "batch_size": config["batch_size"],
                            "num_neighbors": [500, 500],
                            "num_epochs": 100,
                            "out_channels": out_channels,
                            "training_task": config["training_task"],
                            "encoders": encoder_,
                            "decoders": decoder_,
                            "message_sens": msg_sens
                        }

                        wandb.init(
                            project=config["wandb_project_name"],
                            name=run_name,
                            config=run_config,
                            settings=wandb.Settings(start_method="thread")
                        )

                        autoencoder = MRGAE(encoder, decoder, projections=config["projections"]).to(device)
                        optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                        performances = train_X_reconstruction(autoencoder, data, optimizer, config["num_epochs"], 0, out_channels,
                                    gdp, file_name, device, save_dir=save_dir, loss_fct=["MSE"],
                                    wandb=wandb)
                        results.append(performances)

                        wandb.finish()
    df = pd.DataFrame(results)

    # Sauvegarde en fichier Excel
    df.to_excel("results.xlsx", index=False)
#
# def main():
#     print(44)

if __name__ == "__main__":
    main()