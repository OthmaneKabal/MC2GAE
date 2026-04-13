
import sys
import os
import random
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch_geometric.nn import GAE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

# Imports locaux (sans le préfixe src.)
from train_optimize_parms import train_GAE,train_Contrastive , train_X_reconstruction, train_DisMult, train_Double_Reconstruction, train_Contrastive
from Dismult import DistMultDecoder
from GCNDecoder import GCNDecoder
from GCNEncoder import GCNEncoder
from GATDecoder import GATDecoder
from GATEncoder import GATEncoder
from MLPDecoder import MLPDecoder
from TransGCNEncoder import TransGCNEncoder
from TransGCNDecoder import TransGCNDecoder
from RGCNEncoder import RGCNEncoder
from RGCNDecoder import RGCNDecoder
from GraphDataPreparation import GraphDataPreparation
from MRGAE import MRGAE
from config import config
from utils.utils import set_seed
import copy
from data_augmentation import relation_based_edge_dropping_balanced, view_partial_features_masking

# Initialisation des seeds pour reproductibilité
seed = config["seed"]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
set_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialisation wandb
import wandb
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
    print(data)
    data = data.to(device)
    masked_features_data = None
    removed_edge_indices = None
    # data = Data(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type).to(device)
    for task in config["training_task"]:
        save_dir = config["root_save_dir"] + f"/{task}"
        msg_sens = config["message_sens"][0]
        if task == "Recons_X":

            for out_channels in config["hyperparams_grid"]["out_channels"]:
                for encoder_ in config["encoders"]:
                    print(encoder_)
                    for decoder_ in config["decoders"]:
                        if (encoder_ in ["RGCN", "GCN", "GAT"]) and (decoder_ in ["TransGCN_conv", "TransGCN_attn", "RotatEGCN_conv", "RotatEGCN_attn"]):
                            print(f"Skipping invalid combination: enc={encoder_}, dec={decoder_}")
                            continue
                        if (encoder_ in ["RGCN", "GAT"]) and (decoder_ in ["GAT","RGCN"]):
                            print(f"Skipping invalid combination: enc={encoder_}, dec={decoder_}")
                            continue
                        use_num_bases = (encoder_ == "RGCN") or (decoder_ == "RGCN")
                        if use_num_bases:
                            for num_bases in config["hyperparams_grid"]["num_bases"]:
                                if encoder_ == "GCN":
                                    encoder = GCNEncoder(data, out_channels, config["num_layers"],
                                                         message_sens=msg_sens).to(device)
                                elif encoder_ == "RGCN":
                                    encoder = RGCNEncoder(data, out_channels, config["num_layers"], num_bases,
                                                          message_sens=msg_sens).to(device)
                                elif encoder_ == "TransGCN_conv":
                                    encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                              kg_score_fn = 'TransE',variant = 'conv',
                                                              use_edges_info = config["use_edges_info"], activation = 'relu',
                                                              bias = False ).to(device)

                                elif encoder_ == "TransGCN_attn":
                                    encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                              kg_score_fn = 'TransE',variant = 'attn',
                                                              use_edges_info = config["use_edges_info"], activation = 'relu',
                                                              bias = False ).to(device)

                                elif encoder_ == "RotatEGCN_conv":
                                    encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                              kg_score_fn = 'RotatE',variant = 'conv',
                                                              use_edges_info = config["use_edges_info"], activation = 'relu',
                                                              bias = False ).to(device)

                                elif encoder_ == "RotatEGCN_attn":
                                    encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                              kg_score_fn = 'RotatE',variant = 'attn',
                                                              use_edges_info = config["use_edges_info"], activation = 'relu',
                                                              bias = False ).to(device)


                                elif encoder_ == "GAT":
                                    encoder = GATEncoder(data, out_channels, config["num_layers"])
                                    # (self, data: Data, out_channels, num_layers=2, heads=4, dropout=0.5)
                                    # print(encoder)
                                else:
                                    print("invalid encoder type ! ")
                                    raise ValueError("Invalid encoder type!")


                                if decoder_ == "GCN":
                                    decoder = GCNDecoder(encoder, data, config["alpha"], message_sens=msg_sens).to(device)
                                elif decoder_ == "RGCN":
                                    decoder = RGCNDecoder(encoder, data, num_bases, config["alpha"],
                                                          message_sens=msg_sens).to(device)
                                elif decoder_ == "MLP":
                                    decoder = MLPDecoder(encoder, data, config["alpha"]).to(device)

                                elif decoder_ == "TransGCN_conv":
                                    decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3, kg_score_fn = 'TransE',
                                                              variant='conv',
                                                              use_edges_info = config["use_edges_info"]).to(device)

                                elif decoder_ == "TransGCN_attn":
                                    decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                              kg_score_fn='TransE',variant='attn',
                                                              use_edges_info=config["use_edges_info"]).to(device)
                                elif decoder_ == "RotatEGCN_conv":
                                    decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                              kg_score_fn='RotatE',variant='conv',
                                                              use_edges_info=config["use_edges_info"]).to(device)

                                elif decoder_ == "RotatEGCN_attn":
                                    decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                              kg_score_fn='RotatE',variant='attn',
                                                              use_edges_info=config["use_edges_info"]).to(device)


                                elif decoder_ == "GAT":
                                    decoder = GATDecoder(encoder, data, heads=4, alpha=0.01, dropout=0.3)

                                else:
                                    print('invalid decoder !')
                                    raise ValueError("Invalid encoder type!")



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
                                # print(encoder)
                                # print(decoder)
                                # exit(-1)
                                autoencoder = MRGAE(encoder, decoder, projections=config["projections"]).to(device)
                                optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                                local_data = copy.deepcopy(data)
                                performances = train_X_reconstruction(autoencoder, local_data, optimizer, config["num_epochs"],
                                            gdp, file_name,device, config,loss_fct=["MSE"], save_dir = save_dir,
                                            wandb=wandb, seed = config["seed"])
                                results.append(performances)
                                wandb.finish()

                        else:  # Si num_bases n'est pas utilisé
                            if encoder_ == "GCN":
                                encoder = GCNEncoder(data, out_channels, config["num_layers"], message_sens=msg_sens).to(device)

                            elif encoder_ == "TransGCN_conv":
                                encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                          kg_score_fn='TransE', variant='conv',
                                                          use_edges_info=config["use_edges_info"], activation='relu',
                                                          bias=False).to(device)

                            elif encoder_ == "TransGCN_attn":
                                encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                          kg_score_fn='TransE', variant='attn',
                                                          use_edges_info=config["use_edges_info"], activation='relu',
                                                          bias=False).to(device)

                            elif encoder_ == "RotatEGCN_conv":
                                encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                          kg_score_fn='RotatE', variant='conv',
                                                          use_edges_info=config["use_edges_info"], activation='relu',
                                                          bias=False).to(device)

                            elif encoder_ == "RotatEGCN_attn":
                                encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                          kg_score_fn='RotatE', variant='attn',
                                                          use_edges_info=config["use_edges_info"], activation='relu',
                                                          bias=False).to(device)

                            elif encoder_ == "GAT":
                                encoder = GATEncoder(data, out_channels, config["num_layers"])


                            if decoder_ == "GCN":
                                decoder = GCNDecoder(encoder, data, config["alpha"], message_sens=msg_sens).to(device)
                            elif decoder_ == "MLP":
                                decoder = MLPDecoder(encoder, data, config["alpha"]).to(device)

                            elif decoder_ == "TransGCN_conv":
                                decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                          kg_score_fn='TransE',
                                                          variant='conv',
                                                          use_edges_info=config["use_edges_info"]).to(device)

                            elif decoder_ == "TransGCN_attn":
                                decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                          kg_score_fn='TransE', variant='attn',
                                                          use_edges_info=config["use_edges_info"]).to(device)
                            elif decoder_ == "RotatEGCN_conv":
                                decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                          kg_score_fn='RotatE', variant='conv',
                                                          use_edges_info=config["use_edges_info"]).to(device)

                            elif decoder_ == "RotatEGCN_attn":
                                decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                                          kg_score_fn='RotatE', variant='attn',
                                                          use_edges_info=config["use_edges_info"]).to(device)

                            elif decoder_ == "GAT":
                                decoder = GATDecoder(encoder, data, heads=4, alpha=0.01, dropout=0.3)



                            else:
                                print("Error: RGCN decoder requires num_bases but is not defined!")
                                raise ValueError("Invalid encoder type!")


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
                            local_data = copy.deepcopy(data)

                            autoencoder = MRGAE(encoder, decoder, projections=config["projections"]).to(device)
                            optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                            performances = train_X_reconstruction(autoencoder, local_data, optimizer, config["num_epochs"],
                                            gdp, file_name, device, config,save_dir=save_dir, loss_fct=["MSE"],
                                        wandb=wandb, seed = config["seed"])
                            results.append(performances)

                            wandb.finish()

        elif task == "Recons_A":

            for out_channels in config["hyperparams_grid"]["out_channels"]:
                for encoder_ in config["encoders"]:
                    if encoder_ == "RGCN":
                        for num_bases in config["hyperparams_grid"]["num_bases"]:
                            encoder = RGCNEncoder(data, out_channels, config["num_layers"], num_bases,
                                              message_sens=msg_sens).to(device)
                            run_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_GAE"
                            file_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_GAE"
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
                                "training_task": task,
                                "encoders": encoder_,
                                "decoders": "Dot Product",
                                "message_sens": msg_sens
                            }

                            wandb.init(
                                project=config["wandb_project_name"],
                                name=run_name,
                                config=run_config,
                                settings=wandb.Settings(start_method="thread")
                            )
                            local_data = copy.deepcopy(data)

                            autoencoder = GAE(encoder).to(device)
                            optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                            performances = train_GAE(autoencoder, local_data, optimizer, config["num_epochs"], gdp,save_file = file_name,
                                         save_dir=config["root_save_dir"],device = device, wandb=wandb, seed=seed)

                            results.append(performances)
                            wandb.finish()
                    else:
                        if encoder_ == "GCN":
                            encoder = GCNEncoder(data, out_channels, config["num_layers"],
                                                 message_sens=msg_sens).to(device)

                        elif encoder_ == "TransGCN_conv":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='TransE', variant='conv',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "TransGCN_attn":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='TransE', variant='attn',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "RotatEGCN_conv":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='RotatE', variant='conv',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "RotatEGCN_attn":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='RotatE', variant='attn',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)



                        elif encoder_ == "GAT":
                            encoder = GATEncoder(data, out_channels, config["num_layers"])
                        else:
                            print("invalid encoder type!")
                            raise ValueError("Invalid encoder type!")



                        run_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_GAE"
                        file_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_GAE"
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
                            "training_task": task,
                            "encoders": encoder_,
                            "decoders": "Dot Product",
                            "message_sens": msg_sens
                        }

                        wandb.init(
                            project=config["wandb_project_name"],
                            name=run_name,
                            config=run_config,
                            settings=wandb.Settings(start_method="thread")
                        )
                        local_data = copy.deepcopy(data)

                        autoencoder = GAE(encoder).to(device)
                        optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                        performances = train_GAE(autoencoder, local_data, optimizer, config["num_epochs"], gdp,
                                                 save_file=file_name,
                                                 save_dir=config["root_save_dir"], device=device, wandb=wandb, seed = config["seed"])

                        results.append(performances)
                        wandb.finish()

        elif task == "Recons_R":
            for out_channels in config["hyperparams_grid"]["out_channels"]:
                for encoder_ in config["encoders"]:
                    if encoder_ == "RGCN":
                        for num_bases in config["hyperparams_grid"]["num_bases"]:
                            encoder = RGCNEncoder(data, out_channels, config["num_layers"], num_bases,
                                              message_sens=msg_sens).to(device)
                            run_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_Dismult"
                            file_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_Dismult"
                            run_config = {
                                "device": config["device"],
                                "num_layers": 2,
                                "alpha": config["alpha"],
                                "max_masking_percentage": config["max_masking_percentage"],
                                "total_drop_rate": config["total_drop_rate"],
                                "learning_rate": config["learning_rate"],
                                "batch_size": config["batch_size"],
                                "num_neighbors": config["num_neighbors"],
                                "num_epochs": config["num_epochs"],
                                "bases": num_bases,
                                "out_channels": out_channels,
                                "training_task": task,
                                "encoders": encoder_,
                                "decoders": "DisMult",
                                "message_sens": msg_sens
                            }

                            wandb.init(
                                project=config["wandb_project_name"],
                                name=run_name,
                                config=run_config,
                                settings=wandb.Settings(start_method="thread")
                            )
                            r_decoder = DistMultDecoder(data.num_edge_types, out_channels[-1])
                            autoencoder = MRGAE(encoder,x_decoder = None, r_decoder= r_decoder).to(device)
                            optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])

                            local_data = copy.deepcopy(data)

                            performances = train_DisMult(autoencoder, local_data, optimizer, config["num_epochs"],gdp,file_name,device,save_dir=config["root_save_dir"]+"/"+task, wandb=wandb, seed = config["seed"])
                            # performances = train_GAE(autoencoder, data, optimizer, config["num_epochs"], gdp,save_file = file_name,
                            #              save_dir=config["root_save_dir"],device = device, wandb=wandb)
                            #
                            #
                            results.append(performances)

                            wandb.finish()
                    else:
                        if encoder_ == "GCN":
                            encoder = GCNEncoder(data, out_channels, config["num_layers"],
                                             message_sens=msg_sens).to(device)


                        elif encoder_ == "TransGCN_conv":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='TransE', variant='conv',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "TransGCN_attn":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='TransE', variant='attn',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "RotatEGCN_conv":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='RotatE', variant='conv',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "RotatEGCN_attn":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='RotatE', variant='attn',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)





                        elif encoder_ == "GAT":
                            encoder = GATEncoder(data, out_channels, config["num_layers"])

                        else:
                            print("invalid encoder type!")
                            raise ValueError("Invalid encoder type!")


                        run_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_Dismult" + transgcn_params
                        file_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}_Dismult" + transgcn_params
                        run_config = {
                            "device": config["device"],
                            "num_layers": 2,
                            "alpha": config["alpha"],
                            "max_masking_percentage": config["max_masking_percentage"],
                            "total_drop_rate": config["total_drop_rate"],
                            "learning_rate": config["learning_rate"],
                            "batch_size": config["batch_size"],
                            "num_neighbors": config["num_neighbors"],
                            "num_epochs": config["num_epochs"],
                            "out_channels": out_channels,
                            "training_task": task,
                            "encoders": encoder_,
                            "decoders": "DisMult",
                            "message_sens": msg_sens
                        }

                        wandb.init(
                            project=config["wandb_project_name"],
                            name=run_name,
                            config=run_config,
                            settings=wandb.Settings(start_method="thread")
                        )
                        r_decoder = DistMultDecoder(data.num_edge_types, out_channels[-1])
                        autoencoder = MRGAE(encoder, x_decoder=None, r_decoder=r_decoder).to(device)
                        optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])

                        performances = train_DisMult(autoencoder, data, optimizer, config["num_epochs"], gdp, file_name,
                                                     device, save_dir=config["root_save_dir"]+"/"+task, wandb=wandb, seed = config["seed"])
                        results.append(performances)

                        wandb.finish()

        elif task == "Double_reconstruction":
            for cmb in config["param_combinations"]:
                if cmb["encoder"] == "GCN":
                    encoder = GCNEncoder(data, cmb["out_channels"], config["num_layers"],
                                             message_sens=msg_sens).to(device)
                elif cmb["encoder"] == "RGCN":
                    encoder = RGCNEncoder(data, cmb["out_channels"], config["num_layers"], 5,
                                          message_sens=msg_sens).to(device)
                elif cmb["encoder"] == "TransGCN":
                    encoder = TransGCNEncoder(data, cmb["out_channels"], config["num_layers"], dropout=0.2,
                                              kg_score_fn=config["kg_score_fn"], variant=config["variant"],
                                              use_edges_info=config["use_edges_info"], activation='relu',
                                              bias=False).to(device)


                else:
                    print("invalid encoder type!")
                    raise ValueError("Invalid encoder type!")


                if cmb["decoder"] == "GCN":
                    decoder = GCNDecoder(encoder, data, config["alpha"], message_sens=msg_sens).to(device)
                elif cmb["decoder"] == "RGCN":
                    decoder = RGCNDecoder(encoder, data, 5, config["alpha"],
                                          message_sens=msg_sens).to(device)
                elif cmb["decoder"] == "MLP":
                    decoder = MLPDecoder(encoder, data, config["alpha"]).to(device)
                elif cmb["decoder"] == "TransGCN":
                    decoder = TransGCNDecoder(encoder, data, config["alpha"], dropout=0.3,
                                              kg_score_fn=config["kg_score_fn"], variant=config["variant"],
                                              use_edges_info=config["use_edges_info"]).to(device)


                else:
                    print("invalid decoder type!")
                    raise ValueError("Invalid encoder type!")


                run_name = f"{task}_channels_{'-'.join(map(str, cmb['out_channels']))}_enc-{cmb['encoder']}_dec-{cmb['decoder']}_R_Dismult"
                file_name = f"{task}_channels_{'-'.join(map(str, cmb['out_channels']))}_enc-{cmb['encoder']}_dec-{cmb['decoder']}_R_Dismult"
                run_config = {
                    "device": config["device"],
                    "num_layers": 2,
                    "alpha": config["alpha"],
                    "max_masking_percentage": config["max_masking_percentage"],
                    "total_drop_rate": config["total_drop_rate"],
                    "learning_rate": config["learning_rate"],
                    "batch_size": config["batch_size"],
                    "num_neighbors": config["num_neighbors"],
                    "num_epochs": config["num_epochs"],
                    "out_channels": cmb["out_channels"],
                    "training_task": task,
                    "encoder": cmb["encoder"],
                    "decoder": cmb["decoder"],
                    "r_decoder": "DisMult",
                    "message_sens": msg_sens
                }

                wandb.init(
                    project=config["wandb_project_name"],
                    name=run_name,
                    config=run_config,
                    settings=wandb.Settings(start_method="thread")
                )



                r_decoder = DistMultDecoder(data.num_edge_types, cmb["out_channels"][-1])
                local_data = copy.deepcopy(data)

                autoencoder = MRGAE(encoder, x_decoder=decoder, r_decoder=r_decoder).to(device)
                optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])

                performances = train_Double_Reconstruction(autoencoder, local_data, optimizer, config["num_epochs"], gdp, file_name,
                                             device, save_dir=config["root_save_dir"] + "/" + task, wandb=wandb, seed = config["seed"])
                results.append(performances)

                wandb.finish()

        elif task == "Contrastive":
            if masked_features_data is None or removed_edge_indices is None:
                print("\n--- Preparing views ONCE for contrastive learning ---\n")
                masked_features_data = view_partial_features_masking(
                    data, max_masking_percentage=config["max_masking_percentage"], random_seed=seed
                )
                _, removed_edge_indices, _ = relation_based_edge_dropping_balanced(
                    data, config["total_drop_rate"], max_drop_fraction_per_node=0.3, random_seed=seed
                )
                removed_edge_indices = removed_edge_indices.to(device)
                        
            for out_channels in config["hyperparams_grid"]["out_channels"]:
                for encoder_ in config["encoders"]:
                    if encoder_ == "RGCN":
                        for num_bases in config["hyperparams_grid"]["num_bases"]:
                            encoder = RGCNEncoder(data, out_channels, config["num_layers"], num_bases,
                                                  message_sens=msg_sens).to(device)

                            run_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}"
                            file_name = f"{task}_bases-{num_bases}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}"
                            run_config = {
                                "device": config["device"],
                                "num_layers": config["num_layers"],
                                "learning_rate": config["learning_rate"],
                                "batch_size": config["batch_size"],
                                "num_neighbors": config["num_neighbors"],
                                "num_epochs": config["num_epochs"],
                                "bases": num_bases,
                                "out_channels": out_channels,
                                "training_task": task,
                                "encoders": encoder_,
                                "projections": config["projections"]
                            }

                            wandb.init(
                                project=config["wandb_project_name"],
                                name=run_name,
                                config=run_config,
                                settings=wandb.Settings(start_method="thread")
                            )

                            autoencoder = MRGAE(encoder, x_decoder=None, r_decoder=None, projections=[out_channels[-1], out_channels[-1]]).to(
                                device)
                            optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                            local_data = copy.deepcopy(data)

                            performances = train_Contrastive(
                                autoencoder, local_data, optimizer, config["num_epochs"], gdp, file_name,
                                masked_features_data, removed_edge_indices,
                                device=device, save_dir=config["root_save_dir"] + "/" + task,
                                wandb=wandb, seed=config["seed"]
                            )
                            results.append(performances)
                            wandb.finish()

                    else:  # GCN, TransGCN, etc.
                        if encoder_ == "GCN":
                            encoder = GCNEncoder(data, out_channels, config["num_layers"], message_sens=msg_sens).to(device)

                        elif encoder_ == "TransGCN_conv":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='TransE', variant='conv',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "TransGCN_attn":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='TransE', variant='attn',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "RotatEGCN_conv":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='RotatE', variant='conv',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "RotatEGCN_attn":
                            encoder = TransGCNEncoder(data, out_channels, config["num_layers"], dropout=0.2,
                                                      kg_score_fn='RotatE', variant='attn',
                                                      use_edges_info=config["use_edges_info"], activation='relu',
                                                      bias=False).to(device)

                        elif encoder_ == "GAT":
                            encoder = GATEncoder(data, out_channels, config["num_layers"]).to(device)

                        else:
                            raise ValueError("Invalid encoder for Contrastive task!")

                        run_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}"
                        file_name = f"{task}_channels_{'-'.join(map(str, out_channels))}_enc-{encoder_}"
                        run_config = {
                            "device": config["device"],
                            "num_layers": config["num_layers"],
                            "learning_rate": config["learning_rate"],
                            "batch_size": config["batch_size"],
                            "num_neighbors": config["num_neighbors"],
                            "num_epochs": config["num_epochs"],
                            "out_channels": out_channels,
                            "training_task": task,
                            "encoders": encoder_,
                            "projections": config["projections"]
                        }

                        wandb.init(
                            project=config["wandb_project_name"],
                            name=run_name,
                            config=run_config,
                            settings=wandb.Settings(start_method="thread")
                        )
                        autoencoder = MRGAE(encoder, x_decoder=None, r_decoder=None, projections= [out_channels[-1], out_channels[-1]]).to(device)
                        optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
                        local_data = copy.deepcopy(data)


                        performances = train_Contrastive(
                            autoencoder, local_data, optimizer, config["num_epochs"], gdp, file_name,
                            masked_features_data, removed_edge_indices,
                            device=device, save_dir=config["root_save_dir"] + "/" + task,
                            wandb=wandb, seed = config["seed"]
                        )                       
                        results.append(performances)
                        wandb.finish()

    df = pd.DataFrame(results)
    # Sauvegarde en fichier Excel
    df.to_excel("Recons_R_noisy_mapped_seed_0.xlsx", index=False)
#
# def main():
#     print(44)

if __name__ == "__main__":
    main()
