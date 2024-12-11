import sys
import os

from src.layers.GCNDecoder import GCNDecoder
from src.layers.GCNEncoder import GCNEncoder
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



wandb_project_name = "classic_DGI"

save_dir = config["save_dir"]
Entities_path = config["Entities_path"]
KG_path = config["KG_path"]
Edges_path = config["Edges_path"]
device = config["device"]

gdp = GraphDataPreparation(Entities_path, KG_path, edges_embd_path=Edges_path, is_directed=True)
data = gdp.prepare_graph_with_type()
data = Data(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type).to(device)


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DeepGraphInfomax, BatchNorm
from torch_geometric.data import Data
# Define the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# class Encoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super(Encoder, self).__init__()
#         self.conv = GCNConv(in_channels, hidden_channels)

#     def forward(self, x, edge_index):
#         x = self.conv(x, edge_index)
#         return x

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout_rate=0.4):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout_rate = dropout_rate

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i < self.num_layers - 1:  # Apply ReLU and Dropout only if it's not the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x

hidden_channels = 512

# Define the encoder model and the Deep Graph Infomax model
encoder = Encoder(data.num_node_features, hidden_channels=hidden_channels)
model = DeepGraphInfomax(
    hidden_channels=hidden_channels, encoder=encoder,
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=lambda x, edge_index: (x[torch.randperm(x.size(0))], edge_index)
)
model = model.to(device)
lr=0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# KG_path = config["KG_path"]
# Gs_path = config["Gs_path_no_other"]
# thresholds_list = [0.6,0.7,0.8]
#
# evaluate_all(KG_path, Gs_path, "checkpoints/DGI", config, embedding_model = "GNN", with_other = False, thresholds_list = thresholds_list)
# exit(2222)

def train():
    model.train()
    optimizer.zero_grad()
    try:
        pos_z, neg_z, summary = model(data.x, data.edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        return loss.item()
    except Exception as e:
        print(f"Error during training: {e}")
        return None

# Training loop
num_epochs = 400
wandb.init(
            project=wandb_project_name,
            name=f"DGI_{hidden_channels}",
            config= {"hidden_channels": hidden_channels, "num_epochs": num_epochs, "lr": lr  },
            settings=wandb.Settings(start_method="thread")
        )
best_accuracy = 0
for epoch in tqdm(range(num_epochs)):

    loss = train()
    if loss is not None:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs}, Error occurred")

    metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
    if metrics["accuracy"] > best_accuracy:
        best_accuracy = metrics["accuracy"]
        save_model_with_hyperparams(model, optimizer, epoch, 0, [512,512], save_dir="DGI",
                                    is_best_acc=True)
    print(metrics)
    wandb.log(
        {"epoch": epoch + 1, "global loss": loss,
         "accuracy": metrics["accuracy"], "f1-score": metrics["f1_score"],
         "recall": metrics["recall"], "precision": metrics["precision"],
         })

wandb.finish()

# Extract the embeddings after training
try:
    data_x = data.x.to(device)
    data_edge_index = data.edge_index.to(device)  # Keep it as a PyTorch tensor
    model.eval()
    embeddings = model.encoder(data_x, data_edge_index)
    print("Embeddings extracted successfully")
    print(embeddings)

except Exception as e:
    print(f"Error during embedding extraction: {e}")












# import sys
# import os
#
# import torch
# from torch_geometric.nn import DeepGraphInfomax
#
# # Ajouter les dossiers 'layers' et 'data' au chemin Python
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
# from data_augmentation import view_partial_features_masking
# from data_augmentation import relation_based_edge_dropping
# from GraphDataLoader import GraphDataLoader
# import torch.optim as optim
# from RGCNEncoder1 import RGCNEncoder1
# from RGCNDecoder import RGCNDecoder
# from torch.nn import MSELoss
# from torch_geometric.data import Data
# from config import config
# from GraphDataPreparation import GraphDataPreparation
# from MC2GEA import  MC2GEA
# from tqdm import tqdm
# import wandb
# import torch
# import random
# import numpy as np
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
#
#
# wandb.require("legacy-service")
# wandb.login(key="c278e62d2025b60ff8b984a40f7b62b697f9b4fd", relogin=True)
# # Configuration du projet wandb
# wandb_project_name = "DGI"
#
# # Chargement des données
# device = config['device']
# Entities_path = config["Entities_path"]
# Edges_path = config["Edges_path"]
# KG_path = config["KG_path"]
#
# # Préparation du graphe
# gdp = GraphDataPreparation(Entities_path, KG_path, edges_embd_path=Edges_path, is_directed=True)
# data = gdp.prepare_graph_with_type()
# data = Data(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type).to(device)
#
# # Dictionnaire pour stocker les combinaisons d'hyperparamètres
#
# hyperparams_grid = {
#     "num_bases": [10,30],  # Exemple de valeurs pour num_bases
#     "out_channels": [[600,500],[500,500],[500, 250], [128, 64], [50,25]]  # Exemple de valeurs pour out_channels
# }
#
# training_options = ["contrastive"]
#
# # Fonction de sauvegarde de modèle incluant les hyperparamètres dans le nom du fichier
# def save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir="DGI",
#                                 is_best=False):
#     os.makedirs(save_dir, exist_ok=True)
#     base_filename = f"best_model_bases{num_bases}_channels{'-'.join(map(str, out_channels))}"
#     checkpoint_path = os.path.join(save_dir,
#                                    f"{base_filename}.pth" if is_best else f"{base_filename}_epoch_{epoch + 1}.pth")
#
#     torch.save({
#         'epoch': epoch + 1,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'num_bases': num_bases,
#         'out_channels': out_channels
#     }, checkpoint_path)
#     print(f"Model saved at '{checkpoint_path}'")
#
# def corruption(x, edge_index, edge_type):
#     # Corruption des features
#     corrupted_x = x[torch.randperm(x.size(0))]
#     # Les edge_index et edge_type restent inchangés
#     return corrupted_x, edge_index, edge_type
#
#
#
#
# # Fonction d'entraînement avec suivi de la perte dans wandb
# def train_with_hyperparams(model, data, optimizer, num_epochs, num_bases, out_channels, save_every=10,
#                            save_dir="DGI", training_options = ["contrastive"]):
#     model.train()
#     best_loss = float('inf')
#     G_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"],
#                                      batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         with tqdm(total=len(G_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
#             if "contrastive" in training_options:
#                 for G_batch in G_data_loader:
#                     G_batch.to(device)
#
#                     corrupted_x, corrupted_edge_index, corrupted_edge_type = model.corruption(
#                         G_batch.x, G_batch.edge_index, G_batch.edge_type
#                     )
#                     pos_z, neg_z, summary = model(G_batch.x, G_batch.edge_index, G_batch.edge_type)
#                     loss = model.loss(pos_z, neg_z, summary)
#                     loss.backward()
#                     optimizer.step()
#                     total_loss += loss.item()
#                     batch_pbar.update(1)
#             else:
#                 for batch in G_data_loader:
#                     batch = batch.to(device)
#                     n_id = batch.n_id ## The global node index for every sampled node
#                     mask = torch.isin(n_id, batch.input_id) ## mask to get only the embedding of input_id nodes
#                     optimizer.zero_grad()
#                     embeddings = model.encode(batch)
#                     # print(batch)
#                     reconstructed_x = model.decode_x(batch, embeddings)
#                     reconstructed_x = reconstructed_x[mask]
#                     loss = model.recon_x_loss(batch.x[mask], reconstructed_x)
#                     loss.backward()
#                     optimizer.step()
#                     total_loss += loss.item()
#                     batch_pbar.set_postfix(batch_loss=loss.item())
#                     batch_pbar.update(1)
#
#         avg_loss = total_loss / len(G_data_loader)
#
#         # Loguer la perte de chaque époque dans wandb
#         if "contrastive" in training_options and len(training_options) == 1:
#             wandb.log({"epoch": epoch + 1, "DGI": avg_loss})
#         elif "Recostruct_X" in training_options and len(training_options) == 1:
#             wandb.log({"epoch": epoch + 1, "Contrastive_loss": avg_loss})
#
#         # Sauvegarde du modèle si la perte est la plus faible
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
#                                         is_best=True)
#             print(f'Model saved with Avg Loss: {best_loss:.4f}')
#
#
# # Boucle d'optimisation des hyperparamètres
# for num_bases in hyperparams_grid["num_bases"]:
#     for out_channels in hyperparams_grid["out_channels"]:
#         # Nom unique pour chaque run, incluant les hyperparamètres
#         run_name = f"bases_{num_bases}_channels_{'-'.join(map(str, out_channels))}"
#
#         # Initialisation de wandb pour chaque combinaison d'hyperparamètres
#         wandb.init(
#             project=wandb_project_name,
#             name=run_name,
#             config={
#                 "num_bases": num_bases,
#                 "out_channels": out_channels,
#                 "learning_rate": config["learning_rate"],
#                 "batch_size": config["batch_size"],
#                 "num_epochs": config["num_epochs"]
#             },
#             settings=wandb.Settings(start_method="thread")
#         )
#
#         print(f"\nTraining with num_bases={num_bases} and out_channels={out_channels}...\n")
#
#         # Initialiser le modèle avec les hyperparamètres actuels
#         RGCN_encoder = RGCNEncoder1(data, out_channels, config["num_layers"], num_bases).to(device)
#         # RGCN_decoder = RGCNDecoder(RGCN_encoder, data, num_bases, config["alpha"]).to(device)
#         # autoencoder = MC2GEA(RGCN_encoder, RGCN_decoder).to(device)
#         model = DeepGraphInfomax(
#             hidden_channels=out_channels[1], encoder=RGCN_encoder,
#             summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
#             corruption=corruption
#         ).to(device)
#
#         # Optimizer
#         optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
#
#         # Lancer l'entraînement avec les hyperparamètres actuels
#         train_with_hyperparams(model, data, optimizer, config["num_epochs"], num_bases, out_channels)
#
#         # Finir le run dans wandb
#         wandb.finish()
#
#
