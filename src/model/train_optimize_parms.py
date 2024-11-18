# train.py
import sys
import os

import torch
# Ajouter les dossiers 'layers' et 'data' au chemin Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
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
wandb.require("legacy-service")
wandb.login(key="c278e62d2025b60ff8b984a40f7b62b697f9b4fd",relogin=True)
# Configuration du projet wandb
wandb_project_name = "MC2GAE_hyperparam_optimization"

# Chargement des données
device = config['device']
Entities_path = config["Entities_path"]
Edges_path = config["Edges_path"]
KG_path = config["KG_path"]

# Préparation du graphe
gdp = GraphDataPreparation(Entities_path, KG_path, edges_embd_path=Edges_path, is_directed=True)
data = gdp.prepare_graph_with_type()
data = Data(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type).to(device)

# Dictionnaire pour stocker les combinaisons d'hyperparamètres
hyperparams_grid = {
    "num_bases": [10 , 30, 40, 50, 55, 110],  # Exemple de valeurs pour num_bases
    "out_channels": [[768,768],[500, 250], [256, 128], [128, 64], [64, 32], [32, 16], [25,25]]  # Exemple de valeurs pour out_channels
}


# Fonction de sauvegarde de modèle incluant les hyperparamètres dans le nom du fichier
def save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir="checkpoints",
                                is_best=False):
    os.makedirs(save_dir, exist_ok=True)
    base_filename = f"best_model_bases{num_bases}_channels{'-'.join(map(str, out_channels))}"
    checkpoint_path = os.path.join(save_dir,
                                   f"{base_filename}.pth" if is_best else f"{base_filename}_epoch_{epoch + 1}.pth")

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_bases': num_bases,
        'out_channels': out_channels
    }, checkpoint_path)
    print(f"Model saved at '{checkpoint_path}'")


# Fonction d'entraînement avec suivi de la perte dans wandb
def train_with_hyperparams(model, data, optimizer, num_epochs, num_bases, out_channels, save_every=10,
                           save_dir="checkpoints"):
    model.train()
    best_loss = float('inf')

    # Application du masque de features
    masked_features_data = view_partial_features_masking(data)
    G1_data_loader = GraphDataLoader(masked_features_data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()

    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
            for batch in G1_data_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                # Forward pass
                embeddings = model.encode(batch)
                reconstructed_x = model.decode_x(batch, embeddings)
                # Calcul de la perte de reconstruction
                loss = model.recon_x_loss(batch, reconstructed_x)
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_pbar.set_postfix(batch_loss=loss.item())
                batch_pbar.update(1)

        avg_loss = total_loss / len(G1_data_loader)

        # Loguer la perte de chaque époque dans wandb
        wandb.log({"epoch": epoch + 1, "Reconstruct_X_loss": avg_loss})

        # Sauvegarde du modèle si la perte est la plus faible
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                        is_best=True)
            print(f'Model saved with Avg Loss: {best_loss:.4f}')


# Boucle d'optimisation des hyperparamètres
for num_bases in hyperparams_grid["num_bases"]:
    for out_channels in hyperparams_grid["out_channels"]:
        # Nom unique pour chaque run, incluant les hyperparamètres
        run_name = f"bases_{num_bases}_channels_{'-'.join(map(str, out_channels))}"

        # Initialisation de wandb pour chaque combinaison d'hyperparamètres
        wandb.init(
            project=wandb_project_name,
            name=run_name,
            config={
                "num_bases": num_bases,
                "out_channels": out_channels,
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "num_epochs": config["num_epochs"]
            },
            settings=wandb.Settings(start_method="thread")
        )

        print(f"\nTraining with num_bases={num_bases} and out_channels={out_channels}...\n")

        # Initialiser le modèle avec les hyperparamètres actuels
        RGCN_encoder = RGCNEncoder(data, out_channels, config["num_layers"], num_bases).to(device)
        RGCN_decoder = RGCNDecoder(RGCN_encoder, data, num_bases, config["alpha"]).to(device)
        autoencoder = MC2GEA(RGCN_encoder, RGCN_decoder, options=config["options"]).to(device)

        # Optimizer
        optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])

        # Lancer l'entraînement avec les hyperparamètres actuels
        train_with_hyperparams(autoencoder, data, optimizer, config["num_epochs"], num_bases, out_channels)

        # Finir le run dans wandb
        wandb.finish()
