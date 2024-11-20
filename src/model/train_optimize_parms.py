# train.py
import sys
import os

import torch
# Ajouter les dossiers 'layers' et 'data' au chemin Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
from data_augmentation import view_partial_features_masking
from data_augmentation import relation_based_edge_dropping
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
wandb.login(key="c278e62d2025b60ff8b984a40f7b62b697f9b4fd", relogin=True)
# Configuration du projet wandb
wandb_project_name = "MC2GAE_hyperparam_optimization_contrastive"

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
    "num_bases": [10, 55, 110],  # Exemple de valeurs pour num_bases
    "out_channels": [[600,500],[500, 250], [128, 64], [50,25]]  # Exemple de valeurs pour out_channels
}

training_options = ["contrastive"]

# Fonction de sauvegarde de modèle incluant les hyperparamètres dans le nom du fichier
def save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir="checkpoints_contrastive",
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
                           save_dir="checkpoints_contrastive", training_options = ["contrastive"]):
    model.train()
    best_loss = float('inf')

    # Application du masque de features
    print("\nmask_features...\n")

    masked_features_data = view_partial_features_masking(data)
    print("\nEdge_dripping...\n")
    masked_edges_data = relation_based_edge_dropping(data, config["total_drop_rate"])
    G1_data_loader = GraphDataLoader(masked_features_data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()

    G2_data_loader = GraphDataLoader(masked_edges_data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
            if "contrastive" in training_options:
                for G1_batch, G2_batch in zip(G1_data_loader, G2_data_loader):
                    G1_batch.to(device)
                    G2_batch.to(device)
                    n_id_1 = G1_batch.n_id  ## The global node index for every sampled node
                    mask_G1 = torch.isin(n_id_1, G1_batch.input_id) ## mask to get only the embedding of input_id nodes
                    n_id_2 = G2_batch.n_id
                    mask_G2 = torch.isin(n_id_2, G2_batch.input_id)
                    H1_batch = model.encode(G1_batch)
                    H2_batch = model.encode(G2_batch)
                    loss = model.contrastive_loss(H1_batch[mask_G1], H2_batch[mask_G2])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)
            else:
                for batch in G1_data_loader:
                    batch = batch.to(device)
                    n_id = batch.n_id ## The global node index for every sampled node
                    mask = torch.isin(n_id, batch.input_id) ## mask to get only the embedding of input_id nodes
                    optimizer.zero_grad()
                    embeddings = model.encode(batch)
                    # print(batch)
                    reconstructed_x = model.decode_x(batch, embeddings)
                    reconstructed_x = reconstructed_x[mask]
                    loss = model.recon_x_loss(batch.x[mask], reconstructed_x)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)

        avg_loss = total_loss / len(G1_data_loader)

        # Loguer la perte de chaque époque dans wandb
        if "contrastive" in training_options and len(training_options) == 1:
            wandb.log({"epoch": epoch + 1, "Reconstruct_X_loss": avg_loss})
        elif "contrastive" in training_options and len(training_options) == 1:
            wandb.log({"epoch": epoch + 1, "Contrastive_loss": avg_loss})

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
        autoencoder = MC2GEA(RGCN_encoder, RGCN_decoder).to(device)

        # Optimizer
        optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])

        # Lancer l'entraînement avec les hyperparamètres actuels
        train_with_hyperparams(autoencoder, data, optimizer, config["num_epochs"], num_bases, out_channels)

        # Finir le run dans wandb
        wandb.finish()



