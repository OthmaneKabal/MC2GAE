# train.py
import sys
import os
from glob import glob1

from numpy.ma.setup import configuration
from sympy.core.random import shuffle

# Ajouter les dossiers 'layers' et 'data' au chemin Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
from data_augmentation import view_partial_features_masking
from data_augmentation import relation_based_edge_dropping
from GraphDataLoader import GraphDataLoader
from utils import save_model, load_model_checkpoint
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


# Chargement des données
device = config['device']
Entities_path = config["Entities_path"]
Edges_path = config["Edges_path"]
KG_path = config["KG_path"]
continue_train = False
ckpt_file = "checkpoints/model_epoch_20.pth"
# Préparation du graphe
gdp = GraphDataPreparation(Entities_path, KG_path, edges_embd_path=Edges_path, is_directed=True)
data = gdp.prepare_graph_with_type()
data = Data(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type).to(device)
## data loader




# Initialisation du modèle
RGCN_encoder = RGCNEncoder(data, config["out_channels"], config["num_layers"], config["num_bases"]).to(device)
RGCN_decoder = RGCNDecoder(RGCN_encoder, data,config["num_bases"], config["alpha"]).to(device)
autoencoder = MC2GEA(RGCN_encoder, RGCN_decoder, options = config["options"]).to(device)

# Définition de l'optimiseur et de la fonction de perte
optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
loss_fn = MSELoss()

configuration = {}
wandb.init(project="MC2GAE", config=configuration, settings=wandb.Settings(start_method="thread"))  # Remplace "nom_de_ton_projet" par le nom de ton choix
configuration = wandb.config

def train(model, data, optimizer, num_epochs, save_every=10, save_dir="checkpoints"):
    model.train()

    print("\nmask_features...\n")
    masked_features_data = view_partial_features_masking(data)
    G1_data_loader = GraphDataLoader(masked_features_data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()
    # Boucle d'entraînement avec barre de progression pour les époques
    best_loss = float('inf')
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
                # Accumuler la perte pour l'affichage
                total_loss += loss.item()
                wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1})
                # Mise à jour de la barre de progression des batches
                batch_pbar.set_postfix(batch_loss=loss.item())
                batch_pbar.update(1)

        # Calculer la perte moyenne de l'époque
        avg_loss = total_loss / len(G1_data_loader)
        # Loguer les métriques dans wandb
        wandb.log({
            "epoch": epoch + 1,
            "Reconstruct_X_loss": avg_loss

        })

        # Affichage de la perte moyenne pour chaque époque
        print(f'Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss:.4f}')
        # Sauvegarder le modèle tous les 'save_every' epochs
        if (epoch + 1) % save_every == 0:
            save_model(model, optimizer, epoch, save_dir = save_dir)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, optimizer, epoch, save_dir = save_dir, is_best = True)
            print(f'Model saved with Avg Loss: {best_loss:.4f}')


# Lancement de l'entraînement
if continue_train:
    print("\n resume training model .... \n")
    autoencoder, optimizer, start_epoch = load_model_checkpoint(autoencoder, optimizer, ckpt_file)
train(autoencoder, data, optimizer, num_epochs=config["num_epochs"])
# Terminer l'enregistrement avec wandb
wandb.finish()
