# train.py
import sys
import os
from glob import glob1

from data.data_augmentation import view_partial_features_masking
from src.model.utils import save_model, load_model_checkpoint

# Ajouter les dossiers 'layers' et 'data' au chemin Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))

import torch.optim as optim
from RGCNEncoder import RGCNEncoder
from RGCNDecoder import RGCNDecoder
from torch.nn import MSELoss
from torch_geometric.data import Data

from config import config
from GraphDataPreparation import GraphDataPreparation
from MC2GEA import  MC2GEA
# Chargement des données
device = config["device"]
Entities_path = config["Entities_path"]
Edges_path = config["Edges_path"]
KG_path = config["KG_path"]

# Préparation du graphe
gdp = GraphDataPreparation(Entities_path, KG_path, edges_embd_path=Edges_path, is_directed=True)
data = gdp.prepare_graph_with_type()
data = Data(x=data.x, edge_index=data.edge_index, edge_type=data.edge_type).to(device)

# Initialisation du modèle
RGCN_encoder = RGCNEncoder(data, config["out_channels"], config["num_layers"], config["num_bases"]).to(device)
RGCN_decoder = RGCNDecoder(RGCN_encoder, data,config["num_bases"], config["alpha"]).to(device)
autoencoder = MC2GEA(RGCN_encoder, RGCN_decoder).to(device)

# Définition de l'optimiseur et de la fonction de perte
optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
loss_fn = MSELoss()

# # Entraînement
# def train(model, data, optimizer, num_epochs):
#     model.train()
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         embeddings = model.encode(data)
#         reconstructed_x = model.decode_x(data, embeddings)
#         loss = model.recon_x_loss(data, reconstructed_x)
#
#         loss.backward()
#         optimizer.step()
#         if (epoch + 1) % 10 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# train(autoencoder, data, optimizer, loss_fn, config["num_epochs"])

# Training function with saving
from tqdm import tqdm

def train(model, data, optimizer, num_epochs, save_every=10, save_dir="checkpoints"):
    model.train()
    print("\nmask_features...\n")
    masked_features_data = view_partial_features_masking(data)

    # Initialiser la barre de progression
    with tqdm(total=num_epochs, desc="Training Progress", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward pass
            embeddings = model.encode(masked_features_data)
            reconstructed_x = model.decode_x(data, embeddings)
            loss = model.recon_x_loss(data, reconstructed_x)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Affichage de la loss tous les 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

            # Sauvegarder le modèle tous les 50 epochs
            if (epoch + 1) % save_every == 0:
                save_model(model, optimizer, epoch, save_dir=save_dir)

            # Mise à jour de la barre de progression
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)


train(autoencoder, data, optimizer, num_epochs=config["num_epochs"])




# autoencoder.eval()
# embd = autoencoder.encode(data)
# print(data.x.shape,embd.shape)
# print("*"*20, "\n"*2, embd)
#
# model = autoencoder  # Replace with your model instance
# optimizer = optimizer  # Replace with your optimizer instance
#
# # Path to the saved checkpoint (example)
# checkpoint_path = "checkpoints/model_epoch_100.pth"  # Update with the correct path
#
# # Load model and optimizer states
# model, optimizer, start_epoch = load_model_checkpoint(model, optimizer, checkpoint_path)
#
# # You can now resume training from start_epoch if desired
# print(f"Resuming training from epoch {start_epoch}")
# model.eval()
# embd = model.encode(data)
# print(data.x.shape,embd.shape)
# print("*"*20, "\n"*2, embd)