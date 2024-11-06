# train.py
import sys
import os

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

# Entraînement
def train(model, data, optimizer, loss_fn, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        embeddings = model.encode(data)
        reconstructed_x = model.decode_x(data, embeddings)
        loss = model.recon_x_loss(data, reconstructed_x)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

train(autoencoder, data, optimizer, loss_fn, config["num_epochs"])

