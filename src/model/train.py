# train.py
import sys
import os
from glob import glob1

from data.data_augmentation import view_partial_features_masking
from data.data_augmentation import relation_based_edge_dropping

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

print(data)
# Initialisation du modèle
RGCN_encoder = RGCNEncoder(data, config["out_channels"], config["num_layers"], config["num_bases"]).to(device)
RGCN_decoder = RGCNDecoder(RGCN_encoder, data,config["num_bases"], config["alpha"]).to(device)
autoencoder = MC2GEA(RGCN_encoder, RGCN_decoder, contrastive = True).to(device)

# Définition de l'optimiseur et de la fonction de perte
optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
loss_fn = MSELoss()


from tqdm import tqdm

def train(model, data, optimizer, num_epochs, save_every=10, save_dir="checkpoints"):
    model.train()
    print("\nmask_features...\n")
    masked_features_data = view_partial_features_masking(data)
    print("\nedge_dropping...\n")
    masked_edges_data = relation_based_edge_dropping(data, 0.3)

    # Initialiser la barre de progression
    with tqdm(total=num_epochs, desc="Training Progress", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Forward pass
            embeddings = model.encode({"G1":masked_features_data,"G2":masked_edges_data})
            reconstructed_x = model.decode_x(data, embeddings["H1"])

            loss = model.recon_x_loss(data, reconstructed_x) #+ model.contrastive_loss({"G1":masked_edges_data,"G2":masked_edges_data})
            # loss =  model.contrastive_loss(embeddings)

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
if continue_train:
    print("\n resume training model .... \n")
    autoencoder, optimizer, start_epoch = load_model_checkpoint(autoencoder, optimizer, ckpt_file)

train(autoencoder, data, optimizer, num_epochs=config["num_epochs"])

