# train.py
import json
import sys
import os
from collections import Counter

import torch
from networkx.algorithms.triads import all_triplets
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch_geometric.loader import NeighborLoader



# Ajouter les dossiers 'layers' et 'data' au chemin Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'utils')))

from ConvE import ConvE
from utils.ConvENegativeSampling import generate_negatives, get_positives

from utils.ConvEDataLoader import create_data_loader

from utils.utils import generate_relation_embeddings_tensor, removed_edges_train_test_split
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
import random
import numpy as np
from torch_geometric.utils import negative_sampling
import copy
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)  # Remplace 42 par la graine que tu souhaites utiliser.

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

wandb.require("legacy-service")
wandb.login(key="c278e62d2025b60ff8b984a40f7b62b697f9b4fd", relogin=True)
# Configuration du projet wandb
wandb_project_name = "MC2GAE_Reconstruct_X_All_cosine"

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
    "num_bases": [10],  # Exemple de valeurs pour num_bases
    "out_channels": [[640,512] , [256, 128], [768,768], [100,50]]  # Exemple de valeurs pour out_channels
}

training_options = ["Reconstruct_X"]

# Fonction de sauvegarde de modèle incluant les hyperparamètres dans le nom du fichier
def save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir="checkpoints_X_all_cosine",
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


def evaluate_ConvE(model, data, data_loader, test_removed_index, device, relation_embeddings):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Evaluation", unit="batch") as eval_pbar:
            for batch in data_loader:
                batch = batch.to(device)
                removed_batch = copy.copy(batch)

                # Masking the edges based on test_removed_index
                test_removed_index = test_removed_index.to(device)
                mask = torch.isin(batch.e_id, test_removed_index)
                removed_batch.edge_index = removed_batch.edge_index[:, mask]
                removed_batch.edge_type = removed_batch.edge_type[mask]
                removed_batch.e_id = removed_batch.e_id[mask]

                # Encoding the batch
                H_2 = model.encode(batch)

                # Generating negative and positive triplets for evaluation
                negative_triplets = generate_negatives(data, removed_batch, negative_ratio=1)
                positive_triplets = get_positives(removed_batch)

                # DataLoader for ConvE evaluation
                eval_loader = create_data_loader(
                    positive_triplets,
                    negative_triplets,
                    H_2,
                    relation_embeddings,
                    batch_size=config["batch_size"] * 3,
                    shuffle=False
                )

                convE_loss = 0
                convE_batches_processed = 0

                for eval_batch in eval_loader:
                    preds = model.r_decoder(eval_batch[0], eval_batch[1], eval_batch[2])
                    loss = model.recon_r_loss(preds, eval_batch[3].to(device))
                    convE_loss += loss.item()

                    predicted_labels = (preds > 0.5).long().detach()
                    all_preds.extend(predicted_labels.cpu().numpy())
                    all_labels.extend(eval_batch[3].cpu().numpy())

                    convE_batches_processed += 1

                total_loss += convE_loss / convE_batches_processed
                eval_pbar.update(1)

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(
        f"\nEvaluation Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")

    return avg_loss, accuracy, recall, precision, f1




# Fonction d'entraînement avec suivi de la perte dans wandb
def train_with_hyperparams(model, data, optimizer, num_epochs, num_bases, out_channels, save_every=10,
                           save_dir="checkpoints_X_all_cosine", training_options = training_options, device = config['device']):

    unique_relations = list(set([i.item() for i in data.edge_type]))
    relation_embeddings = generate_relation_embeddings_tensor(unique_relations, out_channels[1], device,
                                                              seed=42)


    best_loss = float('inf')
    best_F1 = 0

    # Application du masque de features
    print("\nmask_features...\n")

    masked_features_data= view_partial_features_masking(data)


    print("\nEdge_dripping...\n")
    masked_edges_data, removed_edge_indices, removed_edge_types  = relation_based_edge_dropping_balanced(data, config["total_drop_rate"], max_drop_fraction_per_node=0.3, random_seed=42)



    train_removed_edges_indices, test_removed_edges_indices, train_relations, test_relations = removed_edges_train_test_split(removed_edge_indices, removed_edge_types)
    # print(len(removed_edge_indices),"---")
    removed_edge_indices = train_removed_edges_indices.to(device)
    test_removed_edges_indices = test_removed_edges_indices.to(device)


    set_seed(42)
    G1_data_loader = GraphDataLoader(masked_features_data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()

    G2_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()





    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_mse_loss = 0
        total_cos_loss = 0

        if "contrastive" in training_options:
            with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
                for G1_batch, G2_batch in zip(G1_data_loader, G2_data_loader):
                    G1_batch.to(device)
                    G2_batch.to(device)
                    n_id_1 = G1_batch.n_id  ## The global node index for every sampled node
                    mask_G1 = torch.isin(n_id_1, G1_batch.input_id) ## mask to get only the embedding of input_id nodes
                    n_id_2 = G2_batch.n_id
                    mask_G2 = torch.isin(n_id_2, G2_batch.input_id)
                    H1_batch = model.encode(G1_batch)
                    H2_batch = model.encode(G2_batch)
                    H1_projected = model.projector_fc1(H1_batch)[mask_G1]
                    H2_projected = model.projector_fc2(H2_batch)[mask_G2]
                    # Compute contrastive loss
                    loss = model.contrastive_loss(H1_projected, H2_projected)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)
                    # loss = model.contrastive_loss(H1_batch[mask_G1], H2_batch[mask_G2])
                    # loss.backward()
                    # optimizer.step()
                    # total_loss += loss.item()
                    # batch_pbar.set_postfix(batch_loss=loss.item())
                    # batch_pbar.update(1)

        elif "reconstruct_r" in training_options:

            with tqdm(total=len(G2_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as main_pbar:
                nb_intersections = 0
                matching_e_ids = []
                all_preds = []
                all_labels = []

                for G2_batch in G2_data_loader:

                    G2_batch = G2_batch.to(device)
                    removed_batch = copy.copy(G2_batch)

                    removed_edge_indices = removed_edge_indices.to(device)
                    mask = torch.isin(removed_edge_indices, G2_batch.e_id)
                    intersections = removed_edge_indices[mask]
                    # Obtenir les nœuds cibles des arêtes intersectantes
                    intersection_targets = data.edge_index[1][intersections]
                    # Trouver les intersections qui vérifient la condition
                    # (les nœuds cibles sont dans input_id)
                    matching_mask = torch.isin(intersection_targets, G2_batch.input_id)
                    # Récupérer les e_id correspondants
                    batch_matching_e_ids = intersections[matching_mask]
                    edges_mask = torch.isin(G2_batch.e_id,batch_matching_e_ids) ## mask pour les edges à supprimer dans le batch

                    ## the final masked batch
                    G2_batch.edge_index = G2_batch.edge_index[:,~edges_mask]
                    G2_batch.edge_type = G2_batch.edge_type[~edges_mask]
                    G2_batch.e_id = G2_batch.e_id[~edges_mask]

                    removed_batch.edge_index = removed_batch.edge_index[:, edges_mask]
                    removed_batch.edge_type = removed_batch.edge_type[edges_mask]
                    removed_batch.e_id = removed_batch.e_id[edges_mask]
                    optimizer.zero_grad()
                    H_2 = model.encode(G2_batch)

                    # Générer les triplets négatifs et positifs
                    negative_triplets = generate_negatives(data, G2_batch, negative_ratio=1)
                    positive_triplets = get_positives(G2_batch)
                    ## Generate negative examples from removed edges:
                    negative_triplets_removed = generate_negatives(data, removed_batch, negative_ratio=1)
                    positive_triplets_removed = get_positives(removed_batch)

                    all_positive_triplets = torch.cat((positive_triplets, positive_triplets_removed), dim=0)
                    all_negative_triplets = torch.cat((negative_triplets, negative_triplets_removed), dim=0)

                #     # Créer le DataLoader pour les batchs ConvE
                    convE_loader = create_data_loader(all_positive_triplets, all_negative_triplets, H_2, relation_embeddings,
                                                      config["batch_size"]*3, shuffle=True)

                    convE_loss = 0
                    convE_batches_processed = 0
                    avg_convE_loss = 0
                    for convE_batch in convE_loader:
                        # Prédictions et calcul de la perte
                        preds = model.r_decoder(convE_batch[0], convE_batch[1], convE_batch[2])
                        loss = model.recon_r_loss(preds, convE_batch[3].to(device))
                        # Backpropagation avec accumulation des gradients
                        predicted_labels = (preds > 0.5).long().detach()  # Seuil pour convertir les scores en 0/1
                        all_preds.extend(predicted_labels.cpu().numpy())
                        all_labels.extend(convE_batch[3].cpu().numpy())
                        loss.backward(retain_graph=True)
                        # Accumuler la perte totale pour ConvE
                        convE_loss += loss.item()
                        convE_batches_processed += 1
                        # Mise à jour de la barre principale avec les détails du batch ConvE
                        main_pbar.set_postfix(
                            convE_batches=f"{convE_batches_processed}/{len(convE_loader)}"
                        )

                    # Optimisation après accumulation
                    optimizer.step()
                    avg_convE_loss += convE_loss/convE_batches_processed
                    total_loss += avg_convE_loss
                    # Mise à jour de la barre principale pour chaque batch du graphe
                    main_pbar.update(1)
                    main_pbar.set_postfix(
                        convE_batches=f"{convE_batches_processed}/{len(convE_loader)}",
                        total_loss=f"{convE_loss:.4f}"
                    )


                avg_loss = total_loss / len(G2_data_loader)
                accuracy_train = accuracy_score(all_labels, all_preds)
                f1_train = f1_score(all_labels, all_preds, average="macro")
                print(f"\nEpoch {epoch + 1}: train_accuracy = {accuracy_train:.4f}, f1-score_train = {f1_train:.4f}")
                # Loguer la perte de chaque époque dans wandb

                # test_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"],
                #                                  batch_size=config["batch_size"],
                #                                  shuffle=config["shuffle"],
                #                                    input_nodes = data.edge_index[:,removed_edge_indices]).get_loader()

                test_data_loader = NeighborLoader(
                    data,
                    input_nodes=data.edge_index[1][removed_edge_indices],  # Les nœuds que tu veux embeder
                    num_neighbors=config["num_neighbors"],  # Nombre de voisins à échantillonner par couche
                    batch_size=config["batch_size"],
                    shuffle=False
                )

                test_avg_loss, test_accuracy, test_recall, test_precision, test_f1 = evaluate_ConvE(model, data, test_data_loader, test_removed_edges_indices, device, relation_embeddings)


                if "reconstruct_r" in training_options and len(training_options) == 1:
                    wandb.log({"epoch": epoch + 1, "train_loss": avg_loss, "train_accuracy": accuracy_train, "f1_train": f1_train, "test_loss": test_avg_loss,
                               "test_accuracy": test_accuracy,"test_recall": test_recall,
                               "test_precision": test_precision, "test_f1": test_f1})
                if test_f1 > best_F1:
                    best_F1 = test_f1
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best=True)
                    print(f'\nModel saved with Avg Loss: {avg_loss:.4f} , test_F1-Score: {test_f1:.4f}, test_accuracy = {test_accuracy:.4f}')


        else:
            print("Training with X ")
            with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
                for batch in G1_data_loader:
                    batch = batch.to(device)
                    n_id = batch.n_id ## The global node index for every sampled node
                    mask = torch.isin(n_id, batch.input_id) ## mask to get only the embedding of input_id nodes
                    optimizer.zero_grad()
                    embeddings = model.encode(batch)
                    # print(batch)
                    reconstructed_x = model.decode_x(batch, embeddings)
                    reconstructed_x = reconstructed_x[mask]

                    # Calcul de la perte avec conservation de la similarité
                    mse_loss, cos_loss = model.recon_x_loss(data.x[n_id[mask]], reconstructed_x, embeddings[mask])
                 
                    loss = mse_loss +  cos_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_mse_loss += mse_loss.item()
                    total_cos_loss += cos_loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)

                avg_loss = total_loss / len(G1_data_loader)
                avg_mse_loss = total_mse_loss / len(G1_data_loader)
                avg_cos_loss = total_cos_loss / len(G1_data_loader)

            # Loguer la perte de chaque époque dans wandb
                if "Reconstruct_X"  in training_options and len(training_options) == 1:
                    wandb.log({"epoch": epoch + 1, "global loss": avg_loss, "mse_loss":avg_mse_loss,"cos_loss": avg_cos_loss})
            # elif "contrastive" in training_options and len(training_options) == 1:
            #     wandb.log({"epoch": epoch + 1, "Contrastive_loss": avg_loss})
            #


            # Sauvegarde du modèle si la perte est la plus faible
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best=True)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}')


# Boucle d'optimisation des hyperparamètres
for num_bases in hyperparams_grid["num_bases"]:
    for out_channels in hyperparams_grid["out_channels"]:
        config["convE_config"]["embedding_dim"] = out_channels[1]
        config["convE_config"]["hidden_size"] = config["coresp_hidden_sizes"][out_channels[1]]
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
        r_decoder = ConvE(config["convE_config"])

        autoencoder = MC2GEA(RGCN_encoder, RGCN_decoder, r_decoder = r_decoder).to(device)
        # Optimizer

        optimizer = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])

        # Lancer l'entraînement avec les hyperparamètres actuels
        train_with_hyperparams(autoencoder, data, optimizer, config["num_epochs"], num_bases, out_channels)

        # Finir le run dans wandb
        wandb.finish()


