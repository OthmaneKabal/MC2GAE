import sys
import os

from loss_func import recon_r_loss, sce_loss_fnc, similarity_pair_loss, mse_loss_fnc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'utils')))
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch_geometric.loader import NeighborLoader
from evaluate import evaluate, evaluate_all
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

from config import config
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
def train_model(model, data, optimizer, num_epochs, num_bases, out_channels, gdp,
                           save_dir="ckpt_", training_options = "Reconstruct_X_MSE", device = "cuda", wandb = None, split = False, seed = 42):

    unique_relations = list(set([i.item() for i in data.edge_type]))
    relation_embeddings = generate_relation_embeddings_tensor(unique_relations, out_channels[1], device,
                                                              seed=42)

    best_loss = float('inf')
    best_F1 = 0
    best_accuracy = 0
    # Application du masque de features
    print("\nmask_features...\n")
    masked_features_data= view_partial_features_masking(data)
    print("\nRelations_dripping (Masking)...\n")
    masked_edges_data, removed_edge_indices, removed_edge_types  = relation_based_edge_dropping_balanced(data, config["total_drop_rate"], max_drop_fraction_per_node=0.3, random_seed=42)
    removed_edge_indices = removed_edge_indices.to(device)
    removed_edge_types = removed_edge_types.to(device)

    if split:
        train_removed_edges_indices, test_removed_edges_indices, train_relations, test_relations = removed_edges_train_test_split(removed_edge_indices, removed_edge_types)
    # print(len(removed_edge_indices),"---")
    # removed_edge_indices = train_removed_edges_indices.to(device)
    # test_removed_edges_indices = test_removed_edges_indices.to(device)


    set_seed(seed)
    G1_data_loader = GraphDataLoader(masked_features_data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()
    G_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()
    G2_data_loader = GraphDataLoader(masked_edges_data, num_neighbors=config["num_neighbors"],
                                    batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_mse_loss = 0
        total_cos_loss = 0
        if "contrastive" in training_options and len(training_options)==1:
            with tqdm(total=len(G_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
                for g_batch in G_data_loader:
                    g_batch.to(device)
                    g1_batch = copy.deepcopy(g_batch)
                    g2_batch = copy.deepcopy(g_batch)
                    g1_batch.x = masked_features_data.x[g_batch.n_id]
                    edges_mask = ~torch.isin(g2_batch.e_id, removed_edge_indices.to(device))
                    g2_batch.edge_index = g2_batch.edge_index[:,edges_mask]
                    g2_batch.e_id = g2_batch.e_id[edges_mask]
                    g2_batch.edge_type = g2_batch.edge_type[edges_mask]
                    ################################## Verification ###############################
                    # mask2 = torch.isin(removed_edge_indices, g2_batch.e_id)
                    # # print(removed_edge_types.shape)
                    # # print(mask2.shape, removed_edge_indices.shape)
                    # print(removed_edge_types[mask2].shape)
                    # sorted_g2_batch, i_batch = torch.sort(g2_batch.e_id)
                    # sorted_g2_, i = torch.sort(removed_edge_indices[mask2])
                    # print(g2_batch.edge_type[i_batch] == removed_edge_types[mask2][i])
                    # print(sorted_g2_batch == sorted_g2_)
                    #################################################################################
                    h1_batch = model.encode(g1_batch)
                    h2_batch = model.encode(g2_batch)

                    loss = model.contrastive_loss(h1_batch, h2_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)
                    # H1_projected = model.projector_fc1(H1_batch)[mask_G1]
                    # H2_projected = model.projector_fc2(H2_batch)[mask_G2]

                    # G1_batch.to(device)
                    # G2_batch.to(device)
                    # n_id_1 = G1_batch.n_id  ## The global node index for every sampled node
                    # mask_G1 = torch.isin(n_id_1, G1_batch.input_id) ## mask to get only the embedding of input_id nodes
                    # n_id_2 = G2_batch.n_id
                    # mask_G2 = torch.isin(n_id_2, G2_batch.input_id)
                    # H1_batch = model.encode(G1_batch)
                    # H2_batch = model.encode(G2_batch)
                    # H1_projected = model.projector_fc1(H1_batch)[mask_G1]
                    # H2_projected = model.projector_fc2(H2_batch)[mask_G2]
                    # # Compute contrastive loss
                    # loss = model.contrastive_loss(H1_projected, H2_projected)
                    # loss.backward()
                    # optimizer.step()
                    # total_loss += loss.item()
                    # batch_pbar.set_postfix(batch_loss=loss.item())
                    # batch_pbar.update(1)


        elif "reconstruct_r" in training_options and len(training_options)==1:
            print("Reconstruct R only ....")
            with tqdm(total=len(G_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as main_pbar:
                nb_intersections = 0
                matching_e_ids = []
                all_preds = []
                all_labels = []

                for G2_batch in G_data_loader:

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
                        loss = recon_r_loss(preds, convE_batch[3].to(device))
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


        elif "Reconstruct_X_MSE_Pairs_similarity" in training_options and len(training_options) == 1:
            print("\nReconstruct_X_MSE_Pairs_similarity only...\n")
            with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
                for batch in G1_data_loader:
                    batch = batch.to(device)
                    n_id = batch.n_id ## The global node index for every sampled node
                    mask = torch.isin(n_id, batch.input_id) ## mask to get only the embedding of input_id nodes
                    optimizer.zero_grad()
                    embeddings = model.encode(batch)
                    reconstructed_x = model.decode_x(batch, embeddings)
                    reconstructed_x = reconstructed_x[mask]
                    cos_loss = similarity_pair_loss(data.x[n_id[mask]], reconstructed_x, embeddings[mask])
                    mse_loss = mse_loss_fnc(data.x[n_id[mask]], reconstructed_x)
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
                metrics = evaluate(model, data, config["Gs_path"], config["core_concepts"], gdp)
            # Sauvegarde du modèle si la perte est la plus faible
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,is_best_acc = False)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log(
                    {"epoch": epoch + 1, "global loss": avg_loss, "mse_loss": avg_mse_loss, "cos_loss": avg_cos_loss,
                     "accuracy": metrics["accuracy"], "f1-score": metrics["f1_score"],
                     "recall": metrics["recall"], "precision": metrics["precision"],
                     })

        elif "SCE_Recons_X" in training_options and len(training_options) == 1:
            print("SCE_Recons_X\n")
            with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
                for batch in G1_data_loader:
                    batch = batch.to(device)
                    n_id = batch.n_id  ## The global node index for every sampled node
                    mask = torch.isin(n_id, batch.input_id)  ## mask to get only the embedding of input_id nodes
                    optimizer.zero_grad()
                    embeddings = model.encode(batch)
                    # print(batch)
                    reconstructed_x = model.decode_x(batch, embeddings)
                    reconstructed_x = reconstructed_x[mask]
                    sce_loss = sce_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    loss = sce_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)
                avg_loss = total_loss / len(G1_data_loader)
                print("Evaluation\n")
                metrics = evaluate(model, data, config["Gs_path"], config["core_concepts"], gdp)
                print("\n")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log({"epoch": epoch + 1, "sce loss": avg_loss,
                           "accuracy": metrics["accuracy"], "f1-score": metrics["f1_score"],
                           "recall": metrics["recall"], "precision": metrics["precision"], })

        elif "MSE_Recons_X" in training_options and len(training_options) == 1:
            print("MSE_Recons_X\n")
            with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                      unit="batch") as batch_pbar:
                for batch in G1_data_loader:
                    batch = batch.to(device)
                    n_id = batch.n_id  ## The global node index for every sampled node
                    mask = torch.isin(n_id, batch.input_id)  ## mask to get only the embedding of input_id nodes
                    optimizer.zero_grad()
                    embeddings = model.encode(batch)
                    reconstructed_x = model.decode_x(batch, embeddings)
                    reconstructed_x = reconstructed_x[mask]
                    mce_loss = mse_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    loss = mce_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)
                avg_loss = total_loss / len(G1_data_loader)

                print("Evaluation\n")
                metrics = evaluate(model, data, config["Gs_path"], config["core_concepts"], gdp)
                print("\n")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log({"epoch": epoch + 1, "mce loss": avg_loss,
                           "accuracy": metrics["accuracy"], "f1-score": metrics["f1_score"],
                           "recall": metrics["recall"], "precision": metrics["precision"], })





