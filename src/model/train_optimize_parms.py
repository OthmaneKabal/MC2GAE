import sys
import os

from networkx.algorithms.connectivity import edge_augmentation

from loss_func import recon_r_loss, sce_loss_fnc, similarity_pair_loss, mse_loss_fnc, contrastive_loss, \
    contrastive_loss_exclude_is, calculate_cluster_assignments, inter_cluster_loss, intra_cluster_loss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'utils')))
from src.layers.TransGCNDecoder import TransGCNDecoder
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch_geometric.loader import NeighborLoader
from evaluate import evaluate
from utils.ConvENegativeSampling import generate_negatives, get_positives
from utils.ConvEDataLoader import create_data_loader
from utils.utils import generate_relation_embeddings_tensor, removed_edges_train_test_split, \
    save_model_with_hyperparams, set_seed, save_model, calculate_metrics
from data_augmentation import relation_based_edge_dropping_balanced
from data_augmentation import view_partial_features_masking
from GraphDataLoader import GraphDataLoader
import torch.nn.functional as F
from src.layers.TransGCNEncoder import TransGCNEncoder

import pandas as pd
from config import config
from MRGAE import  MRGAE
from tqdm import tqdm
import wandb
import torch
import random
import numpy as np
seed = 42
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
import torch
import copy
set_seed(42)
torch.backends.cudnn.deterministic = True
import torch_geometric.transforms as T




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

def train_GAE(model, data, optimizer, num_epochs, gdp,save_file,
                           save_dir="GAE", device = "cuda", wandb = None, seed = 42):

    best_loss = float('inf')
    best_accuracy = 0
    best_metrics = {}
    set_seed(seed)
    G_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"],
                                    batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()
    G_data_loader.edge_attr = data.edge_attr
    total_loss = 0
    transform_directed_without_split = T.Compose([
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=False,
                          split_labels=True, add_negative_train_samples=True),
    ])
    print("Negative_sampling ....\n")
    train_data_directed_without_split, val_data_directed_without_split, test_data_directed_without_split = transform_directed_without_split(
        data)

    for epoch in range(num_epochs):
        model.train()

        with tqdm(total=len(G_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
            if  isinstance(model.encoder, TransGCNEncoder):
                z,_ = model.encode(data)
            else:
                z = model.encode(data)
            loss = model.recon_loss(z, train_data_directed_without_split.pos_edge_label_index)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_pbar.set_postfix(batch_loss=loss.item())
            batch_pbar.update(1)
            avg_loss = total_loss / len(G_data_loader)
            print("\nEvaluation\n")
            print(data)
            # metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
            metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp, config)

            print("\n")
            print(metrics)
            print("\n")
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model, optimizer, epoch, save_dir=save_dir, file_name = save_file, is_best_acc=False)


                print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                save_model(model, optimizer, epoch, save_dir=save_dir, file_name=save_file, is_best_acc=False)

                print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                best_metrics = metrics
            wandb.log({"epoch": epoch + 1, "loss": avg_loss,
                       "accuracy": metrics["accuracy"], "f1-score": metrics["f1_score"],
                       "recall": metrics["recall"], "precision": metrics["precision"], })
    best_metrics["exp_name"] = save_file
    return best_metrics




def train_DisMult(model, data, optimizer,num_epochs,gdp, save_file,device,
                           save_dir="train_R_reconstruction", wandb = None, seed = 42):
    best_loss = float('inf')
    best_F1 = 0
    best_accuracy = 0
    best_metrics = {}
    best_epoch = 0
    print("\nRelations_dripping (Masking)...\n")
    masked_edges_data, removed_edge_indices, removed_edge_types = relation_based_edge_dropping_balanced(data, config[
        "total_drop_rate"], max_drop_fraction_per_node=0.3, random_seed=42)
    removed_edge_indices = removed_edge_indices.to(device)
    removed_edge_types = removed_edge_types.to(device)

    set_seed(seed)
    G_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"],
                                    batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()
    G_data_loader.edge_attr = data.edge_attr
    set_seed(seed)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_true_labels = []
        with tqdm(total=len(G_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as main_pbar:

            for G2_batch in G_data_loader:

                G2_batch = G2_batch.to(device)
                removed_batch = copy.copy(G2_batch)

                removed_edge_indices = removed_edge_indices.to(device)
                mask = torch.isin(removed_edge_indices, G2_batch.e_id)
                intersections = removed_edge_indices[mask]
                # Obtenir les nœuds cibles des arêtes intersectantes avec le batch graph
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
                pos_edge_index = torch.stack((all_positive_triplets[:, 0], all_positive_triplets[:, 2]))  # (2, num_edges)
                pos_edge_type = all_positive_triplets[:, 1]
                neg_edge_index = torch.stack((all_negative_triplets[:, 0], all_negative_triplets[:, 2]))  # (2, num_edges)
                neg_edge_type = all_negative_triplets[:, 1]
                # Scores pour les triplets positifs et négatifs
                pos_scores = model.recon_r_(H_2, pos_edge_index, pos_edge_type)
                neg_scores = model.recon_r_(H_2, neg_edge_index, neg_edge_type)
                # Fonction de perte : Binary Cross Entropy

                pos_preds = (torch.sigmoid(pos_scores) > 0.55).int()
                neg_preds = (torch.sigmoid(neg_scores) > 0.55).int()

                # True labels
                pos_labels = torch.ones_like(pos_preds)
                neg_labels = torch.zeros_like(neg_preds)

                # Collect predictions and true labels
                all_preds.extend(pos_preds.cpu().numpy())
                all_preds.extend(neg_preds.cpu().numpy())
                all_true_labels.extend(pos_labels.cpu().numpy())
                all_true_labels.extend(neg_labels.cpu().numpy())
                loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
                       F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                main_pbar.update(1)


            avg_loss = total_loss / len(G_data_loader)
            print("Evaluation\n")
            # metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
            metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp, config)

            print("\n")
            print(metrics)

            R_accuracy, R_precision, R_recall, R_f1 = calculate_metrics(all_preds, all_true_labels)
            print(f"R_accuracy: {R_accuracy}, R_precision: {R_precision}, R_recall: {R_recall},R_f1: {R_f1}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model, optimizer, epoch, save_dir = save_dir, file_name = save_file, is_best_acc=False)
                print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_metrics = metrics
                save_model(model, optimizer, epoch, save_dir=save_dir,file_name= save_file , is_best_acc=True)
                print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
            wandb.log({"epoch": epoch + 1, "loss": avg_loss,
                       "accuracy": metrics["accuracy"], "f1-score": metrics["f1_score"],
                       "recall": metrics["recall"], "precision": metrics["precision"],
                       "R_accuracy": R_accuracy, "R_precision": R_precision,
                       "R_recall": R_recall, "R_f1": R_f1,})

    best_metrics["exp_name"] = save_file
    return best_metrics



def train_X_reconstruction(model, data ,optimizer, num_epochs, gdp, save_file,device, config,loss_fct = ["MSE"],
                           save_dir="train_X_reconstruction", wandb = None, seed = 42):


    best_loss = float('inf')
    best_F1 = 0
    best_accuracy = 0
    best_metrics = {}
    best_epoch = 0
    print("\nmask_features...\n")
    masked_features_data = view_partial_features_masking(data, max_masking_percentage=config["max_masking_percentage"])
    set_seed(seed)
    G1_data_loader = GraphDataLoader(masked_features_data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()
    G1_data_loader.edge_attr = masked_features_data.edge_attr
    set_seed(seed)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        if "MSE" in loss_fct:
            total_mse_loss = 0
        if "PCSE" in loss_fct:
            total_cos_loss = 0
        if "SCE" in loss_fct:
            total_sce_loss = 0

        print("\nMSE_Recons_X\n")
        with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}",
                  unit="batch") as batch_pbar:
            for batch in G1_data_loader:

                batch = batch.to(device)
                n_id = batch.n_id  ## The global node index for every sampled node
                mask = torch.isin(n_id, batch.input_id)  ## mask to get only the embedding of input_id nodes
                optimizer.zero_grad()
                r_embd = None
                if isinstance(model.encoder, TransGCNEncoder):
                    embeddings, r_embd = model.encode(batch)

                else:
                    embeddings = model.encode(batch)

                # embeddings = model.encode(batch)
                if isinstance(model.x_decoder, TransGCNDecoder):

                    reconstructed_x = model.decode_x(batch, embeddings, r_embd)
                else:
                    reconstructed_x = model.decode_x(batch, embeddings)

                reconstructed_x = reconstructed_x[mask]

                total_loss = 0.0

                # Vérifier chaque terme et ajouter le loss correspondant au total
                if "MSE" in loss_fct:
                    mse_loss = mse_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    total_loss += mse_loss

                if "PCSE" in loss_fct:
                    pcse_loss = similarity_pair_loss(data.x[n_id[mask]], reconstructed_x, embeddings[mask])
                    total_loss += pcse_loss

                if "SCE" in loss_fct:
                    sce_loss = sce_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    total_loss += sce_loss

                loss = total_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_pbar.set_postfix(batch_loss=loss.item())
                batch_pbar.update(1)
            avg_loss = total_loss / len(G1_data_loader)

            print("Evaluation\n")
            metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp, config)
            print("\n")
            print(metrics)
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model, optimizer, epoch, save_dir = save_dir, file_name = save_file, is_best_acc=False)

                print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_metrics = metrics
                # best_epoch = epoch
                # save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                #                             is_best_acc=True)
                save_model(model, optimizer, epoch, save_dir=save_dir,file_name= save_file , is_best_acc=True)
                print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
            wandb.log({"epoch": epoch + 1, "mce loss": avg_loss,
                       "accuracy": metrics["accuracy"], "f1-score": metrics["f1_score"],
                       "recall": metrics["recall"], "precision": metrics["precision"], })

    best_metrics["exp_name"] = save_file
    return best_metrics


def train_Contrastive(model, data, optimizer, num_epochs, gdp, save_file,
                      device="cuda", save_dir="contrastive_training",
                      wandb=None, seed=42):
    import copy
    set_seed(seed)
    best_loss = float('inf')
    best_accuracy = 0
    best_metrics = {}

    print("\n--- Preparing views for contrastive learning ---\n")
    masked_features_data = view_partial_features_masking(data, max_masking_percentage=config["max_masking_percentage"])
    masked_edges_data, removed_edge_indices, _ = relation_based_edge_dropping_balanced(
        data, config["total_drop_rate"], max_drop_fraction_per_node=0.3, random_seed=42
    )

    removed_edge_indices = removed_edge_indices.to(device)

    G_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"],
                                    batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        with tqdm(total=len(G_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for batch in G_data_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                # View 1: masking features
                view_1 = copy.deepcopy(batch)
                view_1.x = masked_features_data.x[view_1.n_id]

                # View 2: masking edges
                view_2 = copy.deepcopy(batch)

                edge_mask = ~torch.isin(view_2.e_id, removed_edge_indices)
                view_2.edge_index = view_2.edge_index[:, edge_mask]
                view_2.edge_type = view_2.edge_type[edge_mask]
                view_2.edge_attr = view_2.edge_attr[edge_mask]
                view_2.e_id = view_2.e_id[edge_mask]

                # Masquer les input_id
                mask_nodes = torch.isin(batch.n_id, batch.input_id)

                # Encodage + projection (cas TransGCNEncoder ou non)
                if isinstance(model.encoder, TransGCNEncoder):
                    h1, _ = model.encoder(view_1)
                    h2, _ = model.encoder(view_2)
                else:
                    h1 = model.encode(view_1)
                    h2 = model.encode(view_2)


                ##
                if not isinstance(mask_nodes, torch.Tensor):
                    mask_nodes = torch.tensor(mask_nodes)

                # Si mask_nodes est un masque booléen
                if mask_nodes.dtype == torch.bool:
                    mask_nodes = mask_nodes.to(h1.device)

                # Sinon on suppose que c'est une liste d'indices
                else:
                    mask_nodes = mask_nodes.long().to(h1.device)

                ###

                # Appliquer les projecteurs
                z1 = model.projector_fc1(h1[mask_nodes])
                z2 = model.projector_fc2(h2[mask_nodes])

                # Calcul de la perte contrastive standard
                c_loss = contrastive_loss(z1, z2)

                c_loss.backward()
                optimizer.step()
                total_loss += c_loss.item()

                pbar.set_postfix(loss=c_loss.item())
                pbar.update(1)

        avg_loss = total_loss / len(G_data_loader)

        print("\n--- Evaluation ---")
        metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp, config)
        print(metrics)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, optimizer, epoch, save_dir=save_dir, file_name=save_file, is_best_acc=False)
            print(f"Model saved with lowest contrastive loss: {best_loss:.4f}")

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_metrics = metrics
            save_model(model, optimizer, epoch, save_dir=save_dir, file_name=save_file, is_best_acc=True)
            print(f"Model saved with best accuracy: {best_accuracy:.4f}")

        if wandb is not None:
            wandb.log({
                "epoch": epoch + 1,
                "contrastive_loss": avg_loss,
                "accuracy": metrics["accuracy"],
                "f1-score": metrics["f1_score"],
                "recall": metrics["recall"],
                "precision": metrics["precision"]
            })

    best_metrics["exp_name"] = save_file
    return best_metrics


def train_Double_Reconstruction(model, data, optimizer,num_epochs,gdp, save_file,device, loss_fct = ["MSE"],
                           save_dir="train_R_reconstruction", wandb = None, seed = 42):
    best_loss = float('inf')
    best_F1 = 0
    best_accuracy = 0
    best_metrics = {}
    best_epoch = 0
    print("\nRelations_dripping (Masking)...\n")
    masked_features_data = view_partial_features_masking(data, max_masking_percentage=config["max_masking_percentage"])
    masked_edges_data, removed_edge_indices, removed_edge_types = relation_based_edge_dropping_balanced(data, config[
        "total_drop_rate"], max_drop_fraction_per_node=0.3, random_seed=42)
    removed_edge_indices = removed_edge_indices.to(device)
    removed_edge_types = removed_edge_types.to(device)

    set_seed(seed)
    G_data_loader = GraphDataLoader(masked_features_data, num_neighbors=config["num_neighbors"],
                                    batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()
    G_data_loader.data.edge_attr = data.edge_attr
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_Recons_X_loss = 0
        total_R_loss = 0
        all_preds = []
        all_true_labels = []
        with tqdm(total=len(G_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as main_pbar:

            for G2_batch in G_data_loader:

                G2_batch = G2_batch.to(device)
                removed_batch = copy.copy(G2_batch)

                removed_edge_indices = removed_edge_indices.to(device)
                mask = torch.isin(removed_edge_indices, G2_batch.e_id)
                intersections = removed_edge_indices[mask]
                # Obtenir les nœuds cibles des arêtes intersectantes avec le batch graph
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

                #### Features reconstruction
                n_id_fm = G2_batch.n_id  ## The global node index for every sampled node
                mask_fm = torch.isin(n_id_fm, G2_batch.input_id)  ## mask to get only the embedding of input_id nodes
                optimizer.zero_grad()
                H_2 = model.encode(G2_batch)
                reconstructed_x = model.decode_x(G2_batch, H_2)
                reconstructed_x = reconstructed_x[mask_fm]
                ##############################
                Recons_X_loss = 0.0

                # Vérifier chaque terme et ajouter le loss correspondant au total
                if "MSE" in loss_fct:
                    mse_loss = mse_loss_fnc(data.x[G2_batch.n_id[mask_fm]], reconstructed_x)
                    Recons_X_loss += mse_loss

                if "PCSE" in loss_fct:
                    pcse_loss = similarity_pair_loss(data.x[G2_batch.n_id[mask_fm]], reconstructed_x)
                    Recons_X_loss += pcse_loss

                if "SCE" in loss_fct:
                    sce_loss = sce_loss_fnc(data.x[G2_batch.n_id[mask_fm]], reconstructed_x)
                    Recons_X_loss += sce_loss




                # Générer les triplets négatifs et positifs
                negative_triplets = generate_negatives(data, G2_batch, negative_ratio=1)
                positive_triplets = get_positives(G2_batch)
                ## Generate negative examples from removed edges:
                negative_triplets_removed = generate_negatives(data, removed_batch, negative_ratio=1)
                positive_triplets_removed = get_positives(removed_batch)

                all_positive_triplets = torch.cat((positive_triplets, positive_triplets_removed), dim=0)
                all_negative_triplets = torch.cat((negative_triplets, negative_triplets_removed), dim=0)
                pos_edge_index = torch.stack((all_positive_triplets[:, 0], all_positive_triplets[:, 2]))  # (2, num_edges)
                pos_edge_type = all_positive_triplets[:, 1]
                neg_edge_index = torch.stack((all_negative_triplets[:, 0], all_negative_triplets[:, 2]))  # (2, num_edges)
                neg_edge_type = all_negative_triplets[:, 1]
                # Scores pour les triplets positifs et négatifs
                pos_scores = model.recon_r_(H_2, pos_edge_index, pos_edge_type)
                neg_scores = model.recon_r_(H_2, neg_edge_index, neg_edge_type)
                # Fonction de perte : Binary Cross Entropy

                pos_preds = (torch.sigmoid(pos_scores) > 0.55).int()
                neg_preds = (torch.sigmoid(neg_scores) > 0.55).int()

                # True labels
                pos_labels = torch.ones_like(pos_preds)
                neg_labels = torch.zeros_like(neg_preds)

                # Collect predictions and true labels
                all_preds.extend(pos_preds.cpu().numpy())
                all_preds.extend(neg_preds.cpu().numpy())
                all_true_labels.extend(pos_labels.cpu().numpy())
                all_true_labels.extend(neg_labels.cpu().numpy())
                loss_R = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) + \
                       F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))

                loss = Recons_X_loss + loss_R

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_R_loss += loss_R.item()
                total_Recons_X_loss += Recons_X_loss
                main_pbar.update(1)

            avg_R_loss = total_R_loss / len(G_data_loader)
            avg_Recons_X_loss = total_Recons_X_loss / len(G_data_loader)
            avg_loss = total_loss / len(G_data_loader)
            print("Evaluation\n")
            # metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
            metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp, config)

            print("\n")
            print(metrics)

            R_accuracy, R_precision, R_recall, R_f1 = calculate_metrics(all_preds, all_true_labels)
            print(f"R_accuracy: {R_accuracy}, R_precision: {R_precision}, R_recall: {R_recall},R_f1: {R_f1}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model, optimizer, epoch, save_dir = save_dir, file_name = save_file, is_best_acc=False)
                print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_metrics = metrics
                save_model(model, optimizer, epoch, save_dir=save_dir,file_name= save_file , is_best_acc=True)
                print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
            wandb.log({"epoch": epoch + 1, "total_loss": avg_loss,
                       "accuracy": metrics["accuracy"], "f1-score": metrics["f1_score"],
                       "recall": metrics["recall"], "precision": metrics["precision"],
                       "R_accuracy": R_accuracy, "R_precision": R_precision,
                       "R_recall": R_recall, "R_f1": R_f1, "R_loss": avg_R_loss, "X_loss" : avg_Recons_X_loss})

    best_metrics["exp_name"] = save_file
    return best_metrics












# Fonction d'entraînement avec suivi de la perte dans wandb
def train_model(model, data, optimizer, num_epochs, num_bases, out_channels, gdp,
                           save_dir="ckpt_", training_options = "Reconstruct_X_MSE", device = "cuda", wandb = None, split = False, seed = 42):

    unique_relations = list(set([i.item() for i in data.edge_type]))
    relation_embeddings = generate_relation_embeddings_tensor(unique_relations, out_channels[-1], device,
                                                              seed=42)

    best_loss = float('inf')
    best_F1 = 0
    best_accuracy = 0
    # Application du masque de features
    print("\nmask_features...\n")
    masked_features_data= view_partial_features_masking(data, max_masking_percentage = config["max_masking_percentage"])
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

    set_seed(seed)
    G_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()
    set_seed(seed)
    G2_data_loader = GraphDataLoader(masked_edges_data, num_neighbors=config["num_neighbors"],
                                     batch_size=config["batch_size"], shuffle=config["shuffle"]).get_loader()
    set_seed(seed)
    cc_indexes = gdp.get_list_indexes(config["core_concepts"])
    cc_data_loader = GraphDataLoader(data, num_neighbors=config["num_neighbors"], batch_size=len(cc_indexes),
                                     shuffle=config["shuffle"], input_nodes = cc_indexes).get_loader()
    gs_terms = pd.read_excel(config["Gs_path_no_other"], sheet_name='Sheet1')
    gs_terms_indexes = gdp.get_list_indexes(list(gs_terms['term']))

    # cc_data_loader_2 = GraphDataLoader(data, num_neighbors=config["num_neighbors"],batch_size = len(cc_indexes), shuffle=config["shuffle"]).get_loader()
    #
    # cc_graph = next(iter(cc_data_loader))
    # cc_graph_2 = next(iter(cc_data_loader_2))
    #
    # cc_clusters = calculate_cluster_assignments(cc_graph.x[cc_graph.input_id], cc_graph_2.x[cc_graph_2.input_id])
    # print(cc_clusters)
    # inter_cluster_loss(cc_graph.x[range(9,18)], cc_clusters,cc_graph_2.x[cc_graph_2.input_id] )
    # exit(55)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_mse_loss = 0
        total_cos_loss = 0
        total_sce_loss = 0
        total_inter_cluster_loss_ = 0
        total_intra_cluster_loss_ = 0
        if "contrastive" in training_options and len(training_options)==1:
            print("\nContrastive\n")
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
                    nodes_mask = torch.isin(g_batch.n_id, g_batch.input_id)
                    h1_batch = model.encode(g1_batch)
                    h2_batch = model.encode(g2_batch)
                    # mask = torch.isin(n_id, batch.input_id) ## select only input_nodes
                    # h1_projected = model.projector_fc1(h1_batch)[mask]
                    # h2_projected = model.projector_fc2(h1_batch)[mask]

                    mask_is = g1_batch.edge_type == gdp.predicate_to_id["is"]
                    c_loss = contrastive_loss_exclude_is(h1_batch,h2_batch,g1_batch.edge_index,mask_is)


                    # exit(12354)

                    # n_id = g1_batch.n_id  ## The global node index for every sampled node
                    # mask_1 = torch.isin(n_id, g1_batch.input_id)
                    # c_loss = contrastive_loss(h1_batch, h2_batch)
                    # reconstructed_x = model.decode_x(g1_batch, h1_batch)
                    # reconstructed_x = reconstructed_x[mask_1]
                    # mce_loss = mse_loss_fnc(data.x[n_id[mask_1]], reconstructed_x)
                    loss = c_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)
                avg_loss = total_loss / len(G1_data_loader)
                print("\nEvaluation\n")
                print(data)
                metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
                print("\n")
                print(metrics)
                print("\n")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=True)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log({"epoch": epoch + 1, "contrastive loss": avg_loss,
                           "accuracy": metrics["accuracy"], "f1-score": metrics["f1_score"],
                           "recall": metrics["recall"], "precision": metrics["precision"], })


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
                    # Obtenir les nœuds cibles des arêtes intersectantes avec le batch graph
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
                metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
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
                metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
                print("\n")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=True)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log({"epoch": epoch + 1, "sce loss": avg_loss,
                           "accuracy": metrics["accuracy"], "f1-score": metrics["f1_score"],
                           "recall": metrics["recall"], "precision": metrics["precision"], })

        elif "MSE_Recons_X" in training_options and len(training_options) == 1:
            print("\nMSE_Recons_X\n")
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
                    mse_loss = mse_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    loss = mse_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)
                avg_loss = total_loss / len(G1_data_loader)

                print("Evaluation\n")
                metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
                print("\n")
                print(metrics)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=True)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log({"epoch": epoch + 1, "mce loss": avg_loss,
                           "accuracy": metrics["accuracy"], "f1-score": metrics["f1_score"],
                           "recall": metrics["recall"], "precision": metrics["precision"], })


        elif "MSE_Recons_X"  in training_options and "SCE_Recons_X" in training_options:
            print("\nMSE_Recons_X + SCE_Recons_X\n")
            with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
                for batch in G1_data_loader:
                    batch = batch.to(device)
                    n_id = batch.n_id  ## The global node index for every sampled node
                    mask = torch.isin(n_id, batch.input_id)  ## mask to get only the embedding of input_id nodes
                    optimizer.zero_grad()
                    embeddings = model.encode(batch)
                    reconstructed_x = model.decode_x(batch, embeddings)
                    reconstructed_x = reconstructed_x[mask]
                    sce_loss = sce_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    mse_loss = mse_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    loss = mse_loss + sce_loss
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_mse_loss += mse_loss.item()
                    total_sce_loss += sce_loss.item()
                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)

                avg_loss = total_loss / len(G1_data_loader)
                avg_mse_loss = total_mse_loss / len(G1_data_loader)
                avg_sce_loss = total_sce_loss / len(G1_data_loader)
                metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
                # Sauvegarde du modèle si la perte est la plus faible
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=True)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log(
                    {"epoch": epoch + 1, "global loss": avg_loss, "mse_loss": avg_mse_loss, "sce_loss": total_sce_loss,
                     "accuracy": metrics["accuracy"], "f1-score": metrics["f1_score"],
                     "recall": metrics["recall"], "precision": metrics["precision"],
                     })

        elif "MSE_Recons_X"  in training_options and "clustering_obj" in training_options:
            warm_up_epochs = 15
            gradual_introduction_epochs = 50
            print("\nMSE_Recons_X + clustering_obj\n")
            intra_drap = True
            with tqdm(total=len(G1_data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as batch_pbar:
                for batch in G1_data_loader:
                    batch = batch.to(device)
                    cc_graph = next(iter(cc_data_loader)).to(device)
                    n_id = batch.n_id  ## The global node index for every sampled node
                    mask = torch.isin(n_id, batch.input_id) ## mask to get only the embedding of input_id nodes
                    optimizer.zero_grad()
                    cc_embeddings = model.encode(cc_graph)[cc_graph.input_id]
                    embeddings = model.encode(batch)
                    reconstructed_x = model.decode_x(batch, embeddings)
                    reconstructed_x = reconstructed_x[mask]
                    mse_loss = mse_loss_fnc(data.x[n_id[mask]], reconstructed_x)
                    ####### clustering loss ######"
                    if epoch < warm_up_epochs:
                        # Pendant la période de warm-up, seule la reconstruction MSE est optimisée
                        loss = mse_loss
                        total_inter_cluster_loss_ = 0
                        total_intra_cluster_loss_ = 0

                    else:
                        # Calcul du coefficient d'introduction progressive
                        clustering_weight = min(1.0, (epoch - warm_up_epochs) / gradual_introduction_epochs)
                        # Calcul des losses de clustering
                        mse_loss = torch.tensor(0)
                        gs_mask = torch.isin(batch.input_id, torch.tensor(gs_terms_indexes).to(device))
                        if sum(gs_mask) == 0:
                            inter_cluster_loss_ = torch.tensor(0)
                        else:
                            gs_batch_indexes = batch.input_id[gs_mask]
                            gs_mask_embd = torch.isin(batch.n_id, gs_batch_indexes)
                            cluster_assignments = calculate_cluster_assignments(embeddings[gs_mask_embd], cc_embeddings)

                            inter_cluster_loss_ = inter_cluster_loss(embeddings[gs_mask_embd], cluster_assignments, cc_embeddings)
                        if intra_drap:
                            intra_cluster_loss_ = intra_cluster_loss(cc_embeddings)
                            intra_drap = False
                        else:
                            intra_cluster_loss_ = torch.tensor(0.0, device=device, requires_grad=True)
                        # Combinaison des losses avec un poids progressif
                        loss = mse_loss + clustering_weight * (intra_cluster_loss_ + inter_cluster_loss_)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_mse_loss += mse_loss.item()

                    # Gestion des pertes de clustering (uniquement après la période de warm-up)
                    if epoch >= warm_up_epochs:
                        total_inter_cluster_loss_ += inter_cluster_loss_.item()
                        total_intra_cluster_loss_ += intra_cluster_loss_.item()

                    batch_pbar.set_postfix(batch_loss=loss.item())
                    batch_pbar.update(1)

                avg_loss = total_loss / len(G1_data_loader)

                avg_mse_loss = total_mse_loss / len(G1_data_loader)
                avg_inter_cluster_loss = total_inter_cluster_loss_ / len(G1_data_loader)
                avg_intra_cluster_loss = total_intra_cluster_loss_ / len(G1_data_loader)
                print("\nEvaluation:")
                metrics = evaluate(model, data, config["Gs_path_no_other"], config["core_concepts"], gdp)
                print("\n", metrics,"\n")
                print(f"\n Loss: total:{avg_loss}, mse:{avg_mse_loss},inter:{avg_inter_cluster_loss},intra:{avg_intra_cluster_loss} \n")
                # Sauvegarde du modèle si la perte est la plus faible
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=False)
                    print(f'Model saved with Avg Loss: {best_loss:.4f}\n')
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir=save_dir,
                                                is_best_acc=True)
                    print(f'Model saved with Accuracy: {best_accuracy:.4f}\n')
                wandb.log(
                    {"epoch": epoch + 1, "global loss": avg_loss, "mse_loss": avg_mse_loss, "intra_cluster_loss": avg_intra_cluster_loss,
                     "inter_cluster_loss": avg_inter_cluster_loss, "accuracy": metrics["accuracy"], "f1-score": metrics["f1_score"],
                     "Recall": metrics["recall"], "precision": metrics["precision"],
                     })




