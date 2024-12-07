import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch
from sklearn.model_selection import StratifiedShuffleSplit
def load_gold_standard_labels(gs_path):
    """
    Charge les labels du gold standard à partir d'un fichier Excel.

    :param gs_path: Chemin vers le fichier Excel contenant le GS
    :return: DataFrame avec les colonnes 'term' et 'label'
    """
    # Charger la feuille contenant les termes et labels
    gold_standard_df = pd.read_excel(gs_path, sheet_name='Sheet1')

    # Vérifier que les colonnes nécessaires sont présentes
    if 'term' not in gold_standard_df.columns or 'label' not in gold_standard_df.columns:
        raise ValueError("Le fichier GS doit contenir les colonnes 'term' et 'label'.")

    return gold_standard_df[['term', 'label']]


def save_model(model, optimizer, epoch, save_dir="checkpoints", name = False):
    """
    Saves the model and optimizer state dictionaries.

    Parameters:
    - model (nn.Module): The model to save.
    - optimizer (torch.optim.Optimizer): The optimizer to save.
    - epoch (int): The current epoch number.
    - save_dir (str): Directory where the checkpoint will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define the checkpoint path
    checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")

    if name:
        checkpoint_path = os.path.join(save_dir, f"best_model.pth")


    # Save the model and optimizer states

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    print(f"Model saved at '{checkpoint_path}'")


def load_model_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer states from a checkpoint file.

    Parameters:
    - model (nn.Module): The model to load the state dictionary into.
    - optimizer (torch.optim.Optimizer): The optimizer to load the state dictionary into.
    - checkpoint_path (str): Path to the checkpoint file.

    Returns:
    - model: The model with loaded weights.
    - optimizer: The optimizer with loaded state.
    - start_epoch (int): The epoch to resume training from.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    # Load state dictionaries
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Retrieve the last completed epoch
    start_epoch = checkpoint['epoch']

    print(f"Checkpoint loaded from '{checkpoint_path}', resuming from epoch {start_epoch}")

    return model, optimizer, start_epoch


def generate_relation_embeddings_tensor(relations, embedding_size, device, seed=42):
    """
    Génère un tenseur d'embedding pour des relations uniques.

    Args:
        relations (list): Liste des relations uniques.
        embedding_size (int): Taille des vecteurs d'embedding.
        seed (int): Seed pour garantir la reproductibilité.

    Returns:
        torch.Tensor: Tenseur contenant les embeddings de taille [nb_relations, embedding_size].
    """
    # Fixer le seed pour la reproductibilité
    torch.manual_seed(seed)

    # Nombre de relations uniques
    num_relations = len(relations)

    # Définir une couche d'embedding
    embedding_layer = torch.nn.Embedding(num_relations, embedding_size)

    # Générer les embeddings pour tous les indices
    indices = torch.arange(num_relations)
    embeddings = embedding_layer(indices)  # Shape: [num_relations, embedding_size]

    return embeddings.to(device)





# def removed_edges_train_test_split(indices: torch.Tensor, labels: torch.Tensor, test_size=0.2, random_state=42,output_device="cuda"):
#     """
#     Split GPU tensors into training and testing sets, preserving class balance, with a specified output device.
#
#     Args:
#         indices (torch.Tensor): Tensor of indices (on GPU).
#         labels (torch.Tensor): Tensor of labels (on GPU).
#         test_size (float): Proportion of the dataset to include in the test split.
#         random_state (int): Seed for reproducibility.
#         output_device (str): Device for the output tensors (e.g., "cuda" or "cpu").
#
#     Returns:
#         train_indices (torch.Tensor): Tensor of training indices (on output_device).
#         test_indices (torch.Tensor): Tensor of testing indices (on output_device).
#         train_labels (torch.Tensor): Tensor of training labels (on output_device).
#         test_labels (torch.Tensor): Tensor of testing labels (on output_device).
#     """
#
#     # Move tensors to CPU for compatibility with StratifiedShuffleSplit
#     indices_cpu = indices.cpu().numpy()
#     labels_cpu = labels.cpu().numpy()
#
#     # Initialize StratifiedShuffleSplit
#     splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
#
#     # Perform the split
#     for train_idx, test_idx in splitter.split(indices_cpu, labels_cpu):
#         train_indices = torch.tensor(indices_cpu[train_idx], device=output_device)
#         test_indices = torch.tensor(indices_cpu[test_idx], device=output_device)
#         train_labels = torch.tensor(labels_cpu[train_idx], device=output_device)
#         test_labels = torch.tensor(labels_cpu[test_idx], device=output_device)
#
#     return train_indices, test_indices, train_labels, test_labels
#
#




def removed_edges_train_test_split(indices: torch.Tensor, labels: torch.Tensor, test_size=0.25, random_state=42, output_device="cuda"):
    """
    Split GPU tensors into training and testing sets, preserving class balance, with a specified output device.

    Args:
        indices (torch.Tensor): Tensor of indices (on GPU).
        labels (torch.Tensor): Tensor of labels (on GPU).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.
        output_device (str): Device for the output tensors (e.g., "cuda" or "cpu").

    Returns:
        train_indices (torch.Tensor): Tensor of training indices (on output_device).
        test_indices (torch.Tensor): Tensor of testing indices (on output_device).
        train_labels (torch.Tensor): Tensor of training labels (on output_device).
        test_labels (torch.Tensor): Tensor of testing labels (on output_device).
    """

    # Move tensors to CPU for compatibility with train_test_split
    indices_cpu = indices.cpu().numpy()
    labels_cpu = labels.cpu().numpy()

    # Perform stratified train-test split
    train_idx, test_idx, train_labels_cpu, test_labels_cpu = train_test_split(
        indices_cpu,
        labels_cpu,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_cpu
    )

    # Convert the results back to PyTorch tensors on the specified device
    train_indices = torch.tensor(train_idx, device=output_device)
    test_indices = torch.tensor(test_idx, device=output_device)
    train_labels = torch.tensor(train_labels_cpu, device=output_device)
    test_labels = torch.tensor(test_labels_cpu, device=output_device)

    return train_indices, test_indices, train_labels, test_labels


def save_model_with_hyperparams(model, optimizer, epoch, num_bases, out_channels, save_dir="ckpt",
                                is_best_acc=False):
    os.makedirs(save_dir, exist_ok=True)
    base_filename = f"best_model_bases{num_bases}_channels{'-'.join(map(str, out_channels))}"
    checkpoint_path = os.path.join(save_dir,
                                    f"{base_filename}_best_acc.pth") if is_best_acc else f"{base_filename}_best_loss.pth"


    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_bases': num_bases,
        'out_channels': out_channels
    }, checkpoint_path)
    print(f"Model saved at '{checkpoint_path}'")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False